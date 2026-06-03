"""
Phase 3 — filter cluster lead-lag candidates.

Pipeline (applied per (signal, horizon) slice of lag_corr_long):
  1. For each ordered pair (i -> j), keep peak |corr| at non-zero lag.
  2. Significance: circular-shuffle null (N permutations of follower series within day).
     Drop pairs below 95th-pct null peak |corr|.
  3. Direction asymmetry: keep pair only if |corr(i->j,k*)| > 1.3 * |corr(j->i,k*)|.
  4. Common-driver removal: residualize i and j against mean of other 8 clusters
     at time t; recompute lag-corr at k*. Drop if residual |corr| < 0.5 * raw.
  5. Half-sample stability: split rows in half (by day), recompute lag-corr at k*.
     Keep pairs whose half-1 and half-2 corr have same sign and |corr| >= 0.4 * full.

Inputs:
  - data/derived/cluster_signals.parquet
  - data/derived/lag_corr_long.parquet

Outputs:
  - data/derived/candidates_raw.parquet
  - data/derived/candidates_filtered.parquet     (post all gates)
  - data/derived/null_peak.parquet               (null distribution)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:/Users/Utente/OneDrive/Desktop/IMC_prosperity/Github_IMC/ROUND_5")
DER = ROOT / "data" / "derived"

N_PERM = 200
RNG = np.random.default_rng(42)
TARGETS = [("R", "slow"), ("V", "fast")]   # primary leadership signals


def lag_corr_pair(x: pd.Series, y: pd.Series, k: int) -> float:
    """corr(x(t), y(t+k)) with per-day shift."""
    ys = y.groupby(level="day").shift(-k)
    m = x.notna() & ys.notna()
    if m.sum() < 100:
        return np.nan
    a = x[m].values
    b = ys[m].values
    a = (a - a.mean()) / a.std(ddof=0)
    b = (b - b.mean()) / b.std(ddof=0)
    return float((a * b).mean())


def circular_shift_per_day(s: pd.Series, rng: np.random.Generator) -> pd.Series:
    """Circular-shift the series within each day by a random offset."""
    out = s.copy()
    for d, idx in s.groupby(level="day").groups.items():
        n = len(idx)
        off = int(rng.integers(low=1, high=n))
        out.loc[idx] = np.roll(s.loc[idx].values, off)
    return out


def null_peak_dist(x: pd.Series, y: pd.Series, lags: list[int],
                   n_perm: int, rng: np.random.Generator) -> np.ndarray:
    peaks = np.empty(n_perm)
    for p in range(n_perm):
        ys = circular_shift_per_day(y, rng)
        cs = [abs(lag_corr_pair(x, ys, k)) for k in lags if k != 0]
        peaks[p] = np.nanmax(cs)
    return peaks


def main() -> None:
    sig = pd.read_parquet(DER / "cluster_signals.parquet")
    long = pd.read_parquet(DER / "lag_corr_long.parquet")

    R = sig["R"]
    V = sig["V"]

    # slow-aggregated R rebuild (must match phase2b)
    AGG = 50
    Rtmp = R.copy()
    Rtmp["bar"] = Rtmp.groupby(level="day").cumcount() // AGG
    R_slow = (Rtmp.reset_index()
                  .groupby(["day", "bar"])
                  .sum(numeric_only=True)
                  .drop(columns=["timestamp"], errors="ignore"))
    R_slow.index = R_slow.index.set_names(["day", "timestamp"])

    series_for = {("R", "slow"): R_slow, ("V", "fast"): V}

    raw_rows = []
    null_rows = []
    keep_rows = []

    for key in TARGETS:
        signal, horizon = key
        X = series_for[key]
        sub = long[(long["signal"] == signal) & (long["horizon"] == horizon)]
        lags = sorted(sub["lag"].unique().tolist())
        nz_lags = [k for k in lags if k != 0]
        clusters = sorted(sub["c_i"].unique())

        # 1) peak per ordered pair
        nz = sub[sub["lag"] != 0].copy()
        nz["abscorr"] = nz["corr"].abs()
        idx = nz.groupby(["c_i", "c_j"])["abscorr"].idxmax()
        peak = nz.loc[idx, ["c_i", "c_j", "lag", "corr", "abscorr"]].copy()
        peak["signal"] = signal
        peak["horizon"] = horizon
        raw_rows.append(peak)

        # 2) null per pair (only compute for top-30 candidates; cheap and enough)
        peak = peak.sort_values("abscorr", ascending=False).reset_index(drop=True)
        topN = peak.head(30).copy()

        for _, row in topN.iterrows():
            ci, cj, k, c = row["c_i"], row["c_j"], int(row["lag"]), row["corr"]
            if ci == cj:
                continue
            x = X[ci].dropna()
            y = X[cj].dropna()
            null = null_peak_dist(x, y, nz_lags, N_PERM, RNG)
            p95 = float(np.quantile(null, 0.95))
            null_rows.append({"signal": signal, "horizon": horizon,
                              "c_i": ci, "c_j": cj, "p95": p95,
                              "null_mean": float(null.mean()),
                              "null_max": float(null.max())})
            sig_pass = abs(c) > p95

            # 3) direction asymmetry
            c_rev = lag_corr_pair(X[cj], X[ci], k)
            asym_pass = abs(c) > 1.3 * abs(c_rev) if not np.isnan(c_rev) else False

            # 4) common-driver: mean of OTHER clusters
            others = [u for u in clusters if u not in (ci, cj)]
            common = X[others].mean(axis=1)
            xi_res = X[ci] - lin_proj(X[ci], common)
            xj_res = X[cj] - lin_proj(X[cj], common)
            c_res = lag_corr_pair(xi_res, xj_res, k)
            cd_pass = (abs(c_res) >= 0.5 * abs(c)) if not np.isnan(c_res) else False

            # 5) half-sample stability (split by ROW count)
            n = len(X)
            half1 = X.iloc[: n // 2]
            half2 = X.iloc[n // 2 :]
            c1 = lag_corr_pair(half1[ci], half1[cj], k)
            c2 = lag_corr_pair(half2[ci], half2[cj], k)
            stab_pass = (
                not np.isnan(c1) and not np.isnan(c2)
                and np.sign(c1) == np.sign(c2) == np.sign(c)
                and min(abs(c1), abs(c2)) >= 0.4 * abs(c)
            )

            keep_rows.append({
                "signal": signal, "horizon": horizon,
                "leader": ci, "follower": cj, "lag": k,
                "corr": c, "corr_reverse": c_rev,
                "corr_residual": c_res,
                "corr_half1": c1, "corr_half2": c2,
                "p95_null": p95,
                "pass_sig": bool(sig_pass),
                "pass_asym": bool(asym_pass),
                "pass_cd": bool(cd_pass),
                "pass_stab": bool(stab_pass),
            })

    raw = pd.concat(raw_rows, ignore_index=True)
    raw.to_parquet(DER / "candidates_raw.parquet")
    pd.DataFrame(null_rows).to_parquet(DER / "null_peak.parquet")

    cands = pd.DataFrame(keep_rows)
    cands["pass_all"] = cands[["pass_sig", "pass_asym", "pass_cd", "pass_stab"]].all(axis=1)
    cands = cands.sort_values(["pass_all", "corr"],
                              key=lambda s: s.abs() if s.name == "corr" else s,
                              ascending=[False, False])
    cands.to_parquet(DER / "candidates_filtered.parquet")

    print("FILTERED CANDIDATES (sorted, pass_all on top):")
    cols = ["signal", "horizon", "leader", "follower", "lag",
            "corr", "corr_reverse", "corr_residual",
            "corr_half1", "corr_half2",
            "pass_sig", "pass_asym", "pass_cd", "pass_stab", "pass_all"]
    print(cands[cols].head(30).to_string(index=False))

    print(f"\nsurvivors: {cands['pass_all'].sum()} / {len(cands)}")
    print(f"saved -> {DER}")


def lin_proj(y: pd.Series, x: pd.Series) -> pd.Series:
    """Return projection of y on x (i.e. fitted values from OLS y ~ x)."""
    m = y.notna() & x.notna()
    if m.sum() < 10:
        return pd.Series(0.0, index=y.index)
    yy = y[m].values
    xx = x[m].values
    xx = xx - xx.mean()
    yy = yy - yy.mean()
    beta = (xx * yy).sum() / (xx * xx).sum()
    proj = pd.Series(np.nan, index=y.index)
    proj[m] = beta * (x[m].values - x[m].mean())
    return proj


if __name__ == "__main__":
    main()
