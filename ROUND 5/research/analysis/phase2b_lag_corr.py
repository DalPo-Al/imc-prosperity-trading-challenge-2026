"""
Phase 2b — lead-lag scan at cluster level.

Computes lagged cross-correlation tensor for cluster signals:
    rho[s, i, j, k] = corr( signal_i(t),  signal_j(t + k) )

Convention: k>0 means cluster i LEADS cluster j by k ticks.

Signals scanned:
    R   cluster return         (fast horizon, lag in ticks)
    V   realized vol           (slow horizon, can use larger lag)
    |R| absolute return        (vol-shock proxy at fast horizon)

Two horizons:
    fast = 1-tick returns,         K_fast lags
    slow = AGG-tick aggregated,    K_slow lags (returns summed over AGG)

Outputs:
    data/derived/lag_corr_long.parquet
        columns: signal, horizon, c_i, c_j, lag, corr
    data/derived/peak_lag_<signal>_<horizon>.parquet
        10x10 peak-lag corr matrix (signed) and matching argmax-lag matrix
    analysis/figs/peak_lag_<signal>_<horizon>.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(r"C:/Users/Utente/OneDrive/Desktop/IMC_prosperity/Github_IMC/ROUND_5")
DER = ROOT / "data" / "derived"
FIG = ROOT / "analysis" / "figs"
FIG.mkdir(parents=True, exist_ok=True)

K_FAST = 10        # lags at fast horizon (ticks)
K_SLOW = 6         # lags at slow horizon
AGG = 50           # ticks per slow bar


def per_day_shift(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Shift each column by k WITHIN day so day boundary doesn't leak."""
    return df.groupby(level="day").shift(k)


def lag_corr_matrix(X: pd.DataFrame, k: int) -> pd.DataFrame:
    """corr(X_i(t), X_j(t+k)) for all i,j. Rows = i (leader), cols = j (follower)."""
    Xj = per_day_shift(X, -k)             # X_j(t+k) aligned at t
    valid = X.notna() & Xj.notna()
    # vectorized via standardized cov
    Xc = X.where(valid)
    Yc = Xj.where(valid)
    Xz = (Xc - Xc.mean()) / Xc.std(ddof=0)
    Yz = (Yc - Yc.mean()) / Yc.std(ddof=0)
    n = valid.sum()                       # per-column counts (rough; use min for joint)
    # Joint count for pair (i,j): take min over the two; close enough at 30K rows
    M = (Xz.fillna(0).T @ Yz.fillna(0)) / np.minimum.outer(n.values, n.values)
    M.index.name, M.columns.name = "c_i", "c_j"
    return M


def scan_signal(X: pd.DataFrame, lags: list[int],
                signal: str, horizon: str) -> pd.DataFrame:
    rows = []
    for k in lags:
        M = lag_corr_matrix(X, k)
        s = M.stack()
        s.index = s.index.set_names(["c_i", "c_j"])
        s = s.rename("corr").reset_index()
        s["lag"] = k
        s["signal"] = signal
        s["horizon"] = horizon
        rows.append(s)
    return pd.concat(rows, ignore_index=True)


def aggregate_slow(R: pd.DataFrame, agg: int) -> pd.DataFrame:
    """Sum returns into AGG-tick bars per day."""
    R2 = R.copy()
    R2["bar"] = R2.groupby(level="day").cumcount() // agg
    R_slow = (R2.reset_index()
                .groupby(["day", "bar"])
                .sum(numeric_only=True)
                .drop(columns=["timestamp"], errors="ignore"))
    R_slow.index = R_slow.index.set_names(["day", "timestamp"])
    return R_slow


def peak_lag_matrices(long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """For each (c_i, c_j) pick lag k maximizing |corr| over k != 0; return signed corr and lag."""
    nz = long[long["lag"] != 0].copy()
    nz["abscorr"] = nz["corr"].abs()
    idx = nz.groupby(["c_i", "c_j"])["abscorr"].idxmax()
    pick = nz.loc[idx]
    corr_M = pick.pivot(index="c_i", columns="c_j", values="corr")
    lag_M = pick.pivot(index="c_i", columns="c_j", values="lag")
    return corr_M, lag_M


def heatmap_peak(corr_M: pd.DataFrame, lag_M: pd.DataFrame, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(corr_M.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(len(corr_M.columns)), corr_M.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr_M.index)), corr_M.index)
    for i in range(corr_M.shape[0]):
        for j in range(corr_M.shape[1]):
            v = corr_M.values[i, j]
            k = lag_M.values[i, j]
            if pd.isna(v):
                continue
            ax.text(j, i, f"{v:+.2f}\n@{int(k):+d}",
                    ha="center", va="center",
                    color="white" if abs(v) > 0.4 else "black",
                    fontsize=7)
    ax.set_title(title + "  (rows lead cols)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    sig = pd.read_parquet(DER / "cluster_signals.parquet")
    R = sig["R"]
    V = sig["V"]

    # --- fast horizon ---
    lags_fast = list(range(-K_FAST, K_FAST + 1))
    long_R = scan_signal(R, lags_fast, "R", "fast")
    long_aR = scan_signal(R.abs(), lags_fast, "absR", "fast")
    long_V = scan_signal(V, lags_fast, "V", "fast")

    # --- slow horizon (returns aggregated) ---
    R_slow = aggregate_slow(R, AGG)
    lags_slow = list(range(-K_SLOW, K_SLOW + 1))
    long_Rs = scan_signal(R_slow, lags_slow, "R", "slow")

    long = pd.concat([long_R, long_aR, long_V, long_Rs], ignore_index=True)
    long.to_parquet(DER / "lag_corr_long.parquet")

    # peak-lag matrices per (signal, horizon)
    summary_rows = []
    for (s, h), grp in long.groupby(["signal", "horizon"]):
        corr_M, lag_M = peak_lag_matrices(grp)
        corr_M.to_parquet(DER / f"peak_corr_{s}_{h}.parquet")
        lag_M.to_parquet(DER / f"peak_lag_{s}_{h}.parquet")
        heatmap_peak(corr_M, lag_M, f"Peak |corr| lead-lag — {s} ({h})",
                     FIG / f"peak_lag_{s}_{h}.png")
        # top 10 ordered (i->j) pairs by |corr|
        m = corr_M.copy()
        np.fill_diagonal(m.values, np.nan)
        s_top = m.stack().rename("corr")
        s_top = s_top.reindex(s_top.abs().sort_values(ascending=False).index).head(10)
        for (ci, cj), v in s_top.items():
            k = lag_M.loc[ci, cj]
            summary_rows.append({"signal": s, "horizon": h,
                                 "leader": ci, "follower": cj,
                                 "lag": int(k), "corr": float(v)})

    summary = pd.DataFrame(summary_rows)
    summary.to_parquet(DER / "lag_corr_top.parquet")

    print("TOP lead-lag candidates (per signal/horizon, |corr| sort):\n")
    for (s, h), grp in summary.groupby(["signal", "horizon"]):
        print(f"--- {s} / {h} ---")
        print(grp[["leader", "follower", "lag", "corr"]].to_string(index=False))
        print()
    print(f"saved -> {DER}")


if __name__ == "__main__":
    main()
