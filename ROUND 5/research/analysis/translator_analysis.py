"""
TRANSLATOR cluster — intra-cluster lead-lag + ARMA fit on differenced series.

Steps:
  1. Extract TRANSLATOR assets from prices panel.
  2. Intra-cluster 5x5 lag-corr scan (returns, K=20).
  3. ACF/PACF on each differenced series to guide ARMA order.
  4. Grid-search ARMA(p,q) p,q in {0..4}, select by AIC. Fit per asset.
  5. Outputs: figs + ranked tables.

Outputs:
  analysis/figs/translator_lagcorr_heatmap.png
  analysis/figs/translator_acf_pacf.png
  analysis/figs/translator_lagcurves.png
  analysis/figs/translator_arma_residuals.png
  data/derived/translator_lag_pairs.parquet
  data/derived/translator_arma_results.parquet
"""

from __future__ import annotations

import warnings
from itertools import product as iproduct
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.stattools import durbin_watson

warnings.filterwarnings("ignore")

ROOT = Path(r"C:/Users/Utente/OneDrive/Desktop/IMC_prosperity/Github_IMC/ROUND_5")
DER = ROOT / "data" / "derived"
FIG = ROOT / "analysis" / "figs"
FIG.mkdir(parents=True, exist_ok=True)

K_LAG = 20          # tick lags for intra-cluster scan
ARMA_P_MAX = 4
ARMA_Q_MAX = 4
AGG = 1             # keep tick-level; try slow too if needed

ASSETS = [
    "TRANSLATOR_ASTRO_BLACK",
    "TRANSLATOR_ECLIPSE_CHARCOAL",
    "TRANSLATOR_GRAPHITE_MIST",
    "TRANSLATOR_SPACE_GRAY",
    "TRANSLATOR_VOID_BLUE",
]


# ── helpers ──────────────────────────────────────────────────────────────────

def per_day_shift(s: pd.Series, k: int) -> pd.Series:
    return s.groupby(level="day").shift(k)


def lag_corr(x: pd.Series, y: pd.Series, k: int) -> float:
    """corr(x(t), y(t+k))  i.e. x leads y if k>0."""
    ys = per_day_shift(y, -k)
    m = x.notna() & ys.notna()
    if m.sum() < 100:
        return np.nan
    a, b = x[m].values.copy(), ys[m].values.copy()
    a -= a.mean(); a /= a.std(ddof=0) + 1e-12
    b -= b.mean(); b /= b.std(ddof=0) + 1e-12
    return float((a * b).mean())


def lag_matrix(rets: pd.DataFrame, k: int) -> pd.DataFrame:
    """5x5 matrix: row=leader, col=follower at lag k."""
    M = pd.DataFrame(index=ASSETS, columns=ASSETS, dtype=float)
    for i in ASSETS:
        for j in ASSETS:
            M.loc[i, j] = lag_corr(rets[i], rets[j], k)
    return M


def short_name(a: str) -> str:
    return a.replace("TRANSLATOR_", "").replace("_", " ").title()


# ── 1. load ───────────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = pd.read_parquet(DER / "prices_panel.parquet")[ASSETS]
    rets = pd.read_parquet(DER / "returns_panel.parquet")[ASSETS]
    return prices, rets


# ── 2. intra-cluster lead-lag ─────────────────────────────────────────────────

def run_lag_scan(rets: pd.DataFrame) -> pd.DataFrame:
    lags = list(range(-K_LAG, K_LAG + 1))
    rows = []
    pairs = [(i, j) for i in ASSETS for j in ASSETS if i != j]
    for (ci, cj) in pairs:
        for k in lags:
            c = lag_corr(rets[ci], rets[cj], k)
            rows.append({"leader": ci, "follower": cj, "lag": k, "corr": c})
    long = pd.DataFrame(rows)
    long.to_parquet(DER / "translator_lag_pairs.parquet")
    return long


def plot_lag_heatmap(long: pd.DataFrame) -> None:
    """Heatmap of peak |corr| at non-zero lag (5x5)."""
    nz = long[long["lag"] != 0].copy()
    nz["abscorr"] = nz["corr"].abs()
    peak = nz.groupby(["leader", "follower"]).apply(
        lambda g: g.loc[g["abscorr"].idxmax(), ["lag", "corr"]]
    ).reset_index()

    corr_M = peak.pivot(index="leader", columns="follower", values="corr")
    lag_M = peak.pivot(index="leader", columns="follower", values="lag")

    # reorder
    corr_M = corr_M.loc[ASSETS, ASSETS]
    lag_M = lag_M.loc[ASSETS, ASSETS]
    np.fill_diagonal(corr_M.values, np.nan)
    np.fill_diagonal(lag_M.values, 0)

    labs = [short_name(a) for a in ASSETS]
    vmax = np.nanmax(np.abs(corr_M.values))

    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(corr_M.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    ax.set_xticks(range(5), labs, rotation=40, ha="right")
    ax.set_yticks(range(5), labs)
    for i in range(5):
        for j in range(5):
            if i == j:
                continue
            v, k = corr_M.values[i, j], int(lag_M.values[i, j])
            ax.text(j, i, f"{v:+.3f}\nk={k:+d}",
                    ha="center", va="center", fontsize=7.5,
                    color="white" if abs(v) > 0.5 * vmax else "black")
    ax.set_title("TRANSLATOR intra-cluster  peak |lag-corr|\n(row leads col; k = ticks, + means row leads)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(FIG / "translator_lagcorr_heatmap.png", dpi=130)
    plt.close(fig)

    # also plot lag-corr curves for strongest off-diagonal pairs
    peak_nz = peak[peak["leader"] != peak["follower"]].copy()
    peak_nz["abscorr"] = peak_nz["corr"].abs()
    top4 = peak_nz.nlargest(4, "abscorr")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    for ax, (_, row) in zip(axes, top4.iterrows()):
        sub = long[(long["leader"] == row["leader"]) &
                   (long["follower"] == row["follower"])].sort_values("lag")
        ax.bar(sub["lag"], sub["corr"],
               color=["#c44" if abs(k) == abs(int(row["lag"])) else "#789" for k in sub["lag"]])
        ax.axhline(0, color="black", lw=0.6)
        ax.set_title(f"{short_name(row['leader'])} → {short_name(row['follower'])}", fontsize=9)
        ax.set_xlabel("lag k")
        ax.set_ylabel("corr")
    fig.suptitle("TRANSLATOR  top-4 intra-cluster lag-corr curves", y=1.01)
    fig.tight_layout()
    fig.savefig(FIG / "translator_lagcurves.png", dpi=130)
    plt.close(fig)

    return corr_M, lag_M


# ── 3. ACF / PACF ─────────────────────────────────────────────────────────────

def plot_acf_pacf(rets: pd.DataFrame) -> None:
    """ACF + PACF for each TRANSLATOR asset return (concat across days, no day-gap)."""
    fig, axes = plt.subplots(5, 2, figsize=(14, 16))
    for row_ax, asset in zip(axes, ASSETS):
        # concatenate days; drop first-tick NaN per day
        series = rets[asset].dropna().values
        plot_acf(series, ax=row_ax[0], lags=30, title=f"ACF  {short_name(asset)}")
        plot_pacf(series, ax=row_ax[1], lags=30, method="ywm",
                  title=f"PACF  {short_name(asset)}")
    fig.tight_layout()
    fig.savefig(FIG / "translator_acf_pacf.png", dpi=100)
    plt.close(fig)


# ── 4. ARMA grid search ───────────────────────────────────────────────────────

def arma_grid(series: np.ndarray, p_max: int, q_max: int) -> pd.DataFrame:
    rows = []
    for p, q in iproduct(range(p_max + 1), range(q_max + 1)):
        if p == 0 and q == 0:
            continue
        try:
            res = ARIMA(series, order=(p, 0, q)).fit(method_kwargs={"warn_convergence": False})
            dw = durbin_watson(res.resid)
            rows.append({
                "p": p, "q": q,
                "aic": res.aic, "bic": res.bic,
                "llf": res.llf,
                "dw": dw,
                "n_params": p + q,
            })
        except Exception:
            pass
    return pd.DataFrame(rows).sort_values("aic").reset_index(drop=True)


def run_arma(rets: pd.DataFrame) -> pd.DataFrame:
    all_results = []
    best_models = {}

    for asset in ASSETS:
        series = rets[asset].dropna().values
        grid = arma_grid(series, ARMA_P_MAX, ARMA_Q_MAX)
        best = grid.iloc[0]
        best_models[asset] = best
        grid["asset"] = asset
        all_results.append(grid)

        print(f"{short_name(asset):25s}  best ARMA({int(best.p)},{int(best.q)})"
              f"  AIC={best.aic:.1f}  BIC={best.bic:.1f}  DW={best.dw:.3f}")

    full = pd.concat(all_results, ignore_index=True)
    full.to_parquet(DER / "translator_arma_results.parquet")
    return full, best_models


def plot_residuals(rets: pd.DataFrame, best_models: dict) -> None:
    fig, axes = plt.subplots(5, 1, figsize=(12, 16), sharex=False)
    for ax, asset in zip(axes, ASSETS):
        p, q = int(best_models[asset]["p"]), int(best_models[asset]["q"])
        series = rets[asset].dropna().values
        res = ARIMA(series, order=(p, 0, q)).fit(method_kwargs={"warn_convergence": False})
        ax.plot(res.resid[:3000], lw=0.5, alpha=0.8)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_title(f"{short_name(asset)}  ARMA({p},{q})  AIC={res.aic:.1f}", fontsize=9)
    fig.suptitle("TRANSLATOR  ARMA residuals (first 3000 ticks)", y=1.01)
    fig.tight_layout()
    fig.savefig(FIG / "translator_arma_residuals.png", dpi=100)
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading...")
    prices, rets = load_data()

    print("\n── Lag scan (K=20, tick level) ──")
    long = run_lag_scan(rets)
    corr_M, lag_M = plot_lag_heatmap(long)

    print("\nPeak lag-corr matrix (rows lead cols):")
    np.fill_diagonal(corr_M.values, np.nan)
    corr_disp = corr_M.copy()
    corr_disp.index = [short_name(a) for a in ASSETS]
    corr_disp.columns = [short_name(a) for a in ASSETS]
    print(corr_disp.round(4).to_string())

    print("\n── ACF / PACF ──")
    plot_acf_pacf(rets)

    print("\n── ARMA grid search ──")
    full, best_models = run_arma(rets)

    print("\nTop 3 per asset by AIC:")
    for asset in ASSETS:
        top3 = full[full["asset"] == asset].head(3)[["p", "q", "aic", "bic", "dw"]]
        print(f"  {short_name(asset)}:")
        print(top3.to_string(index=False))

    plot_residuals(rets, best_models)
    print(f"\nFigs → {FIG}")


if __name__ == "__main__":
    main()
