"""
Phase 4 + Phase 5 — drill-down and final outputs.

For each surviving cluster pair (leader -> follower at lag k*):
  - asset-level 5x5 lag-corr at lag k* (drill-down)
  - row sums / col sums to detect single-driver vs distributed
  - lag-corr curves per pair (cluster level + best asset pair)
  - network graph of survivors
  - final CSV ranked candidates with all metrics

Inputs:
  - data/derived/cluster_signals.parquet
  - data/derived/prices_panel.parquet
  - data/derived/returns_panel.parquet
  - data/derived/cluster_map.json
  - data/derived/candidates_filtered.parquet

Outputs:
  - analysis/figs/lagcurve_<leader>_<follower>.png
  - analysis/figs/drilldown_<leader>_<follower>.png
  - analysis/figs/leadlag_network.png
  - data/derived/survivors_unique.csv
  - data/derived/drilldown_<leader>_<follower>.parquet
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(r"C:/Users/Utente/OneDrive/Desktop/IMC_prosperity/Github_IMC/ROUND_5")
DER = ROOT / "data" / "derived"
FIG = ROOT / "analysis" / "figs"
FIG.mkdir(parents=True, exist_ok=True)

AGG = 50  # must match phase2b
K_RANGE = 8  # lags to plot around k*


# ---------- helpers ----------

def per_day_shift(df: pd.DataFrame, k: int) -> pd.DataFrame:
    return df.groupby(level="day").shift(k)


def lag_corr_pair(x: pd.Series, y: pd.Series, k: int) -> float:
    ys = y.groupby(level="day").shift(-k)
    m = x.notna() & ys.notna()
    if m.sum() < 50:
        return np.nan
    a = x[m].values
    b = ys[m].values
    a = (a - a.mean()) / a.std(ddof=0)
    b = (b - b.mean()) / b.std(ddof=0)
    return float((a * b).mean())


def aggregate_slow(R: pd.DataFrame, agg: int) -> pd.DataFrame:
    R2 = R.copy()
    R2["bar"] = R2.groupby(level="day").cumcount() // agg
    R_slow = (R2.reset_index()
                .groupby(["day", "bar"])
                .sum(numeric_only=True)
                .drop(columns=["timestamp"], errors="ignore"))
    R_slow.index = R_slow.index.set_names(["day", "timestamp"])
    return R_slow


def unique_survivors(cands: pd.DataFrame) -> pd.DataFrame:
    """Drop mirrored rows: keep one direction per pair using positive lag convention."""
    s = cands[cands["pass_all"]].copy()
    seen = set()
    rows = []
    for _, r in s.iterrows():
        key = frozenset((r["leader"], r["follower"]))
        if key in seen:
            continue
        # canonical orientation: row with positive lag
        if r["lag"] > 0:
            rows.append(r)
            seen.add(key)
        else:
            mirror = s[(s["leader"] == r["follower"]) & (s["follower"] == r["leader"])
                       & (s["signal"] == r["signal"]) & (s["horizon"] == r["horizon"])]
            if not mirror.empty:
                rows.append(mirror.iloc[0])
                seen.add(key)
    return pd.DataFrame(rows).reset_index(drop=True)


# ---------- core ----------

def lag_curve(x: pd.Series, y: pd.Series, K: int) -> pd.DataFrame:
    ks = list(range(-K, K + 1))
    rows = [{"lag": k, "corr": lag_corr_pair(x, y, k)} for k in ks]
    return pd.DataFrame(rows)


def drill_down(prices: pd.DataFrame, rets: pd.DataFrame,
               cluster_map: dict, leader: str, follower: str,
               k_star: int, signal: str) -> pd.DataFrame:
    """5x5 asset-asset lag-corr at lag k* between leader cluster and follower cluster."""
    a_leader = cluster_map[leader]
    a_follower = cluster_map[follower]

    if signal == "R":
        # use slow-aggregated returns
        R_slow = aggregate_slow(rets, AGG)
        Xl = R_slow[a_leader]
        Xf = R_slow[a_follower]
    else:  # V (rolling vol of asset return) - use abs returns at fast horizon as proxy
        Xl = rets[a_leader].abs()
        Xf = rets[a_follower].abs()

    M = pd.DataFrame(index=a_leader, columns=a_follower, dtype=float)
    for i in a_leader:
        for j in a_follower:
            M.loc[i, j] = lag_corr_pair(Xl[i], Xf[j], k_star)
    return M


def plot_lag_curves(curves: list[tuple[str, pd.DataFrame, int]], path: Path) -> None:
    n = len(curves)
    fig, axes = plt.subplots(n, 1, figsize=(8, 3 * n), squeeze=False)
    for ax, (title, df, k_star) in zip(axes[:, 0], curves):
        ax.bar(df["lag"], df["corr"], color=["#c44" if k == k_star else "#789" for k in df["lag"]])
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xlabel("lag k  (i lead j  if k>0)")
        ax.set_ylabel("corr(i(t), j(t+k))")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_drilldown(M: pd.DataFrame, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    vmax = max(0.05, np.nanmax(np.abs(M.values)))
    im = ax.imshow(M.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    ax.set_xticks(range(M.shape[1]), M.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(M.shape[0]), M.index, fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M.values[i, j]:+.3f}",
                    ha="center", va="center", fontsize=7,
                    color="white" if abs(M.values[i, j]) > 0.5 * vmax else "black")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_network(survivors: pd.DataFrame, path: Path) -> None:
    """Simple cluster lead-lag DAG, positions on circle."""
    clusters = sorted(set(survivors["leader"]).union(survivors["follower"]))
    n = len(clusters)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = {c: (np.cos(a), np.sin(a)) for c, a in zip(clusters, angles)}

    fig, ax = plt.subplots(figsize=(8, 8))
    for c, (x, y) in pos.items():
        ax.scatter(x, y, s=900, color="#bcd", edgecolor="black", zorder=3)
        ax.text(x, y, c, ha="center", va="center", fontsize=9, zorder=4)

    for _, r in survivors.iterrows():
        x0, y0 = pos[r["leader"]]
        x1, y1 = pos[r["follower"]]
        color = "#c33" if r["corr"] > 0 else "#369"
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", lw=2.0 + 6 * abs(r["corr"]),
                            color=color, shrinkA=22, shrinkB=22),
            zorder=2)
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx, my, f"k={int(r['lag'])}\nρ={r['corr']:+.2f}",
                fontsize=8, ha="center",
                bbox=dict(boxstyle="round", fc="white", ec=color, alpha=0.85),
                zorder=5)

    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Cluster lead-lag survivors  (red=+ corr, blue=− corr)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------- main ----------

def main() -> None:
    cands = pd.read_parquet(DER / "candidates_filtered.parquet")
    sig = pd.read_parquet(DER / "cluster_signals.parquet")
    prices = pd.read_parquet(DER / "prices_panel.parquet")
    rets = pd.read_parquet(DER / "returns_panel.parquet")
    with open(DER / "cluster_map.json") as f:
        cluster_map = json.load(f)

    survivors = unique_survivors(cands)
    survivors.to_csv(DER / "survivors_unique.csv", index=False)
    print(f"Unique survivor pairs: {len(survivors)}")
    print(survivors[["signal", "horizon", "leader", "follower",
                     "lag", "corr", "corr_reverse",
                     "corr_residual", "corr_half1", "corr_half2"]].to_string(index=False))

    # series for lag curves: use CLUSTER-level R (aggregated to slow horizon) and V
    R_cluster = sig["R"]                                 # 10 cluster columns
    R_cluster_slow = aggregate_slow(R_cluster, AGG)      # slow agg
    series_for = {("R", "slow"): R_cluster_slow, ("V", "fast"): sig["V"]}

    curves = []
    for _, r in survivors.iterrows():
        X = series_for[(r["signal"], r["horizon"])]
        df = lag_curve(X[r["leader"]], X[r["follower"]], K_RANGE)
        title = f"{r['leader']} -> {r['follower']}  ({r['signal']}/{r['horizon']})  k*={int(r['lag'])}"
        curves.append((title, df, int(r["lag"])))

    if curves:
        plot_lag_curves(curves, FIG / "lagcurves_survivors.png")

    # drill-down per survivor
    for _, r in survivors.iterrows():
        M = drill_down(prices, rets, cluster_map,
                       r["leader"], r["follower"], int(r["lag"]), r["signal"])
        M.to_parquet(DER / f"drilldown_{r['leader']}_{r['follower']}.parquet")
        title = (f"asset-level lag-corr  {r['leader']} (rows) -> {r['follower']} (cols)  "
                 f"@ k={int(r['lag'])}  signal={r['signal']}/{r['horizon']}")
        plot_drilldown(M, title, FIG / f"drilldown_{r['leader']}_{r['follower']}.png")

        row_max = M.abs().max(axis=1).sort_values(ascending=False)
        col_max = M.abs().max(axis=0).sort_values(ascending=False)
        print(f"\n[{r['leader']}->{r['follower']}] top leader assets (by max |corr| over followers):")
        print(row_max.to_string())
        print(f"[{r['leader']}->{r['follower']}] top follower assets (by max |corr| over leaders):")
        print(col_max.to_string())

    if not survivors.empty:
        plot_network(survivors, FIG / "leadlag_network.png")

    print(f"\nfigs -> {FIG}")
    print(f"survivors csv -> {DER / 'survivors_unique.csv'}")


if __name__ == "__main__":
    main()
