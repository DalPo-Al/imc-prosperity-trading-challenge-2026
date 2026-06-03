"""
Phase 1 — cluster-level aggregate signals.

Inputs (from Phase 0):
  - data/derived/prices_panel.parquet     wide mid-prices
  - data/derived/returns_panel.parquet    log returns
  - data/derived/cluster_map.json         {cluster: [assets...]}

Output:
  - data/derived/cluster_signals.parquet
        MultiIndex columns: (signal, cluster)
        signals: P, R, D_ret, D_price_cv, V, breadth, range_rel

Conventions:
  - P_c       = mean mid-price across cluster members at t
  - R_c       = mean log-return across cluster members at t
  - D_ret_c   = cross-sectional std of member returns at t   (scale-free)
  - D_pcv_c   = std(p_i)/mean(p_i) at t                      (price coeff of var)
  - V_c       = rolling std of R_c over VOL_WINDOW            (realized vol)
  - breadth_c = fraction of members with r_i > 0 at t
  - range_rel = (max(p_i) - min(p_i)) / mean(p_i) at t
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:/Users/Utente/OneDrive/Desktop/IMC_prosperity/Github_IMC/ROUND_5")
DER = ROOT / "data" / "derived"

VOL_WINDOW = 100  # ticks; tune later if needed


def cluster_signals(prices: pd.DataFrame,
                    rets: pd.DataFrame,
                    cluster_map: dict[str, list[str]]) -> pd.DataFrame:
    out = {}
    for c, assets in cluster_map.items():
        p = prices[assets]
        r = rets[assets]

        P = p.mean(axis=1)
        R = r.mean(axis=1)
        D_ret = r.std(axis=1, ddof=0)
        D_pcv = p.std(axis=1, ddof=0) / p.mean(axis=1)
        # realized vol of cluster return, per-day rolling so day boundary doesn't leak
        V = (R.groupby(level="day")
              .rolling(VOL_WINDOW, min_periods=VOL_WINDOW // 2)
              .std(ddof=0)
              .reset_index(level=0, drop=True))
        breadth = (r > 0).mean(axis=1)
        rng_rel = (p.max(axis=1) - p.min(axis=1)) / p.mean(axis=1)

        out[("P", c)] = P
        out[("R", c)] = R
        out[("D_ret", c)] = D_ret
        out[("D_pcv", c)] = D_pcv
        out[("V", c)] = V
        out[("breadth", c)] = breadth
        out[("range_rel", c)] = rng_rel

    df = pd.concat(out, axis=1)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["signal", "cluster"])
    return df.sort_index(axis=1)


def main() -> None:
    prices = pd.read_parquet(DER / "prices_panel.parquet")
    rets = pd.read_parquet(DER / "returns_panel.parquet")
    with open(DER / "cluster_map.json") as f:
        cluster_map = json.load(f)

    sig = cluster_signals(prices, rets, cluster_map)
    sig.to_parquet(DER / "cluster_signals.parquet")

    # quick sanity summary
    print(f"signals shape: {sig.shape}")
    print("signal-level NaN ratio:")
    nan_ratio = sig.isna().mean().groupby(level="signal").mean().round(4)
    print(nan_ratio.to_string())
    print("\nsample stats per signal (mean across clusters):")
    desc = sig.groupby(level="signal", axis=1).mean().describe().T[["mean", "std", "min", "max"]]
    print(desc.round(6).to_string())
    print(f"\nsaved -> {DER / 'cluster_signals.parquet'}")


if __name__ == "__main__":
    main()
