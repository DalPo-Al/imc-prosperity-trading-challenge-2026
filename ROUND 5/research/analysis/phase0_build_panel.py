"""
Phase 0 — build clean panel from prices_{2,3,4}.csv.

Outputs (parquet, in ROUND_5/data/derived/):
  - prices_panel.parquet     : wide mid-prices, MultiIndex (day, timestamp), columns = asset
  - returns_panel.parquet    : log-returns of mid-prices, same shape
  - asset_meta.parquet       : asset -> cluster mapping (one row per asset)
  - cluster_map.json         : {cluster: [assets...]}
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:/Users/Utente/OneDrive/Desktop/IMC_prosperity/Github_IMC/ROUND_5")
PRICES_DIR = ROOT / "data" / "prices"
OUT_DIR = ROOT / "data" / "derived"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_long() -> pd.DataFrame:
    frames = []
    for csv in sorted(PRICES_DIR.glob("prices_*.csv")):
        df = pd.read_csv(csv, sep=";")
        frames.append(df)
    long = pd.concat(frames, ignore_index=True)
    long["cluster"] = long["product"].str.split("_").str[0]
    return long


def build_panel(long: pd.DataFrame) -> pd.DataFrame:
    """Wide mid-price panel: rows = (day, timestamp), columns = asset."""
    panel = (
        long.pivot_table(
            index=["day", "timestamp"],
            columns="product",
            values="mid_price",
            aggfunc="last",
        )
        .sort_index()
    )
    # Forward-fill micro-gaps within a day; drop rows still NA on >50% assets.
    panel = panel.groupby(level="day").ffill()
    drop_mask = panel.isna().mean(axis=1) > 0.5
    panel = panel.loc[~drop_mask]
    return panel


def build_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """Log returns; per-day diff so day boundary doesn't leak."""
    log_p = np.log(panel)
    rets = log_p.groupby(level="day").diff()
    return rets


def main() -> None:
    long = load_long()
    print(f"long rows: {len(long):,}  unique products: {long['product'].nunique()}")

    panel = build_panel(long)
    rets = build_returns(panel)

    # asset -> cluster table
    meta = (
        long[["product", "cluster"]]
        .drop_duplicates()
        .rename(columns={"product": "asset"})
        .sort_values(["cluster", "asset"])
        .reset_index(drop=True)
    )

    cluster_map = (
        meta.groupby("cluster")["asset"].apply(list).to_dict()
    )

    # Persist
    panel.to_parquet(OUT_DIR / "prices_panel.parquet")
    rets.to_parquet(OUT_DIR / "returns_panel.parquet")
    meta.to_parquet(OUT_DIR / "asset_meta.parquet")
    with open(OUT_DIR / "cluster_map.json", "w") as f:
        json.dump(cluster_map, f, indent=2)

    print(f"panel shape: {panel.shape}   returns shape: {rets.shape}")
    print(f"clusters ({len(cluster_map)}): "
          + ", ".join(f"{k}({len(v)})" for k, v in cluster_map.items()))
    print(f"saved -> {OUT_DIR}")


if __name__ == "__main__":
    main()
