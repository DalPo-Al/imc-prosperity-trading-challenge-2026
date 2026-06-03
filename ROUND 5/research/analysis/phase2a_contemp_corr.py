"""
Phase 2a — contemporaneous cluster-level dependencies.

Produces three 10x10 correlation matrices on cluster signals:
  - return        Pearson + Spearman on R_c
  - volatility    Pearson on V_c
  - dispersion    Pearson on D_ret_c

Outputs:
  data/derived/corr_return_pearson.parquet
  data/derived/corr_return_spearman.parquet
  data/derived/corr_vol.parquet
  data/derived/corr_disp.parquet
  analysis/figs/contemp_*.png
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


def heatmap(M: pd.DataFrame, title: str, path: Path, cmap: str = "RdBu_r",
            vmin: float = -1.0, vmax: float = 1.0) -> None:
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(M.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xticks(range(len(M.columns)), M.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(M.index)), M.index)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M.values[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if abs(M.values[i, j]) > 0.5 else "black",
                    fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    sig = pd.read_parquet(DER / "cluster_signals.parquet")

    R = sig["R"]            # cluster returns
    V = sig["V"].dropna()   # rolling vol
    D = sig["D_ret"]        # cross-sectional return dispersion

    corr_R_p = R.corr(method="pearson")
    corr_R_s = R.corr(method="spearman")
    corr_V = V.corr(method="pearson")
    corr_D = D.corr(method="pearson")

    corr_R_p.to_parquet(DER / "corr_return_pearson.parquet")
    corr_R_s.to_parquet(DER / "corr_return_spearman.parquet")
    corr_V.to_parquet(DER / "corr_vol.parquet")
    corr_D.to_parquet(DER / "corr_disp.parquet")

    heatmap(corr_R_p, "Cluster return corr (Pearson)", FIG / "contemp_return_pearson.png")
    heatmap(corr_R_s, "Cluster return corr (Spearman)", FIG / "contemp_return_spearman.png")
    heatmap(corr_V, "Cluster realized-vol corr", FIG / "contemp_vol.png")
    heatmap(corr_D, "Cluster return-dispersion corr", FIG / "contemp_disp.png")

    def top_pairs(M: pd.DataFrame, k: int = 8) -> pd.DataFrame:
        m = M.copy()
        m.index = m.index.set_names("c_i")
        m.columns = m.columns.set_names("c_j")
        np.fill_diagonal(m.values, np.nan)
        s = m.where(np.triu(np.ones_like(m, dtype=bool), k=1)).stack()
        s = s.reindex(s.abs().sort_values(ascending=False).index)
        return s.head(k).rename("corr").reset_index()

    print("\nTOP cluster-return pairs (Pearson):")
    print(top_pairs(corr_R_p).to_string(index=False))
    print("\nTOP cluster-vol pairs:")
    print(top_pairs(corr_V).to_string(index=False))
    print("\nTOP cluster-dispersion pairs:")
    print(top_pairs(corr_D).to_string(index=False))
    print(f"\nfigs -> {FIG}")


if __name__ == "__main__":
    main()
