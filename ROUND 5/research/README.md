# Research

## Purpose

Round 5 cross-asset structure analysis on 50 instruments (10 categories × 5 assets: GALAXY, MICROCHIP, OXYGEN, PANEL, PEBBLES, ROBOT, SLEEP, SNACKPACK, TRANSLATOR, UV). Questions: do any categories lead others in returns? Is there exploitable intra-category structure (ARMA, mean-reversion)?

## Methodology

Two parallel angles, both implemented as standalone pipelines:

**Lead-lag pipeline** (`analysis/`): five sequential phases.
- Phase 0 — build clean price/returns panel from raw CSVs; write `data/derived/`.
- Phase 1 — compute cluster-level aggregate signals (mean return, dispersion, vol, breadth).
- Phase 2a/2b — contemporaneous + lagged cross-correlation scan across all 10×10 cluster pairs and signal types (R, V, |R|), fast and slow horizons.
- Phase 3 — four-gate filter: circular-shuffle null (95th pct), direction asymmetry (1.3× ratio), common-driver residualisation (50% retention threshold), half-sample stability (same sign, ≥40% of full corr).
- Phase 4/5 — drill-down to asset-level 5×5 lag-corr for surviving pairs; lag-curve plots; network graph; `survivors_unique.csv`.
- Supplementary: `translator_analysis.py` — intra-cluster lag-corr + ARMA(p,q) grid search on TRANSLATOR assets specifically.

**Category fingerprint pipeline** (`cluster_analysis.py`, `cluster_analysis.ipynb`): structural characterisation of each category via 11 features (trend slope, vol, vol-of-vol, ACF at 1/5/20, Hurst, max drawdown, skew, excess kurtosis, intra-correlation). Dominance ranking, regime-conditional fingerprints, temporal drift over 4 windows, Ward hierarchical clustering into 3 archetypes.

## Key Findings

Three cluster pairs survived all four gates (`survivors_unique.csv`):

| Leader | Follower | Lag | Signal | Corr | Notes |
|--------|----------|-----|--------|------|-------|
| SLEEP | ROBOT | 1 (slow) | R | −0.126 | Anti-correlated, 1 aggregated tick |
| PANEL | SNACKPACK | 1 (slow) | R | +0.120 | Positive, 1 aggregated tick |
| UV | GALAXY | 3 (slow) | R | +0.115 | Positive, 3 aggregated ticks |

All three correlations are ~0.11–0.13 — statistically significant under the null but economically marginal. No result was strong enough to implement satisfyingly. No finding from either pipeline was used in the final Round 5 strategy.

## File Reference

| File | Role | Notes |
|------|------|-------|
| `analysis/phase0_build_panel.py` | Pipeline step | Loads `data/prices/*.csv` → prices/returns panel + `cluster_map.json`. **Note:** hardcodes `ROOT` path — update before re-running. |
| `analysis/phase1_cluster_signals.py` | Pipeline step | Cluster aggregate signals from panel. |
| `analysis/phase2a_contemp_corr.py` | Pipeline step | Contemporaneous 10×10 corr matrices. |
| `analysis/phase2b_lag_corr.py` | Pipeline step | Full lagged corr tensor, fast + slow horizons. |
| `analysis/phase3_filter.py` | Pipeline step | Four-gate significance filter → `candidates_filtered.parquet`. |
| `analysis/phase4_5_outputs.py` | Pipeline step | Drill-down + figures + `survivors_unique.csv`. |
| `analysis/translator_analysis.py` | Supplementary | TRANSLATOR-specific intra-cluster lag + ARMA grid search. |
| `analysis/figs/` | Artefacts | All output plots from the lead-lag pipeline (see below). |
| `cluster_analysis.py` | Analysis script | Category fingerprint + archetype pipeline; all phases implemented as importable functions. |
| `cluster_analysis.ipynb` | Analysis notebook | Calls `cluster_analysis.py` phase by phase; contains methodology narrative. |
| `io_results.py` | Support module | Shared save/load for timestamped run subdirectories (parquet + npz + meta.json). Used by `cluster_analysis.py`. |
| `vis_tools.py` | Support script | Price-panel loader and log-return visualiser. |
| `visualizer.ipynb` | EDA notebook | Thin wrapper over `vis_tools.py` for exploratory plots. |
| `data/derived/cluster_map.json` | Config | `{category: [asset, ...]}` mapping, 10 categories × 5 assets. Required by both pipelines. |
| `data/derived/survivors_unique.csv` | Output | Three surviving lead-lag pairs with all filter metrics. |
| `analysis/figs/others/` | Artefacts | Exploratory PCA/correlation plots from early EDA. |

### Figures (`analysis/figs/`)

| File | Produced by | Content |
|------|-------------|---------|
| `contemp_return_pearson.png`, `contemp_return_spearman.png`, `contemp_vol.png`, `contemp_disp.png` | `phase2a` | 10×10 contemporaneous correlation heatmaps |
| `peak_lag_R_fast.png`, `peak_lag_R_slow.png`, `peak_lag_V_fast.png`, `peak_lag_absR_fast.png` | `phase2b` | Peak lag-corr matrices across horizons |
| `drilldown_PANEL_SNACKPACK.png`, `drilldown_SLEEP_ROBOT.png`, `drilldown_UV_GALAXY.png` | `phase4_5` | 5×5 asset-level lag-corr for each surviving pair |
| `lagcurves_survivors.png` | `phase4_5` | Lag-corr curves for surviving pairs |
| `leadlag_network.png` | `phase4_5` | Directed network of surviving edges |
| `translator_lagcorr_heatmap.png`, `translator_lagcurves.png`, `translator_acf_pacf.png`, `translator_arma_residuals.png` | `translator_analysis` | TRANSLATOR intra-cluster results |

## Dependencies

- `numpy`, `pandas` — numerics and data manipulation throughout.
- `scipy` — statistics, hierarchical clustering (`cluster_analysis.py`).
- `matplotlib` — all plots.
- `statsmodels` — VAR, ARMA, ACF/PACF, ADF (`translator_analysis.py`, `cluster_analysis.py`).
- `sklearn` — `StandardScaler`, `LassoCV` (used in removed pipelines; `cluster_analysis.py` uses only `StandardScaler`).

## Replication

1. Source raw prices into `data/prices/` as `prices_{2,3,4}.csv` (semicolon-delimited; columns: `day`, `timestamp`, `product`, `mid_price`). Not in repo.
2. Fix the hardcoded `ROOT` path in `analysis/phase0_build_panel.py` to point to this directory.
3. Run phases in order: `phase0` → `phase1` → `phase2a` → `phase2b` → `phase3` → `phase4_5`. Each reads from and writes to `data/derived/` and `analysis/figs/`.
4. `cluster_analysis.ipynb` — run cells top to bottom; requires `data/prices/` and `data/derived/cluster_map.json`.

## Usage of findings

None. All survivor correlations (~0.11–0.13) were too weak to implement satisfyingly, and no result from either the lead-lag or fingerprint pipeline was incorporated into the final Round 5 strategy. This folder documents what was investigated and ruled out.

## Authors

Giorgio Cottini.
