"""
Category-level structural fingerprinting and inter-cluster pattern mining.

Pipeline (see cluster_analysis.ipynb for the methodology + reasoning):

    Phase 1 — fingerprint each category as the median across its 5 assets of:
        trend_slope, vol_mean, vol_of_vol, acf_1, acf_5, acf_20,
        hurst, max_dd, skew, xs_kurt
        + intra_corr (per-category, not aggregated from per-asset)

    Phase 2a — rank categories on each fingerprint dimension; surface
               consistent dominance / underperformance.

    Phase 2b — category price indices, full-sample correlation, and
               regime-conditional correlation (first half vs second half).

    Phase 2c — fingerprint recomputed in low/high rolling-vol regimes;
               flag features that flip vs stay stable.

    Phase 2d — fingerprint per 4 equal time windows; report drift L2 per
               category and which pairs converge / diverge.

    Phase 3  — hierarchical (Ward) clustering of the z-scored fingerprint
               into 3 archetypes; auto-emit falsifiable hypotheses as JSON.

All artifacts saved via `io_results.save_run` to `results_clusters/<TS>/`.
Designed to be imported (every phase is a function) AND run standalone.
"""

from __future__ import annotations

import json
import logging
import time
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("clusters")

# Per-asset features (computed for each asset, then median-aggregated to category).
PER_ASSET_FEATURES = (
    "trend_slope",
    "vol_mean",
    "vol_of_vol",
    "acf_1",
    "acf_5",
    "acf_20",
    "hurst",
    "max_dd",
    "skew",
    "xs_kurt",
)
# Intrinsically per-category features (not aggregated from per-asset).
PER_CATEGORY_FEATURES = ("intra_corr",)

ALL_FEATURES = PER_ASSET_FEATURES + PER_CATEGORY_FEATURES


# ============================================================================ #
# Data loading
# ============================================================================ #

def load_panel(
    prices_dir: str | Path = "data/prices",
    cluster_map_path: str | Path = "data/derived/cluster_map.json",
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Returns:
        prices : (T, 50) wide DataFrame, index = global integer time `t`,
                 columns = product names
        clusters : {category: [asset, ...]}  (5 assets per category)
    """
    files = sorted(Path(prices_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSVs under {prices_dir}")
    raw = pd.concat(
        [pd.read_csv(f, sep=";") for f in files], ignore_index=True
    )
    raw = raw.sort_values(["day", "timestamp"]).reset_index(drop=True)
    # global integer time across days
    keys = list(raw.groupby(["day", "timestamp"]).groups.keys())
    time_map = {k: i for i, k in enumerate(keys)}
    raw["t"] = list(map(time_map.__getitem__, zip(raw["day"], raw["timestamp"])))
    wide = (
        raw.pivot_table(index="t", columns="product", values="mid_price",
                        aggfunc="first")
           .sort_index()
    )
    with open(cluster_map_path) as f:
        clusters = json.load(f)
    # sanity: every listed asset must be present in the panel
    missing = {
        cat: [a for a in assets if a not in wide.columns]
        for cat, assets in clusters.items()
    }
    missing = {k: v for k, v in missing.items() if v}
    if missing:
        log.warning("missing assets in panel: %s", missing)
    return wide, clusters


# ============================================================================ #
# Per-asset structural features
# ============================================================================ #

def _log_returns(p: pd.Series) -> pd.Series:
    return np.log(p.replace(0, np.nan)).diff().dropna()


def trend_slope(prices: pd.Series) -> float:
    """OLS slope of log(price) on time, scaled by 1/mean(log price). Unitless."""
    p = np.log(prices.replace(0, np.nan)).dropna()
    if len(p) < 10:
        return np.nan
    x = np.arange(len(p), dtype=float)
    slope, *_ = np.polyfit(x, p.values, 1)
    denom = max(abs(p.mean()), 1e-9)
    return float(slope / denom)


def vol_metrics(returns: pd.Series, window: int = 200) -> Tuple[float, float]:
    """Mean and std of rolling std (vol regime + vol-of-vol)."""
    rv = returns.rolling(window, min_periods=max(20, window // 4)).std()
    rv = rv.dropna()
    if rv.empty:
        return np.nan, np.nan
    return float(rv.mean()), float(rv.std(ddof=1))


def acf_at(returns: pd.Series, lags=(1, 5, 20)) -> Dict[int, float]:
    """Sample ACF of returns at given lags."""
    x = returns.values - returns.values.mean()
    var = float((x * x).mean())
    out = {}
    if var <= 0:
        return {L: 0.0 for L in lags}
    for L in lags:
        if L >= len(x):
            out[L] = np.nan
        else:
            out[L] = float((x[:-L] * x[L:]).mean() / var)
    return out


def hurst_rs(prices: pd.Series, n_chunks: int = 10) -> float:
    """
    Rescaled-range (R/S) Hurst estimator on log-prices.
    H ~= slope of log(R/S) vs log(window). H<0.5 mean-reverting, >0.5 trending.
    Returns NaN if the series is too short.
    """
    p = np.log(prices.replace(0, np.nan)).dropna().values
    n = len(p)
    if n < 200:
        return np.nan
    # geometric grid of window sizes
    sizes = np.unique(np.geomspace(20, n // 2, num=n_chunks).astype(int))
    if len(sizes) < 3:
        return np.nan
    rs = []
    for w in sizes:
        n_seg = n // w
        if n_seg < 1:
            continue
        seg = p[: n_seg * w].reshape(n_seg, w)
        diffs = np.diff(seg, axis=1)
        if diffs.shape[1] == 0:
            continue
        mean = diffs.mean(axis=1, keepdims=True)
        dev = (diffs - mean).cumsum(axis=1)
        R = dev.max(axis=1) - dev.min(axis=1)
        S = diffs.std(axis=1, ddof=1)
        ok = (S > 0) & np.isfinite(S)
        if ok.sum() < 1:
            continue
        rs.append((w, float(np.mean(R[ok] / S[ok]))))
    if len(rs) < 3:
        return np.nan
    ws, vals = zip(*rs)
    h, *_ = np.polyfit(np.log(ws), np.log(vals), 1)
    return float(h)


def max_drawdown(prices: pd.Series) -> float:
    """Max drawdown as a positive fraction; 0 if series only goes up."""
    p = prices.dropna().values
    if len(p) < 2:
        return np.nan
    peak = np.maximum.accumulate(p)
    dd = (p - peak) / np.where(peak == 0, 1, peak)
    return float(-dd.min())


def per_asset_fingerprint(prices: pd.Series, vol_window: int = 200) -> dict:
    """All scalar features for one asset's price series."""
    rets = _log_returns(prices)
    vm, vov = vol_metrics(rets, window=vol_window)
    acf = acf_at(rets, lags=(1, 5, 20))
    return {
        "trend_slope": trend_slope(prices),
        "vol_mean": vm,
        "vol_of_vol": vov,
        "acf_1": acf[1],
        "acf_5": acf[5],
        "acf_20": acf[20],
        "hurst": hurst_rs(prices),
        "max_dd": max_drawdown(prices),
        "skew": float(stats.skew(rets, nan_policy="omit")),
        "xs_kurt": float(stats.kurtosis(rets, nan_policy="omit")),
    }


# ============================================================================ #
# Phase 1 — category fingerprint
# ============================================================================ #

def category_fingerprint(
    prices: pd.DataFrame,
    clusters: Dict[str, List[str]],
    vol_window: int = 200,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        fp        : (10, 11) category fingerprint (median of per-asset features
                    + per-category intra_corr)
        per_asset : (50, 10) raw per-asset feature matrix (long index: asset,
                    column 'category' included)
    """
    rows = []
    for cat, assets in clusters.items():
        for a in assets:
            if a not in prices.columns:
                continue
            feats = per_asset_fingerprint(prices[a], vol_window=vol_window)
            feats["asset"] = a
            feats["category"] = cat
            rows.append(feats)
    per_asset = pd.DataFrame(rows).set_index("asset")

    # median aggregate per category
    fp_assets = per_asset.groupby("category")[list(PER_ASSET_FEATURES)].median()

    # per-category intra-cluster mean pairwise return correlation
    intra = {}
    for cat, assets in clusters.items():
        cols = [a for a in assets if a in prices.columns]
        if len(cols) < 2:
            intra[cat] = np.nan
            continue
        rets = np.log(prices[cols].replace(0, np.nan)).diff().dropna(how="all")
        c = rets.corr().values
        m = ~np.eye(len(cols), dtype=bool)
        intra[cat] = float(np.nanmean(c[m]))
    fp_assets["intra_corr"] = pd.Series(intra)
    fp_assets = fp_assets.reindex(list(clusters.keys()))
    return fp_assets, per_asset


def zscore_fingerprint(fp: pd.DataFrame) -> pd.DataFrame:
    """Column z-score the fingerprint matrix; constant cols -> 0."""
    Z = fp.copy()
    for c in Z.columns:
        col = Z[c].astype(float)
        sd = col.std(ddof=1)
        if not np.isfinite(sd) or sd == 0:
            Z[c] = 0.0
        else:
            Z[c] = (col - col.mean()) / sd
    return Z


# ============================================================================ #
# Phase 2a — dominance / underperformance ranking
# ============================================================================ #

def dominance_ranks(fp: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        ranks   : (10, 11) rank of each category on each feature (1 = lowest)
        summary : per-category mean_rank, n_top2, n_bot2, label
    """
    ranks = fp.rank(method="average")  # higher value -> higher rank
    n = len(ranks)
    n_top = (ranks >= n - 1).sum(axis=1)      # times ranked top-2
    n_bot = (ranks <= 2).sum(axis=1)          # times ranked bottom-2
    summary = pd.DataFrame(
        {
            "mean_rank": ranks.mean(axis=1),
            "rank_std": ranks.std(axis=1, ddof=1),
            "n_top2": n_top,
            "n_bot2": n_bot,
        }
    ).sort_values("mean_rank", ascending=False)
    return ranks, summary


# ============================================================================ #
# Phase 2b — category indices, full + half-sample correlations
# ============================================================================ #

def category_indices(
    prices: pd.DataFrame,
    clusters: Dict[str, List[str]],
    normalize: bool = True,
) -> pd.DataFrame:
    """
    Equal-weight category index = mean of per-asset price (optionally divided
    by its first valid observation so all indices start at 1).
    """
    out = {}
    for cat, assets in clusters.items():
        cols = [a for a in assets if a in prices.columns]
        sub = prices[cols].copy()
        if normalize:
            sub = sub.divide(sub.bfill().iloc[0])
        out[cat] = sub.mean(axis=1, skipna=True)
    df = pd.DataFrame(out).sort_index()
    return df


def comovement(
    indices: pd.DataFrame,
    n_splits: int = 2,
    use_returns: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Returns dict containing:
        corr_full        : (k, k) Pearson correlation
        corr_split_<i>   : per-time-window Pearson, i in 0..n_splits-1
        delta_corr       : full-sample - 1st window  (for each pair)
        sign_flip_pairs  : long-form table of pairs whose sign flipped between
                           any two windows
    """
    X = np.log(indices.replace(0, np.nan)).diff().dropna() if use_returns else indices
    full = X.corr()

    splits = np.array_split(X, n_splits)
    sub_corrs = {f"corr_split_{i}": s.corr() for i, s in enumerate(splits)}

    # detect sign flips
    flips = []
    for a, b in combinations(full.columns, 2):
        cs = [s.loc[a, b] for s in [v for v in sub_corrs.values()]]
        signs = np.sign(cs)
        if len(set(signs)) > 1 and all(np.isfinite(cs)):
            flips.append({
                "a": a, "b": b, "full": float(full.loc[a, b]),
                **{k: float(v.loc[a, b]) for k, v in sub_corrs.items()},
            })
    flips_df = pd.DataFrame(flips).sort_values(
        "full", key=lambda s: s.abs(), ascending=False
    )

    return {"corr_full": full, "delta_corr": None, "sign_flip_pairs": flips_df, **sub_corrs}


# ============================================================================ #
# Phase 2c — threshold-conditional fingerprints
# ============================================================================ #

def regime_fingerprints(
    prices: pd.DataFrame,
    clusters: Dict[str, List[str]],
    indices: pd.DataFrame,
    vol_window: int = 200,
) -> Dict[str, pd.DataFrame]:
    """
    Recompute the fingerprint inside two regimes per category-index:
        - low_vol  : t where rolling vol of the category index is below median
        - high_vol : t where it's above median

    Returns {regime_name: (10, 11) fingerprint}. Stability is reported in
    `regime_stability_summary` via the L1 difference between regimes.
    """
    rets = np.log(indices.replace(0, np.nan)).diff()
    vol = rets.rolling(vol_window, min_periods=vol_window // 4).std()

    fp_low_rows = {}
    fp_high_rows = {}
    for cat, assets in clusters.items():
        v = vol[cat].dropna()
        if v.empty:
            continue
        med = v.median()
        idx_low = v.index[v <= med]
        idx_high = v.index[v > med]

        sub_low = prices.loc[idx_low.intersection(prices.index)]
        sub_high = prices.loc[idx_high.intersection(prices.index)]
        if len(sub_low) < 50 or len(sub_high) < 50:
            continue
        fp_l, _ = category_fingerprint(sub_low, {cat: assets},
                                       vol_window=max(20, vol_window // 4))
        fp_h, _ = category_fingerprint(sub_high, {cat: assets},
                                       vol_window=max(20, vol_window // 4))
        fp_low_rows[cat] = fp_l.iloc[0]
        fp_high_rows[cat] = fp_h.iloc[0]
    fp_low = pd.DataFrame(fp_low_rows).T.reindex(indices.columns)
    fp_high = pd.DataFrame(fp_high_rows).T.reindex(indices.columns)

    # stability = |Δ| per (category, feature); summary = mean over features
    delta = (fp_high - fp_low).abs()
    summary = pd.DataFrame({
        "mean_abs_delta": delta.mean(axis=1, skipna=True),
        "max_abs_delta_feature": delta.idxmax(axis=1),
        "max_abs_delta_value": delta.max(axis=1, skipna=True),
    }).sort_values("mean_abs_delta", ascending=False)
    return {
        "fp_low_vol": fp_low,
        "fp_high_vol": fp_high,
        "regime_delta": delta,
        "regime_stability_summary": summary,
    }


# ============================================================================ #
# Phase 2d — temporal drift across equal windows
# ============================================================================ #

def temporal_drift(
    prices: pd.DataFrame,
    clusters: Dict[str, List[str]],
    n_windows: int = 4,
    vol_window: int = 100,
) -> Dict[str, pd.DataFrame]:
    """
    Cut the panel into `n_windows` equal time slices, recompute the fingerprint
    inside each, and report:
        fp_window_<i>      : per-window fingerprint
        drift_distance     : L2 distance (in z-score space) between consecutive
                             windows per category
        pair_convergence   : per-pair Δdistance between window 0 and window n-1
    """
    T = len(prices)
    edges = np.linspace(0, T, n_windows + 1, dtype=int)
    fp_windows = []
    for w in range(n_windows):
        chunk = prices.iloc[edges[w]: edges[w + 1]]
        if len(chunk) < 100:
            continue
        fp_w, _ = category_fingerprint(chunk, clusters,
                                       vol_window=max(20, vol_window))
        fp_windows.append(fp_w)

    # z-score columns within each window so cross-feature scales are comparable
    fp_z = [zscore_fingerprint(fp) for fp in fp_windows]

    # consecutive-window per-category L2 drift
    drift = {}
    for w in range(1, len(fp_z)):
        d = ((fp_z[w] - fp_z[w - 1]) ** 2).sum(axis=1).pow(0.5)
        drift[f"drift_w{w-1}_w{w}"] = d
    drift_df = pd.DataFrame(drift)
    drift_df["mean_drift"] = drift_df.mean(axis=1)
    drift_df = drift_df.sort_values("mean_drift", ascending=False)

    # pair convergence: distance(cat_i, cat_j) at window 0 vs window n-1
    if len(fp_z) >= 2:
        D0 = pd.DataFrame(
            squareform(pdist(fp_z[0].fillna(0).values)),
            index=fp_z[0].index, columns=fp_z[0].index,
        )
        Dn = pd.DataFrame(
            squareform(pdist(fp_z[-1].fillna(0).values)),
            index=fp_z[-1].index, columns=fp_z[-1].index,
        )
        delta_pair = (Dn - D0)
        # melt to long form, sorted: most-converging first (most negative)
        pairs = []
        for a, b in combinations(delta_pair.index, 2):
            pairs.append({
                "a": a, "b": b,
                "dist_w0": float(D0.loc[a, b]),
                "dist_wN": float(Dn.loc[a, b]),
                "delta": float(delta_pair.loc[a, b]),
            })
        pair_df = pd.DataFrame(pairs).sort_values("delta")
    else:
        pair_df = pd.DataFrame(columns=["a", "b", "dist_w0", "dist_wN", "delta"])

    out = {f"fp_window_{i}": fp for i, fp in enumerate(fp_windows)}
    out["drift_distance"] = drift_df
    out["pair_convergence"] = pair_df
    return out


# ============================================================================ #
# Phase 3 — archetypes + auto-emitted hypotheses
# ============================================================================ #

def archetypes(fp: pd.DataFrame, k: int = 3) -> Dict[str, pd.DataFrame]:
    """
    Hierarchical clustering (Ward) on z-scored fingerprint into k archetypes.
    Returns:
        assignments  : (10, 2) [archetype_id, distance_to_centroid]
        centroids    : (k, F) mean fingerprint per archetype
        linkage      : (k-1, 4) raw linkage matrix (for dendrogram plotting)
    """
    Z = zscore_fingerprint(fp).fillna(0.0)
    L = linkage(Z.values, method="ward")
    labels = fcluster(L, t=k, criterion="maxclust")
    Z["archetype"] = labels
    centroids = Z.groupby("archetype").mean()
    # distance to own centroid
    dists = []
    for cat, row in Z.iterrows():
        a = int(row["archetype"])
        d = float(np.linalg.norm(row[fp.columns].values - centroids.loc[a].values))
        dists.append({"category": cat, "archetype": a, "dist_to_centroid": d})
    assign = (
        pd.DataFrame(dists)
        .set_index("category")
        .sort_values(["archetype", "dist_to_centroid"])
    )
    return {"assignments": assign, "centroids": centroids, "linkage": pd.DataFrame(L,
            columns=["a", "b", "dist", "n"])}


def emit_hypotheses(
    fp: pd.DataFrame,
    fpz: pd.DataFrame,
    dominance: pd.DataFrame,
    comov: Dict[str, pd.DataFrame],
    regimes: Dict[str, pd.DataFrame],
    drift: Dict[str, pd.DataFrame],
    arch: Dict[str, pd.DataFrame],
) -> List[dict]:
    """
    Build a structured list of falsifiable hypotheses populated with the actual
    numbers from the analysis. Each entry records the claim, the supporting
    statistic(s), and a suggested confirmation / rejection test.
    """
    H: List[dict] = []

    # H1 — structural outliers (bigger of n_top2 or n_bot2)
    dom = dominance.copy()
    dom["dominance_score"] = dom["n_top2"] - dom["n_bot2"]
    extremes = pd.concat(
        [dom.nlargest(2, "dominance_score"), dom.nsmallest(2, "dominance_score")]
    )
    for cat, row in extremes.iterrows():
        sign = "dominates" if row["dominance_score"] > 0 else "underperforms"
        H.append({
            "id": f"H_dom_{cat}",
            "type": "outlier",
            "claim": (f"Category {cat} structurally {sign} across many "
                      f"fingerprint dimensions."),
            "evidence": {
                "n_top2": int(row["n_top2"]),
                "n_bot2": int(row["n_bot2"]),
                "mean_rank": float(row["mean_rank"]),
                "dominance_score": int(row["dominance_score"]),
            },
            "test": ("Resample (bootstrap) the per-asset features within the "
                     "category and re-rank; the dominance score should remain "
                     "in the top/bottom 2 of the 10 categories in >95% of "
                     "resamples."),
        })

    # H2 — top |corr| pair, full-sample (positive and negative)
    cf = comov["corr_full"].copy()
    np.fill_diagonal(cf.values, np.nan)
    long = cf.stack().rename("corr").reset_index()
    long.columns = ["a", "b", "corr"]
    long = long[long["a"] < long["b"]]
    for tag, row in [("strong_positive", long.nlargest(1, "corr").iloc[0]),
                     ("strong_negative", long.nsmallest(1, "corr").iloc[0])]:
        H.append({
            "id": f"H_corr_{row['a']}_{row['b']}_{tag}",
            "type": "comovement",
            "claim": (f"Categories {row['a']} and {row['b']} are persistently "
                      f"{'correlated' if row['corr']>0 else 'anti-correlated'} "
                      f"({row['corr']:+.3f})."),
            "evidence": {"corr_full": float(row["corr"])},
            "test": ("Rolling-window correlation (e.g. window=2000); the rolling "
                     "estimate should not change sign anywhere in the sample."),
        })

    # H3 — sign-flipping pairs (regime-conditional comovement)
    flips = comov.get("sign_flip_pairs", pd.DataFrame())
    if not flips.empty:
        top = flips.head(2)
        for _, row in top.iterrows():
            H.append({
                "id": f"H_regime_{row['a']}_{row['b']}",
                "type": "conditional",
                "claim": (f"Co-movement of {row['a']} and {row['b']} is regime-"
                          f"dependent: sign flips between time windows."),
                "evidence": {k: float(v) for k, v in row.items()
                             if k not in ("a", "b") and isinstance(v, (int, float, np.floating))},
                "test": ("Identify a third variable (e.g. one of the other "
                         "category indices' rolling vol) that segments the "
                         "sample so that within-segment correlation is "
                         "single-signed."),
            })

    # H4 — biggest archetype-level contrast
    cents = arch["centroids"]
    if len(cents) >= 2:
        # find archetype pair with largest centroid distance + the feature
        # contributing most to that distance
        a_pairs = list(combinations(cents.index, 2))
        best = max(a_pairs, key=lambda p: float(np.linalg.norm(cents.loc[p[0]] - cents.loc[p[1]])))
        feat_diff = (cents.loc[best[0]] - cents.loc[best[1]]).abs().sort_values(ascending=False)
        top_feat = feat_diff.index[0]
        H.append({
            "id": f"H_archetypes_{best[0]}_{best[1]}",
            "type": "archetype_contrast",
            "claim": (f"Archetypes {best[0]} and {best[1]} are most strongly "
                      f"separated by '{top_feat}'."),
            "evidence": {
                "centroid_dist": float(np.linalg.norm(cents.loc[best[0]] - cents.loc[best[1]])),
                "top_feature": top_feat,
                "centroid_top_feat_a": float(cents.loc[best[0], top_feat]),
                "centroid_top_feat_b": float(cents.loc[best[1], top_feat]),
            },
            "test": ("Hold out one full day's data, re-fingerprint, re-cluster; "
                     "the archetype assignments of categories should be stable "
                     "in >=8/10 cases."),
        })

    # H5 — most temporally drifting category
    if "drift_distance" in drift and not drift["drift_distance"].empty:
        d = drift["drift_distance"].sort_values("mean_drift", ascending=False)
        top_drift = d.index[0]
        stable = d.index[-1]
        H.append({
            "id": f"H_drift_{top_drift}",
            "type": "drift",
            "claim": (f"Category {top_drift} drifts most across time "
                      f"(mean L2 drift between consecutive windows = "
                      f"{d.loc[top_drift, 'mean_drift']:.2f}); "
                      f"{stable} is most stable "
                      f"({d.loc[stable, 'mean_drift']:.2f})."),
            "evidence": {
                "drift_top": float(d.loc[top_drift, "mean_drift"]),
                "drift_stable": float(d.loc[stable, "mean_drift"]),
            },
            "test": ("Refit the windowed fingerprint with overlapping windows "
                     "(stride = window / 2) and check that the L2 trajectory "
                     "is monotonically increasing/decreasing for "
                     f"{top_drift} (i.e. directional, not random)."),
        })

    return H


# ============================================================================ #
# Driver
# ============================================================================ #

def run_full_analysis(
    prices_dir: str | Path = "data/prices",
    cluster_map_path: str | Path = "data/derived/cluster_map.json",
    vol_window: int = 200,
    n_windows: int = 4,
    n_archetypes: int = 3,
    n_corr_splits: int = 2,
) -> dict:
    """End-to-end: returns a flat dict of artifacts ready to be saved."""
    t0 = time.perf_counter()
    log.info("loading panel + cluster map")
    prices, clusters = load_panel(prices_dir, cluster_map_path)
    log.info("panel: T=%d, N=%d, %d categories", *prices.shape, len(clusters))

    # ---- Phase 1
    log.info("[1] fingerprinting categories")
    fp, per_asset = category_fingerprint(prices, clusters, vol_window=vol_window)
    fpz = zscore_fingerprint(fp)

    # ---- Phase 2a
    log.info("[2a] dominance ranking")
    ranks, dom_summary = dominance_ranks(fp)

    # ---- Phase 2b
    log.info("[2b] category indices + comovement")
    indices = category_indices(prices, clusters)
    comov = comovement(indices, n_splits=n_corr_splits, use_returns=True)

    # ---- Phase 2c
    log.info("[2c] regime-conditional fingerprints")
    regimes = regime_fingerprints(prices, clusters, indices, vol_window=vol_window)

    # ---- Phase 2d
    log.info("[2d] temporal drift over %d windows", n_windows)
    drift = temporal_drift(prices, clusters, n_windows=n_windows,
                           vol_window=max(50, vol_window // 2))

    # ---- Phase 3
    log.info("[3] archetypes + hypotheses")
    arch = archetypes(fp, k=n_archetypes)
    hypotheses = emit_hypotheses(fp, fpz, dom_summary, comov, regimes, drift, arch)

    log.info("done in %.1fs", time.perf_counter() - t0)

    artifacts = {
        # Phase 1
        "fingerprint_raw": fp,
        "fingerprint_zscore": fpz,
        "per_asset_features": per_asset,
        # Phase 2a
        "dominance_ranks": ranks,
        "dominance_summary": dom_summary,
        # Phase 2b
        "category_indices": indices,
        "corr_full": comov["corr_full"],
        "sign_flip_pairs": comov["sign_flip_pairs"],
        **{k: v for k, v in comov.items() if k.startswith("corr_split_")},
        # Phase 2c
        **regimes,
        # Phase 2d
        **drift,
        # Phase 3
        "archetype_assignments": arch["assignments"],
        "archetype_centroids": arch["centroids"],
        "archetype_linkage": arch["linkage"],
    }
    meta = {
        "pipeline": "cluster_analysis",
        "n_categories": len(clusters),
        "n_assets_total": int(per_asset.shape[0]),
        "T": int(prices.shape[0]),
        "vol_window": vol_window,
        "n_windows_drift": n_windows,
        "n_archetypes": n_archetypes,
        "n_corr_splits": n_corr_splits,
        "categories": list(clusters.keys()),
        "features": list(ALL_FEATURES),
        "hypotheses": hypotheses,   # populated from real numbers
    }
    return {"artifacts": artifacts, "meta": meta}


if __name__ == "__main__":
    from io_results import make_run_id, save_run

    out = run_full_analysis()
    run_dir = save_run("results_clusters",
                       out["artifacts"], meta=out["meta"],
                       run_id=make_run_id())
    print(f"\nResults saved in: {run_dir}/")
    print("Hypotheses emitted (also persisted under meta.hypotheses):")
    for h in out["meta"]["hypotheses"]:
        print(f"  - [{h['type']}] {h['claim']}")
