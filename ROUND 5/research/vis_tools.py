import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------------------------ #
# Save

files = sorted(Path("data/prices/").glob("*.csv"))

# Stack all rows from all days
raw = pd.concat((pd.read_csv(f, sep=";") for f in files), ignore_index=True)

# Build panel
mid_df = raw.pivot(
    index=["day", "timestamp"], columns="product", values="mid_price"
).sort_index()

mid_df.columns.name = None
log_diff_df = 100 * np.log(mid_df).diff().iloc[1:]

# ------------------------------------------------------------------------------- #

KEYS = [
    "GALAXY_SOUNDS",
    "MICROCHIP",
    "OXYGEN_SHAKE",
    "PANEL",
    "PEBBLES",
    "ROBOT",
    "SLEEP_POD",
    "SNACKPACK",
    "TRANSLATOR",
    "UV_VISOR",
]


def plot_all(mid_prices_df, include_keys):

    plt.figure(figsize=(14, 6))

    # continuous event-time axis (avoids timestamp reset each day)
    x = np.arange(len(mid_prices_df))

    for key in include_keys:

        matching_cols = [col for col in mid_prices_df.columns if col.startswith(key)]

        if not matching_cols:
            print(f"No columns found for {key}")
            continue

        # row-wise average across the 5 assets in the group
        avg_series = mid_prices_df[matching_cols].mean(axis=1)

        # handle possible NaNs
        valid = avg_series.notna()

        plt.plot(x[valid], avg_series[valid], label=key, linewidth=1)

    plt.title("Average Mid Price by Keyword")
    plt.xlabel("Sequential Time")
    plt.ylabel("Average Mid Price")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_clusters(mid_prices_df, ema_window=20, std_mult=2.0):

    def _plot_single_cluster(mid_prices_df, key, ema_window=20, std_mult=2.0):

        cols = [c for c in mid_prices_df.columns if c.startswith(key)]
        if not cols:
            return

        x = np.arange(len(mid_prices_df))
        sub = mid_prices_df[cols]

        # cluster average
        avg = sub.mean(axis=1)

        # EMA + STD
        ema = avg.ewm(span=ema_window, adjust=False).mean()
        std = avg.rolling(ema_window).std()

        upper = ema + std_mult * std
        lower = ema - std_mult * std

        valid = avg.notna()

        plt.figure(figsize=(14, 5))

        # individual assets
        for c in cols:
            plt.plot(x[valid], sub[c][valid], alpha=0.2, linewidth=1)

        # EMA
        plt.plot(x[valid], ema[valid], linewidth=2, linestyle="--", label="EMA")

        # band
        plt.fill_between(x[valid], lower[valid], upper[valid], alpha=0.15)

        plt.title(f"Cluster: {key}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    keys = sorted(set(col.split("_")[0] for col in mid_prices_df.columns))

    for key in keys:
        _plot_single_cluster(
            mid_prices_df, key, ema_window=ema_window, std_mult=std_mult
        )


def plot_some(mid_prices_df, keys):

    plt.figure(figsize=(14, 6))

    # continuous event-time axis (avoids timestamp reset each day)
    x = np.arange(len(mid_prices_df))

    for key in keys:
        plt.plot(x, mid_prices_df[key], label=key, linewidth=1)

    plt.title("Average Mid Price by Keyword")
    plt.xlabel("Sequential Time")
    plt.ylabel("Average Mid Price")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def quantify_coherence(mid_prices_df, key):
    """
    Coherence = avg pairwise correlation within cluster.
    Higher = assets move together. Range [0,1].
    """
    cols = [c for c in mid_prices_df.columns if c.startswith(key)]
    if len(cols) < 2:
        return np.nan

    # drop NaNs for correlation calc
    sub = mid_prices_df[cols].dropna()
    if len(sub) < 2:
        return np.nan

    corr_matrix = sub.corr()
    # upper triangle (avoid diagonal and duplicates)
    mask = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
    pairwise_corrs = corr_matrix.values[mask]

    # clamp to [0,1] and return mean
    pairwise_corrs = np.clip(pairwise_corrs, -1, 1)
    return pairwise_corrs.mean()


def quantify_dispersion(mid_prices_df, key):
    """
    Dispersion = avg std within cluster from cluster mean.
    Lower = tighter cluster.
    """
    cols = [c for c in mid_prices_df.columns if c.startswith(key)]
    if not cols:
        return np.nan

    sub = mid_prices_df[cols]
    cluster_mean = sub.mean(axis=1)

    # distance from mean for each asset each time
    diffs = sub.subtract(cluster_mean, axis=0)

    # std of those differences
    return diffs.std().mean()


def quantify_mode(mid_prices_df, key):
    """
    Mode = principal eigenvalue of cluster covariance.
    Higher = cluster moves in dominant direction.
    """
    cols = [c for c in mid_prices_df.columns if c.startswith(key)]
    if len(cols) < 2:
        return np.nan

    sub = mid_prices_df[cols].dropna()
    if len(sub) < 2:
        return np.nan

    cov_matrix = sub.cov()
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    return eigenvalues[-1]  # largest eigenvalue


def plot_coherence(mid_prices_df, window=20):
    """
    Time series of coherence per cluster (rolling window).
    """
    keys = sorted(set(col.split("_")[0] for col in mid_prices_df.columns))
    x = np.arange(len(mid_prices_df))

    plt.figure(figsize=(14, 6))

    for key in keys:
        cols = [c for c in mid_prices_df.columns if c.startswith(key)]
        if len(cols) < 2:
            continue

        sub = mid_prices_df[cols]
        coherence_ts = []

        for i in range(len(sub)):
            start = max(0, i - window + 1)
            window_data = sub.iloc[start : i + 1].dropna()

            if len(window_data) < 2:
                coherence_ts.append(np.nan)
                continue

            corr_matrix = window_data.corr()
            mask = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
            pairwise_corrs = corr_matrix.values[mask]
            pairwise_corrs = np.clip(pairwise_corrs, -1, 1)
            coherence_ts.append(pairwise_corrs.mean())

        valid = ~np.isnan(coherence_ts)
        plt.plot(x[valid], np.array(coherence_ts)[valid], label=key, linewidth=1.5)

    plt.title(f"Cluster Coherence Over Time (rolling window={window})")
    plt.xlabel("Sequential Time")
    plt.ylabel("Coherence")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_dispersion(mid_prices_df, window=20):
    """
    Time series of dispersion per cluster (rolling window).
    """
    keys = sorted(set(col.split("_")[0] for col in mid_prices_df.columns))
    x = np.arange(len(mid_prices_df))

    plt.figure(figsize=(14, 6))

    for key in keys:
        cols = [c for c in mid_prices_df.columns if c.startswith(key)]
        if not cols:
            continue

        sub = mid_prices_df[cols]
        dispersion_ts = []

        for i in range(len(sub)):
            start = max(0, i - window + 1)
            window_data = sub.iloc[start : i + 1]

            cluster_mean = window_data.mean(axis=1)
            diffs = window_data.subtract(cluster_mean, axis=0)
            dispersion_ts.append(diffs.std().mean())

        valid = ~np.isnan(dispersion_ts)
        plt.plot(x[valid], np.array(dispersion_ts)[valid], label=key, linewidth=1.5)

    plt.title(f"Cluster Dispersion Over Time (rolling window={window})")
    plt.xlabel("Sequential Time")
    plt.ylabel("Dispersion")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_mode(mid_prices_df, window=20):
    """
    Time series of principal eigenvalue per cluster (rolling window).
    High = dominant direction. Low = scattered.
    """
    keys = sorted(set(col.split("_")[0] for col in mid_prices_df.columns))
    x = np.arange(len(mid_prices_df))

    plt.figure(figsize=(14, 6))

    for key in keys:
        cols = [c for c in mid_prices_df.columns if c.startswith(key)]
        if len(cols) < 2:
            continue

        sub = mid_prices_df[cols]
        mode_ts = []

        for i in range(len(sub)):
            start = max(0, i - window + 1)
            window_data = sub.iloc[start : i + 1].dropna()

            if len(window_data) < 2:
                mode_ts.append(np.nan)
                continue

            cov_matrix = window_data.cov()
            eigenvalues = np.linalg.eigvalsh(cov_matrix)
            mode_ts.append(eigenvalues[-1])

        valid = ~np.isnan(mode_ts)
        plt.plot(x[valid], np.array(mode_ts)[valid], label=key, linewidth=1.5)

    plt.title(f"Cluster Mode Over Time (rolling window={window})")
    plt.xlabel("Sequential Time")
    plt.ylabel("Principal Eigenvalue")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
