"""
IV per strike (5000-5500) over time, with EMA overlay.
EMA span chosen to reveal 4-6 cycle modes across 30k timestamps.
BS inversion matches iv_smile_analysis.py exactly (sigma0=0.01, T in days).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- config ----------
ROOT = Path(__file__).parent
DATA = ROOT.parent / "data" / "round3"

TTE_DAY0_DAYS = 8
ROWS_PER_DAY = 10_000
NEWTON_ITERS = 50
NEWTON_TOL = 1e-8

STRIKES = [5000, 5100, 5200, 5300, 5400, 5500]
STRIKE_COLS = [f"VEV_{k}" for k in STRIKES]
COLORS = ["#e41a1c", "#ff7f00", "#4daf4a", "#377eb8", "#984ea3", "#a65628"]

# EMA: target 4-6 peaks over 30k rows → period ~5000-7500 ticks
# span=500 smooths tick noise, passes slow modes
EMA_SPAN = 500


# ---------- load ----------
opt = pd.read_csv(DATA / "clean_options_chain.csv")
und = pd.read_csv(DATA / "clean_VELVETFRUIT_EXTRACT.csv").rename(
    columns={"mid_price": "S"}
)
df = (
    opt.merge(und, on="global_ts", how="inner")
    .sort_values("global_ts")
    .reset_index(drop=True)
)

N = len(df)
S = df["S"].to_numpy()[:, None]
C = df[STRIKE_COLS].to_numpy().astype(float)
K = np.array(STRIKES, dtype=float)[None, :]

tte_raw = np.clip(TTE_DAY0_DAYS * ROWS_PER_DAY - np.arange(N), 1, None)
T = (tte_raw / ROWS_PER_DAY)[:, None]  # days, shape (N,1)
ts = df["global_ts"].to_numpy()


# ---------- BS IV via Newton (identical to iv_smile_analysis.py) ----------
def bs_call_iv(S, K, T, C, sigma0=0.01, iters=NEWTON_ITERS, tol=NEWTON_TOL):
    intrinsic = np.maximum(S - K, 0.0)
    valid = (C > intrinsic + 1e-9) & (C < S) & (T > 0)
    sigma = np.full_like(C, sigma0, dtype=float)
    for _ in range(iters):
        sT = sigma * np.sqrt(T)
        d1 = (np.log(S / K) + 0.5 * sigma * sigma * T) / sT
        d2 = d1 - sT
        price = S * norm.cdf(d1) - K * norm.cdf(d2)
        vega = S * norm.pdf(d1) * np.sqrt(T)
        diff = price - C
        with np.errstate(divide="ignore", invalid="ignore"):
            step = np.where(vega > 1e-12, diff / vega, 0.0)
        sigma = np.clip(sigma - step, 1e-6, 5.0)
        if np.nanmax(np.abs(diff[valid])) < tol:
            break
    sigma[~valid] = np.nan
    return sigma


print(f"[load] N={N} rows, {len(STRIKES)} strikes")
print("[BS]  inverting IV ...")
IV = bs_call_iv(S, K, T, C)
print(
    "[BS]  done.  NaN frac: "
    + ", ".join(f"K{k}:{np.isnan(IV[:,i]).mean():.2%}" for i, k in enumerate(STRIKES))
)

# EMA
IV_df = pd.DataFrame(IV, columns=STRIKE_COLS)
EMA_df = IV_df.ewm(span=EMA_SPAN, min_periods=EMA_SPAN // 2).mean()


# ---------- plot 1: all series on single axes ----------
fig, ax = plt.subplots(figsize=(16, 6))
for i, (col, k) in enumerate(zip(STRIKE_COLS, STRIKES)):
    ax.plot(ts, IV_df[col], color=COLORS[i], alpha=0.20, linewidth=0.5)
    ax.plot(ts, EMA_df[col], color=COLORS[i], linewidth=1.8, label=f"K={k}")

ax.set_xlabel("Timestamp")
ax.set_ylabel("Implied Volatility (per √day)")
ax.set_title(f"IV per strike 5000–5500  |  EMA span={EMA_SPAN}")
ax.legend(loc="upper right", framealpha=0.85)
ax.grid(True, alpha=0.3)
plt.tight_layout()
out1 = ROOT / "iv_per_strike_combined.png"
fig.savefig(out1, dpi=150)
plt.close(fig)
print(f"[saved] {out1}")


# ---------- plot 2: 2×3 subplots ----------
fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=True)
axes_flat = axes.flatten()
for i, (col, k) in enumerate(zip(STRIKE_COLS, STRIKES)):
    ax = axes_flat[i]
    ax.plot(ts, IV_df[col], color=COLORS[i], alpha=0.20, linewidth=0.5)
    ax.plot(ts, EMA_df[col], color=COLORS[i], linewidth=1.8)
    ax.set_title(f"K = {k}", fontsize=11)
    ax.set_ylabel("IV (per √day)")
    ax.grid(True, alpha=0.3)

for ax in axes[-1]:
    ax.set_xlabel("Timestamp")

fig.suptitle(f"IV per strike  |  raw (faded) + EMA-{EMA_SPAN} (solid)", fontsize=13)
plt.tight_layout()
out2 = ROOT / "iv_per_strike_subplots.png"
fig.savefig(out2, dpi=150)
plt.close(fig)
print(f"[saved] {out2}")


# ---------- Kalman: local linear trend ----------
# State: [level, slope].  level_{t+1} = level_t + slope_t + w1
#                         slope_{t+1} = slope_t + w2
# Tune ratio R_OBS / Q_SLOPE for smoothness (higher = smoother).
KF_Q_LEVEL = 1e-10  # near-zero: level changes only through slope
KF_Q_SLOPE = 1e-13  # slope drifts very slowly
KF_R_OBS = 1e-4  # observation noise (~tick-level IV noise)

_F = np.array([[1.0, 1.0], [0.0, 1.0]])
_H = np.array([[1.0, 0.0]])


def kalman_llt(
    series: np.ndarray, q_level=KF_Q_LEVEL, q_slope=KF_Q_SLOPE, r_obs=KF_R_OBS
) -> np.ndarray:
    """Local-linear-trend Kalman filter. Returns filtered level array."""
    Q = np.diag([q_level, q_slope])
    n = len(series)
    first_valid = next((i for i, v in enumerate(series) if not np.isnan(v)), 0)
    x = np.array([series[first_valid], 0.0])
    P = np.eye(2) * 1e-4
    levels = np.full(n, np.nan)
    for t in range(n):
        y = series[t]
        x_p = _F @ x
        P_p = _F @ P @ _F.T + Q
        if np.isnan(y):
            x, P = x_p, P_p
        else:
            inn = y - x_p[0]
            S = P_p[0, 0] + r_obs
            k = P_p[:, 0] / S
            x = x_p + k * inn
            P = P_p - np.outer(k, P_p[0])
        levels[t] = x[0]
    return levels


def plot_kalman_subplots(
    q_level=KF_Q_LEVEL, q_slope=KF_Q_SLOPE, r_obs=KF_R_OBS
) -> Path:
    """2x3 subplot grid: raw IV (faded) + Kalman level (solid). Mirrors EMA subplot layout."""
    KF_df = pd.DataFrame(
        {
            col: kalman_llt(IV[:, i], q_level, q_slope, r_obs)
            for i, col in enumerate(STRIKE_COLS)
        },
    )

    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=True)
    for i, (col, k) in enumerate(zip(STRIKE_COLS, STRIKES)):
        ax = axes.flatten()[i]
        ax.plot(ts, IV_df[col], color=COLORS[i], alpha=0.20, linewidth=0.5)
        ax.plot(ts, KF_df[col], color=COLORS[i], linewidth=1.8)
        ax.set_title(f"K = {k}", fontsize=11)
        ax.set_ylabel("IV (per √day)")
        ax.grid(True, alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("Timestamp")

    fig.suptitle("IV per strike  |  raw (faded) + Kalman LLT (solid)", fontsize=13)
    plt.tight_layout()
    out = ROOT / "iv_per_strike_kalman_subplots.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[saved] {out}")
    return out


plot_kalman_subplots()
