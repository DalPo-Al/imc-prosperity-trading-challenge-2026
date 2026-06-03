"""
Smile reversion analysis for OptionTrader.py launch prep.

Historical data: days 0-2, TTE = 8 → 5  (data row 0 → TTE 8, as confirmed)
Live trading:    days 3-5, TTE = 5 → 2  (OptionTrader DAY=3..5, OPT_TTE_TOTAL=8)

Both use the same underlying clock: TTE = 8 - elapsed_ticks / TICKS_PER_DAY.
Historical data TTE = 8 - global_ts // 100 / TICKS_PER_DAY  (no day offset needed).
OptionTrader TTE = 8 - DAY - timestamp // 100 / TICKS_PER_DAY  (correct as-is).

Hardcoded smile: IV(m) = A + B*m + C*m²,  m = log(K/S)/sqrt(T)
                A=0.012535, B=0.002252, C=0.56799  (fit on historical data)

Outputs:
- Deviation of OLD A,B,C on correctly-TTEd historical data (should be ~0)
- Reversion-time stats per strike for de-trended deviations
- All plots saved to images/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR  = "../../data/round3"
IMG_DIR   = "images"
OPT_CSV   = f"{DATA_DIR}/clean_options_chain.csv"
UND_CSV   = f"{DATA_DIR}/clean_VELVETFRUIT_EXTRACT.csv"

os.makedirs(IMG_DIR, exist_ok=True)

# ── Smile coefficients (fit on historical data at TTE 8→5) ───────────────────
A_OLD, B_OLD, C_OLD = 0.012535, 0.002252, 0.56799

# ── TTE constants ─────────────────────────────────────────────────────────────
TTE_AT_DATA_ROW0 = 8          # confirmed: data day 0 → TTE = 8
TICKS_PER_DAY    = 10_000
TICKS_PER_TS     = 100

# Strike sets
ATM_STRIKES  = [5000, 5100, 5200, 5300, 5400, 5500]
ALL_STRIKES  = [4000, 4500] + ATM_STRIKES + [6000, 6500]


# ── BS helpers ────────────────────────────────────────────────────────────────
def smile_iv(S, K, T, a, b, c):
    m = np.log(K / S) / np.sqrt(T)
    return a + m * (b + c * m)


def bs_call(S, K, T, iv):
    sT = iv * np.sqrt(T)
    d1 = (np.log(S / K) + 0.5 * iv ** 2 * T) / sT
    d2 = d1 - sT
    return S * norm.cdf(d1) - K * norm.cdf(d2)


def bs_call_iv(S, K, T, C, iters=60, tol=1e-9):
    """Newton inversion for IV. Vectorised."""
    intrinsic = np.maximum(S - K, 0.0)
    valid = (C > intrinsic + 1e-9) & (C < S) & (T > 0)
    sigma = np.full_like(C, 0.02, dtype=float)
    for _ in range(iters):
        sT  = sigma * np.sqrt(T)
        d1  = (np.log(S / K) + 0.5 * sigma ** 2 * T) / sT
        d2  = d1 - sT
        p   = S * norm.cdf(d1) - K * norm.cdf(d2)
        veg = S * norm.pdf(d1) * np.sqrt(T)
        diff = p - C
        with np.errstate(divide="ignore", invalid="ignore"):
            step = np.where(veg > 1e-12, diff / veg, 0.0)
        sigma = np.clip(sigma - step, 1e-6, 5.0)
        if np.nanmax(np.abs(diff[valid])) < tol:
            break
    sigma[~valid] = np.nan
    return sigma


# ── Historical data TTE: row 0 → TTE 8, each tick subtracts 1/TICKS_PER_DAY ──
def data_tte(global_ts):
    ticks_elapsed = global_ts // TICKS_PER_TS
    return TTE_AT_DATA_ROW0 - ticks_elapsed / TICKS_PER_DAY


# ── Reversion analysis ────────────────────────────────────────────────────────
def episode_durations(signal, threshold):
    """Lengths of contiguous spans where |signal| > threshold."""
    above = np.abs(signal) > threshold
    durations, i, n = [], 0, len(signal)
    while i < n:
        if above[i]:
            j = i + 1
            while j < n and above[j]:
                j += 1
            durations.append(j - i)
            i = j
        else:
            i += 1
    return np.array(durations)


def acf(x, max_lag):
    s = x - x.mean()
    if s.std() == 0:
        return np.zeros(max_lag + 1)
    return np.array([1.0 if k == 0
                     else np.corrcoef(s[:-k], s[k:])[0, 1]
                     for k in range(max_lag + 1)])


def acf_halflife(x, max_lag=500):
    a = acf(x, max_lag)
    below = np.where(a < 0.5)[0]
    return float(below[0]) if len(below) > 0 else float(max_lag)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading data …")
    opt = pd.read_csv(OPT_CSV)
    und = pd.read_csv(UND_CSV).rename(columns={"mid_price": "S"})
    df  = opt.merge(und, on="global_ts", how="inner").sort_values("global_ts").reset_index(drop=True)

    gts = df["global_ts"].to_numpy()
    S   = df["S"].to_numpy()
    T   = data_tte(gts)   # historical TTE: 8 at row 0, ~5 at row 29999
    print(f"  rows={len(df)}  TTE range = [{T.min():.4f}, {T.max():.4f}] (historical: 8→5)")

    # ── 1. Deviations using OLD A,B,C (correctly TTEd) ────────────────────────
    print("\nDeviations under OLD smile (correct historical TTE 8→5):")
    dev = {}
    for K in ALL_STRIKES:
        market = df[f"VEV_{K}"].to_numpy()
        iv     = smile_iv(S, K, T, A_OLD, B_OLD, C_OLD)
        theo   = bs_call(S, K, T, iv)
        dev[K] = market - theo
        print(f"  K={K:>4}  mean={dev[K].mean():+8.3f}  std={dev[K].std():6.3f}")

    # ── 2. Mean reversion of de-trended deviations ────────────────────────────
    THRESHOLD = 1.0
    ROLL = 200
    print(f"\nReversion stats (de-trended, roll={ROLL}, threshold=±{THRESHOLD})")
    print(f"{'Strike':>6}  {'μ(dev)':>8}  {'σ(dev)':>8}  {'ACF HL':>8}  "
          f"{'#ep':>6}  {'med':>6}  {'mean':>7}  {'p90':>6}")

    eps  = {}
    hls  = {}
    detrended = {}
    for K in ALL_STRIKES:
        d  = dev[K]
        rm = pd.Series(d).rolling(ROLL, min_periods=ROLL // 2, center=True).mean().to_numpy()
        rm = np.where(np.isnan(rm), np.nanmean(d), rm)
        dt = d - rm
        detrended[K] = dt
        ep = episode_durations(dt, THRESHOLD)
        eps[K] = ep
        hl = acf_halflife(dt) if dt.std() > 0 else 0.0
        hls[K] = hl
        if len(ep) > 0:
            print(f"  {K:>4}  {d.mean():+8.3f}  {dt.std():8.3f}  {hl:8.1f}  "
                  f"{len(ep):>6}  {np.median(ep):>6.0f}  {np.mean(ep):>7.1f}  "
                  f"{np.percentile(ep,90):>6.0f}")
        else:
            print(f"  {K:>4}  {d.mean():+8.3f}  {dt.std():8.3f}  {hl:8.1f}  "
                  f"{0:>6}  {'—':>6}  {'—':>7}  {'—':>6}")

    # ── PLOTS ─────────────────────────────────────────────────────────────────
    print("\nPlotting …")

    # Figure 1: Deviation time-series per strike
    fig, axes = plt.subplots(len(ALL_STRIKES), 1, figsize=(16, 22), sharex=True)
    fig.suptitle("Market − BS(smile)  per strike  [historical TTE 8→5, A=0.012535]",
                 fontsize=11, y=0.995)
    for ax, K in zip(axes, ALL_STRIKES):
        ax.plot(gts, dev[K], lw=0.4, alpha=0.7, color="steelblue")
        ax.axhline(0, color="k", lw=0.6)
        for b in [1_000_000, 2_000_000]:
            ax.axvline(b, color="gray", lw=0.6, ls=":")
        ax.set_ylabel(f"K={K}", fontsize=8)
        ax.text(0.01, 0.97, f"μ={dev[K].mean():+.2f}  σ={dev[K].std():.2f}",
                transform=ax.transAxes, fontsize=7, va="top")
    axes[-1].set_xlabel("global_ts")
    fig.tight_layout(rect=[0, 0, 1, 0.995])
    fig.savefig(f"{IMG_DIR}/deviation_timeseries.png", dpi=140)
    plt.close(fig)
    print(f"  → {IMG_DIR}/deviation_timeseries.png")

    # Figure 2: De-trended deviation time-series (ATM only)
    fig, axes = plt.subplots(len(ATM_STRIKES), 1, figsize=(16, 14), sharex=True)
    fig.suptitle(f"De-trended deviation (rolling mean roll={ROLL})  dashed=±{THRESHOLD}",
                 fontsize=11)
    for ax, K in zip(axes, ATM_STRIKES):
        ax.plot(gts, detrended[K], lw=0.4, alpha=0.7, color="steelblue")
        ax.axhline(+THRESHOLD, color="red", lw=0.8, ls="--", alpha=0.7)
        ax.axhline(-THRESHOLD, color="red", lw=0.8, ls="--", alpha=0.7)
        ax.axhline(0, color="k", lw=0.5)
        for b in [1_000_000, 2_000_000]:
            ax.axvline(b, color="gray", lw=0.6, ls=":")
        ax.set_ylabel(f"K={K}", fontsize=9)
        ax.text(0.01, 0.97, f"HL={hls[K]:.0f}t  σ={detrended[K].std():.2f}",
                transform=ax.transAxes, fontsize=7, va="top")
    axes[-1].set_xlabel("global_ts")
    fig.tight_layout()
    fig.savefig(f"{IMG_DIR}/detrended_deviation_timeseries.png", dpi=140)
    plt.close(fig)
    print(f"  → {IMG_DIR}/detrended_deviation_timeseries.png")

    # Figure 3: Reversion-time histograms
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    fig.suptitle(f"Episode duration (ticks) per strike — |Δdev| > {THRESHOLD}", fontsize=11)
    for ax, K in zip(axes.flat, ALL_STRIKES):
        ep = eps[K]
        if len(ep) == 0:
            ax.text(0.5, 0.5, "no episodes", ha="center", va="center", transform=ax.transAxes)
        else:
            bins = min(50, max(8, len(ep) // 20))
            ax.hist(ep, bins=bins, color="steelblue", edgecolor="white", lw=0.3)
            ax.axvline(np.median(ep), color="orange", lw=1.4, ls="--",
                       label=f"med={np.median(ep):.0f}")
            ax.axvline(np.mean(ep),   color="red",    lw=1.4, ls="-",
                       label=f"mean={np.mean(ep):.0f}")
            ax.legend(fontsize=7)
            ax.set_yscale("log")
        ax.set_title(f"K={K}  n={len(ep)}", fontsize=9)
        ax.set_xlabel("ticks", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{IMG_DIR}/reversion_durations.png", dpi=140)
    plt.close(fig)
    print(f"  → {IMG_DIR}/reversion_durations.png")

    # Figure 4: ACF for ATM strikes
    MAX_LAG = 500
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.tab10(np.linspace(0, 0.6, len(ATM_STRIKES)))
    for K, col in zip(ATM_STRIKES, colors):
        a = acf(detrended[K], MAX_LAG)
        ax.plot(np.arange(MAX_LAG + 1), a, lw=1.2, color=col,
                label=f"K={K}  HL={hls[K]:.0f}t")
    ax.axhline(0.5, color="k", lw=0.8, ls="--", label="ACF=0.5")
    ax.axhline(0,   color="k", lw=0.5)
    ax.set_xlabel("lag (ticks)")
    ax.set_ylabel("ACF of de-trended deviation")
    ax.set_title("ACF of de-trended smile deviation — ATM strikes")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{IMG_DIR}/acf_detrended.png", dpi=140)
    plt.close(fig)
    print(f"  → {IMG_DIR}/acf_detrended.png")

    # Figure 5: Summary bars
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4))
    ks  = ATM_STRIKES
    hl_vals = [hls[k] for k in ks]
    ep_vals = [np.median(eps[k]) if len(eps[k]) > 0 else 0 for k in ks]
    bars = a1.bar(range(len(ks)), hl_vals, color="steelblue", edgecolor="white")
    a1.set_xticks(range(len(ks))); a1.set_xticklabels([str(k) for k in ks])
    a1.set_xlabel("Strike"); a1.set_ylabel("ACF half-life (ticks)")
    a1.set_title("ACF half-life of de-trended deviation")
    for bar, v in zip(bars, hl_vals):
        a1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, f"{v:.0f}",
                ha="center", va="bottom", fontsize=8)
    bars2 = a2.bar(range(len(ks)), ep_vals, color="darkorange", edgecolor="white")
    a2.set_xticks(range(len(ks))); a2.set_xticklabels([str(k) for k in ks])
    a2.set_xlabel("Strike"); a2.set_ylabel("Median episode length (ticks)")
    a2.set_title(f"Median reversion time  |Δdev| > {THRESHOLD}")
    for bar, v in zip(bars2, ep_vals):
        a2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, f"{v:.0f}",
                ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{IMG_DIR}/summary_bars.png", dpi=140)
    plt.close(fig)
    print(f"  → {IMG_DIR}/summary_bars.png")


if __name__ == "__main__":
    main()
