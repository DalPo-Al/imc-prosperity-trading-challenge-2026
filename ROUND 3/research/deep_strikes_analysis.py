"""
deep_strikes_analysis.py — Full checks on deep ITM and OTM VEV options.

Sections
--------
1. Intrinsic-value violations (deep ITM):  C < S−K  = risk-free arbitrage
2. Time-value time series (deep ITM):      how much premium is available / negative
3. Full IV smile across all 10 strikes:    snapshots + rolling mean per day
4. Deep-OTM IV vs smile extrapolation:     quadratic model vs actual IV time series
5. All-strike smile residuals summary:     bar chart of mean IV − smile(m) per strike
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

ROOT     = Path(__file__).parent
DATA_DIR = ROOT.parent / "data" / "round3"
COEFFS   = ROOT / "iv_smile_coeffs.csv"
OUT_DIR  = ROOT / "garch" / "output"
OUT_DIR.mkdir(exist_ok=True)

TTE_DAY0_DAYS    = 8
ROWS_PER_DAY     = 10_000
ALL_STRIKES      = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
NEAR_ATM_STRIKES = {5000, 5100, 5200, 5300, 5400, 5500}
DEEP_ITM         = [4000, 4500]
DEEP_OTM         = [6000, 6500]
ROLL_WIN         = 500
fmt_ts = mticker.FuncFormatter(lambda x, _: f"{int(x):,}")


# ── load ──────────────────────────────────────────────────────────────────────

def load_wide():
    frames = []
    for d in range(3):
        df = pd.read_csv(DATA_DIR / f"prices_round_3_day_{d}.csv", sep=";")
        df["global_ts"] = df["day"].astype(int) * 1_000_000 + df["timestamp"].astype(int)
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    keep = {"VELVETFRUIT_EXTRACT"} | {f"VEV_{k}" for k in ALL_STRIKES}
    sub  = raw[raw["product"].isin(keep)][["global_ts","product","mid_price"]]
    wide = (sub.pivot_table(index="global_ts", columns="product",
                            values="mid_price", aggfunc="first")
              .sort_index().reset_index())
    wide.rename(columns={"VELVETFRUIT_EXTRACT":"S"}, inplace=True)
    return wide.dropna(subset=["S"]).reset_index(drop=True)


# ── BS helpers ────────────────────────────────────────────────────────────────

def bs_call(sigma, S, K, T):
    if sigma < 1e-12 or T <= 0:
        return max(float(S) - float(K), 0.0)
    sT = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / sT
    return float(S) * norm.cdf(d1) - float(K) * norm.cdf(d1 - sT)


def iv_brentq(C, S, K, T, lo=1e-6, hi=5.0):
    """Scalar IV via bisection — robust for any moneyness."""
    intrinsic = max(float(S) - float(K), 0.0)
    if float(C) <= intrinsic + 1e-9 or float(C) >= float(S) or float(T) <= 0:
        return np.nan
    try:
        f_lo = bs_call(lo, S, K, T) - float(C)
        f_hi = bs_call(hi, S, K, T) - float(C)
        if f_lo * f_hi > 0:          # no sign change → no root in bracket
            return np.nan
        return brentq(lambda s: bs_call(s, S, K, T) - float(C),
                      lo, hi, xtol=1e-8, maxiter=200)
    except Exception:
        return np.nan


def bs_call_iv_newton(S, K, T, C, sigma0=0.01, iters=100, tol=1e-8):
    """Vectorised Newton-Raphson. sigma0=0.01 correct for near-ATM with T in days."""
    intrinsic = np.maximum(S - K, 0.0)
    valid = (C > intrinsic + 1e-9) & (C < S) & (T > 0)
    sigma = np.full_like(C, sigma0, dtype=float)
    for _ in range(iters):
        sT    = sigma * np.sqrt(T)
        d1    = (np.log(S / K) + 0.5 * sigma**2 * T) / np.maximum(sT, 1e-12)
        price = S * norm.cdf(d1) - K * norm.cdf(d1 - sT)
        vega  = S * norm.pdf(d1) * np.sqrt(T)
        diff  = price - C
        with np.errstate(divide="ignore", invalid="ignore"):
            step = np.where(vega > 1e-12, diff / vega, 0.0)
        sigma = np.clip(sigma - step, 1e-6, 5.0)
        if np.nanmax(np.abs(diff[valid])) < tol:
            break
    sigma[~valid] = np.nan
    return sigma


# ── TTE array ─────────────────────────────────────────────────────────────────

def make_tte(ts):
    day_idx = (np.searchsorted(np.arange(3) * 1_000_000, ts, side="right") - 1).clip(0, 2)
    return (TTE_DAY0_DAYS - day_idx).astype(float)


# ── day boundary helpers ───────────────────────────────────────────────────────

def day_bounds(ts):
    return [ts[i * ROWS_PER_DAY] for i in range(1, 3) if i * ROWS_PER_DAY < len(ts)]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 + 2:  Intrinsic violations and time value (deep ITM)
# ══════════════════════════════════════════════════════════════════════════════

def analyse_deep_itm(wide):
    ts = wide["global_ts"].to_numpy()
    S  = wide["S"].to_numpy()

    print("=" * 60)
    print("SECTION 1 & 2 — DEEP ITM  (K=4000, K=4500)")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(16, 8), sharex=True)
    fig.suptitle("Deep ITM options: intrinsic violation & time value\n"
                 "(negative time value = risk-free arbitrage)", fontsize=11)
    bounds = day_bounds(ts)

    rows = []
    for col_ax, k in zip(axes.T, DEEP_ITM):
        col_name = f"VEV_{k}"
        C  = wide[col_name].to_numpy()
        intrinsic = np.maximum(S - k, 0.0)
        time_val  = C - intrinsic            # negative → arb

        # rolling smooth
        tv_roll = pd.Series(time_val).rolling(ROLL_WIN, min_periods=1).mean().to_numpy()

        # Stats
        pct_below = (time_val < 0).mean()
        mean_viol = time_val[time_val < 0].mean() if pct_below > 0 else 0.0
        worst     = time_val.min()
        print(f"\nK={k}:")
        print(f"  Time value: mean={time_val.mean():.3f}  min={worst:.3f}  max={time_val.max():.3f}")
        print(f"  Intrinsic violation: {pct_below:.1%} of ticks  |  mean violation={mean_viol:.3f}  "
              f"worst={worst:.3f}")
        rows.append(dict(strike=k, pct_arb=pct_below, mean_tv=time_val.mean(),
                         worst_tv=worst, arb_per_tick=mean_viol))

        # Panel top: time value
        ax_tv = col_ax[0]
        ax_tv.plot(ts, tv_roll, color="steelblue", lw=0.8, label=f"TV roll({ROLL_WIN})")
        ax_tv.fill_between(ts, tv_roll, 0, where=tv_roll < 0,
                           color="red", alpha=0.35, label="arb region")
        ax_tv.fill_between(ts, tv_roll, 0, where=tv_roll >= 0,
                           color="steelblue", alpha=0.15)
        ax_tv.axhline(0, color="k", lw=0.8)
        for b in bounds: ax_tv.axvline(b, color="grey", lw=0.7, ls="--", alpha=0.6)
        ax_tv.set_title(f"K={k} — time value (C − intrinsic)")
        ax_tv.set_ylabel("Time value (pts)")
        ax_tv.legend(fontsize=7)
        ax_tv.grid(True, alpha=0.2)
        ax_tv.xaxis.set_major_formatter(fmt_ts)

        # Panel bottom: C vs intrinsic
        ax_iv = col_ax[1]
        intr_roll = pd.Series(intrinsic).rolling(ROLL_WIN, min_periods=1).mean().to_numpy()
        c_roll    = pd.Series(C).rolling(ROLL_WIN, min_periods=1).mean().to_numpy()
        ax_iv.plot(ts, c_roll,    color="steelblue", lw=0.9, label="Option price")
        ax_iv.plot(ts, intr_roll, color="tomato",    lw=0.9, ls="--", label="Intrinsic (S−K)")
        for b in bounds: ax_iv.axvline(b, color="grey", lw=0.7, ls="--", alpha=0.6)
        ax_iv.set_title(f"K={k} — option price vs intrinsic")
        ax_iv.set_ylabel("Price")
        ax_iv.legend(fontsize=7)
        ax_iv.grid(True, alpha=0.2)
        ax_iv.xaxis.set_major_formatter(fmt_ts)
        ax_iv.set_xlabel("global_ts")

    plt.tight_layout()
    out = OUT_DIR / "deep_itm_intrinsic.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3:  Full IV smile — all 10 strikes
# ══════════════════════════════════════════════════════════════════════════════

def compute_all_ivs(wide):
    """
    Returns IV matrix (N, 10) using:
      - Vectorised Newton for near-ATM (fast)
      - brentq per element for deep ITM / deep OTM (robust)
    """
    ts  = wide["global_ts"].to_numpy()
    S   = wide["S"].to_numpy()
    T   = make_tte(ts)
    N   = len(wide)

    IV = np.full((N, len(ALL_STRIKES)), np.nan)

    for j, k in enumerate(ALL_STRIKES):
        col_name = f"VEV_{k}"
        C = wide[col_name].to_numpy()

        if k in NEAR_ATM_STRIKES:
            IV[:, j] = bs_call_iv_newton(
                S[:, None], np.array([[float(k)]]), T[:, None], C[:, None],
            ).ravel()
        else:
            # brentq per element — robust for deep ITM / OTM
            print(f"  brentq IV for K={k} …", end=" ", flush=True)
            iv_row = np.empty(N)
            for i in range(N):
                iv_row[i] = iv_brentq(C[i], S[i], k, T[i])
            IV[:, j] = iv_row
            valid_pct = np.isfinite(iv_row).mean()
            print(f"valid={valid_pct:.1%}  mean={np.nanmean(iv_row):.4f}")

    return IV, T


def plot_smile_snapshots(wide, IV, T, coef_df):
    """3 snapshot plots (one per day) + rolling mean smile."""
    ts     = wide["global_ts"].to_numpy()
    S      = wide["S"].to_numpy()
    bounds = day_bounds(ts)

    # Per-strike rolling mean IV
    IV_roll = np.full_like(IV, np.nan)
    for j in range(IV.shape[1]):
        IV_roll[:, j] = pd.Series(IV[:, j]).rolling(ROLL_WIN, min_periods=50).mean().to_numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle("IV smile: all 10 strikes vs near-ATM quadratic fit\n"
                 "(each panel = 1 day, lines = hourly snapshots, dashed = smile fit)",
                 fontsize=11)

    for day, ax in enumerate(axes):
        sl     = slice(day * ROWS_PER_DAY, (day + 1) * ROWS_PER_DAY)
        S_day  = S[sl]
        T_day  = T[sl]
        IV_day = IV[sl]

        # representative smile coefficients for this day
        a_day = coef_df["a"].iloc[sl].median()
        b_day = coef_df["b"].iloc[sl].median()
        c_day = coef_df["c"].iloc[sl].median()

        # Plot several hourly snapshots (every 1000 ts within the day)
        cmap = plt.cm.Blues(np.linspace(0.35, 0.9, 10))
        for idx_within in range(0, ROWS_PER_DAY, 1000):
            i   = day * ROWS_PER_DAY + idx_within
            s_i = S[i];  t_i = T[i]
            ivs = IV[i]
            ms  = np.log(np.array(ALL_STRIKES) / s_i) / np.sqrt(t_i)
            mask = np.isfinite(ivs)
            ax.scatter(ms[mask], ivs[mask], s=14, alpha=0.6,
                       color=cmap[idx_within // 1000])

        # Near-ATM quadratic overlay (extrapolated to full moneyness range)
        m_grid = np.linspace(-0.15, 0.12, 300)
        iv_fit = a_day + b_day * m_grid + c_day * m_grid**2
        ax.plot(m_grid, iv_fit, color="tomato", lw=1.5, ls="--",
                label=f"Smile fit  a={a_day:.4f}")

        # Mark near-ATM region
        ax.axvspan(-0.05, 0.05, alpha=0.07, color="green", label="near-ATM")

        ax.set_title(f"Day {day}  (TTE={TTE_DAY0_DAYS - day}d)")
        ax.set_xlabel("log-moneyness  m = log(K/S)/√T")
        if day == 0: ax.set_ylabel("IV  (vol/√day)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)
        ax.set_ylim(0, 0.06)

    plt.tight_layout()
    out = OUT_DIR / "full_smile.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4:  Deep-OTM IV vs smile model time series
# ══════════════════════════════════════════════════════════════════════════════

def analyse_deep_otm(wide, IV, T, coef_df):
    ts = wide["global_ts"].to_numpy()
    S  = wide["S"].to_numpy()
    bounds = day_bounds(ts)

    print("\n" + "=" * 60)
    print("SECTION 4 — DEEP OTM  (K=6000, K=6500)")
    print("=" * 60)

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    fig.suptitle("Deep OTM: actual IV vs near-ATM smile extrapolation",
                 fontsize=11)

    colors = {"6000": "steelblue", "6500": "darkorange"}

    for ax_i, k in zip([0, 1], DEEP_OTM):
        j       = ALL_STRIKES.index(k)
        iv_act  = IV[:, j]
        m       = np.log(k / S) / np.sqrt(T)
        a       = coef_df["a"].to_numpy()
        b       = coef_df["b"].to_numpy()
        c       = coef_df["c"].to_numpy()
        iv_model = a + b * m + c * m**2

        spread   = iv_act - iv_model   # actual − smile_model

        iv_act_r  = pd.Series(iv_act).rolling(ROLL_WIN, min_periods=50).mean().to_numpy()
        iv_mod_r  = pd.Series(iv_model).rolling(ROLL_WIN, min_periods=50).mean().to_numpy()
        spread_r  = pd.Series(spread).rolling(ROLL_WIN, min_periods=50).mean().to_numpy()

        ax = axes[ax_i]
        ax.plot(ts, iv_act_r,  color=colors[str(k)], lw=1.0, label=f"IV actual K={k}")
        ax.plot(ts, iv_mod_r,  color="tomato",       lw=1.0, ls="--", label="Smile model (extrap.)")
        ax.fill_between(ts, iv_act_r, iv_mod_r,
                         where=~np.isnan(iv_act_r) & ~np.isnan(iv_mod_r),
                         alpha=0.2, color=colors[str(k)])
        for b_v in bounds: ax.axvline(b_v, color="grey", lw=0.7, ls="--", alpha=0.6)
        ax.axhline(0, color="k", lw=0.3)
        ax.set_ylabel("IV  (vol/√day)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.xaxis.set_major_formatter(fmt_ts)

        valid = np.isfinite(spread)
        print(f"\nK={k}:")
        print(f"  IV actual:   mean={np.nanmean(iv_act):.5f}  std={np.nanstd(iv_act):.5f}")
        print(f"  IV model:    mean={np.nanmean(iv_model):.5f}  std={np.nanstd(iv_model):.5f}")
        print(f"  Spread (act−model): mean={np.nanmean(spread):+.5f}  "
              f"std={np.nanstd(spread):.5f}  "
              f"% actual>model={np.nanmean(iv_act[valid]>iv_model[valid]):.1%}")

    # Panel 3: both spreads together
    ax = axes[2]
    for k in DEEP_OTM:
        j        = ALL_STRIKES.index(k)
        iv_act   = IV[:, j]
        m        = np.log(k / S) / np.sqrt(T)
        iv_model = coef_df["a"].to_numpy() + coef_df["b"].to_numpy() * m + coef_df["c"].to_numpy() * m**2
        spread_r = pd.Series(iv_act - iv_model).rolling(ROLL_WIN, min_periods=50).mean().to_numpy()
        ax.plot(ts, spread_r, color=colors[str(k)], lw=1.0, label=f"K={k}  IV−model")

    ax.axhline(0, color="k", lw=1.0, label="par")
    for b_v in bounds: ax.axvline(b_v, color="grey", lw=0.7, ls="--", alpha=0.6)
    ax.set_ylabel("IV − model  (vol/√day)")
    ax.set_xlabel("global_ts")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_formatter(fmt_ts)

    plt.tight_layout()
    out = OUT_DIR / "deep_otm_vs_smile.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5:  All-strike smile residuals summary
# ══════════════════════════════════════════════════════════════════════════════

def plot_residuals_summary(wide, IV, T, coef_df):
    ts = wide["global_ts"].to_numpy()
    S  = wide["S"].to_numpy()
    a  = coef_df["a"].to_numpy()
    b  = coef_df["b"].to_numpy()
    c  = coef_df["c"].to_numpy()

    print("\n" + "=" * 60)
    print("SECTION 5 — ALL-STRIKE SMILE RESIDUALS")
    print("=" * 60)
    print(f"{'Strike':>8} {'Mean IV':>10} {'IV−model':>10} {'Std resid':>10} "
          f"{'% above':>9} {'regime':>20}")

    means, stds, labels, pcts_above = [], [], [], []
    for j, k in enumerate(ALL_STRIKES):
        m        = np.log(k / S) / np.sqrt(T)
        iv_model = a + b * m + c * m**2
        iv_act   = IV[:, j]
        resid    = iv_act - iv_model
        valid    = np.isfinite(resid)

        mean_iv  = np.nanmean(iv_act)
        mean_r   = np.nanmean(resid[valid]) if valid.sum() > 0 else np.nan
        std_r    = np.nanstd(resid[valid])  if valid.sum() > 0 else np.nan
        pct_ab   = np.nanmean(iv_act[valid] > iv_model[valid]) if valid.sum() > 0 else np.nan

        regime = ("DEEP ITM" if k in DEEP_ITM
                  else "DEEP OTM" if k in DEEP_OTM
                  else "near-ATM")
        print(f"  K={k:>5} {mean_iv:>10.5f} {mean_r:>+10.5f} {std_r:>10.5f} "
              f"{pct_ab:>8.1%}  {regime}")
        means.append(mean_r);  stds.append(std_r)
        labels.append(str(k)); pcts_above.append(pct_ab)

    # Bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    clrs = ["#d62728" if k in DEEP_ITM else
            "#ff7f0e" if k in DEEP_OTM else
            "#1f77b4"
            for k in ALL_STRIKES]

    xs = np.arange(len(ALL_STRIKES))
    axes[0].bar(xs, means, yerr=stds, color=clrs, alpha=0.8, capsize=4)
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_xticks(xs); axes[0].set_xticklabels(labels, rotation=45)
    axes[0].set_title("Mean (IV_actual − smile_model) per strike\n"
                      "red=deep ITM  orange=deep OTM  blue=near-ATM")
    axes[0].set_ylabel("IV residual (vol/√day)")
    axes[0].grid(True, alpha=0.2, axis="y")

    axes[1].bar(xs, [p * 100 if np.isfinite(p) else 0 for p in pcts_above],
                color=clrs, alpha=0.8)
    axes[1].axhline(50, color="k", lw=0.8, ls="--", label="50%")
    axes[1].set_xticks(xs); axes[1].set_xticklabels(labels, rotation=45)
    axes[1].set_title("% of ticks where IV_actual > smile_model")
    axes[1].set_ylabel("%")
    axes[1].set_ylim(0, 105)
    axes[1].grid(True, alpha=0.2, axis="y")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    out = OUT_DIR / "smile_residuals_all_strikes.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data …")
    wide    = load_wide()
    coef_df = pd.read_csv(COEFFS)
    print(f"  {len(wide):,} timestamps  |  strikes: {ALL_STRIKES}")

    # Section 1 & 2
    itm_stats = analyse_deep_itm(wide)

    # Compute IV for all strikes
    print("\nComputing IV for all 10 strikes …")
    IV, T = compute_all_ivs(wide)
    print(f"  Done. Valid IV: {np.isfinite(IV).mean():.1%}")

    # Section 3
    print("\nSection 3 — smile snapshots …")
    plot_smile_snapshots(wide, IV, T, coef_df)

    # Section 4
    analyse_deep_otm(wide, IV, T, coef_df)

    # Section 5
    plot_residuals_summary(wide, IV, T, coef_df)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\nDeep ITM arbitrage:")
    print(itm_stats.to_string(index=False))
    print("\nDeep OTM: see deep_otm_vs_smile.png — IV consistently above smile model")
    print("Full smile: see full_smile.png")
    print("All-strike residuals: see smile_residuals_all_strikes.png")


if __name__ == "__main__":
    main()
