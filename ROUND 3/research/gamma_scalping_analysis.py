"""
gamma_scalping_analysis.py

Full gamma-scalping viability analysis for VEV options (Round 3).

Framework: buy options, delta-hedge, capture σ_rv > σ_iv edge.
Core condition: ½ΓS²σ²_rv·dt > |Θ|·dt  ⟺  σ_rv > σ_iv.

Sections:
  1. IV per strike
  2. BS Greeks (Δ, Γ, Θ, Vega)
  3. Realized vol: GARCH(1,1)-t + rolling windows
  4. σ_rv vs σ_iv — core condition check
  5. Gamma harvest vs theta bleed (expected)
  6. Realized gamma PnL per tick (actual ΔS, continuous hedging)
  7. Discrete hedge simulation — h ∈ {1, 5, 10, 50, 100} ticks
  8. Transaction cost breakeven
  9. Vega risk contribution
  10. Strike ranking

GARCH params from findings.md (fitted on 100×-scaled log-returns):
  μ=5.8e-5, ω=2.32e-4, α=0.10, β=0.40, ν=12
  σ_unconditional = √(ω/(1−α−β)) = 2.154 %/√day
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA = ROOT.parent.parent / "data" / "round3"
IMG  = ROOT / "images"
IMG.mkdir(exist_ok=True)

A, B, C_SMILE = 0.012535, 0.002252, 0.56799   # smile coefficients
TTE0 = 8            # TTE (days) at data row 0
TPD  = 10_000       # ticks per day
TPS  = 100          # ticks per timestamp

# GARCH(1,1)-t params (on 100×-scaled returns); σ in per-√day after unscaling
GARCH_MU    = 5.8e-5
GARCH_OMEGA = 2.32e-4
GARCH_ALPHA = 0.10
GARCH_BETA  = 0.40

# Near-ATM option half-spreads in price pts (from microstructure analysis)
ATM_STRIKES    = [5000, 5100, 5200, 5300, 5400, 5500]
OPT_HALFSPREAD = {5000: 3.022, 5100: 2.148, 5200: 1.444,
                  5300: 1.053, 5400: 0.691, 5500: 0.575}
# noise floor vs residual-std ratio (from IV smile section)
NOISE_SIG_RATIO = {5000: 17.8, 5100: 2.9, 5200: 2.6,
                   5300: 1.3,  5400: 0.95, 5500: 3.1}
UND_HALFSPREAD = 0.5       # underlying half-spread (1 price tick)
HEDGE_FREQS    = [1, 5, 10, 50, 100]

# ── Pure helpers ──────────────────────────────────────────────────────────────

def data_tte(global_ts):
    return TTE0 - (global_ts // TPS) / TPD


def smile_iv(S, K, T):
    m = np.log(K / S) / np.sqrt(np.clip(T, 1e-6, None))
    return A + m * (B + C_SMILE * m)


def bs_call(S, K, T, iv):
    sT = iv * np.sqrt(T)
    d1 = (np.log(S / K) + 0.5 * iv**2 * T) / sT
    return S * norm.cdf(d1) - K * norm.cdf(d1 - sT)


def bs_call_iv(S, K, T, price, iters=60, tol=1e-9):
    """Newton IV inversion — vectorised."""
    intrinsic = np.maximum(S - K, 0.0)
    valid = (price > intrinsic + 1e-9) & (price < S) & (T > 0)
    sigma = np.full_like(price, 0.02, dtype=float)
    for _ in range(iters):
        sT   = sigma * np.sqrt(T)
        d1   = (np.log(S / K) + 0.5 * sigma**2 * T) / sT
        p    = S * norm.cdf(d1) - K * norm.cdf(d1 - sT)
        veg  = S * norm.pdf(d1) * np.sqrt(T)
        diff = p - price
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma -= np.where(veg > 1e-12, diff / veg, 0.0)
        sigma = np.clip(sigma, 1e-6, 5.0)
        if valid.any() and np.nanmax(np.abs(diff[valid])) < tol:
            break
    sigma[~valid] = np.nan
    return sigma


def bs_greeks(S, K, T, iv):
    """Delta, Gamma, Theta (per day, negative for long call), Vega."""
    sT    = iv * np.sqrt(T)
    d1    = (np.log(S / K) + 0.5 * iv**2 * T) / sT
    nd1   = norm.pdf(d1)
    delta = norm.cdf(d1)
    gamma = nd1 / (S * iv * np.sqrt(T))
    theta = -S * nd1 * iv / (2.0 * np.sqrt(T))   # per day, < 0
    vega  = S * nd1 * np.sqrt(T)
    return delta, gamma, theta, vega


def garch_vol_series(log_ret):
    """GARCH(1,1)-t on 100×-scaled returns → σ in per-√day units."""
    r      = log_ret * 100.0
    sigma2 = np.full(len(r), GARCH_OMEGA / (1 - GARCH_ALPHA - GARCH_BETA))
    for i in range(1, len(r)):
        eps2      = (r[i - 1] - GARCH_MU) ** 2
        sigma2[i] = GARCH_OMEGA + GARCH_ALPHA * eps2 + GARCH_BETA * sigma2[i - 1]
    # σ_scaled / 100 * √TPD = √sigma2 / 100 * 100 = √sigma2  (per-√day)
    return np.sqrt(sigma2)


def rolling_rv(log_ret, window):
    """Rolling realized vol in per-√day units."""
    r2  = log_ret ** 2
    out = np.full(len(r2), np.nan)
    for i in range(window, len(r2)):
        out[i] = np.sqrt(np.mean(r2[i - window : i]) * TPD)
    return out


# ── Analysis sections ─────────────────────────────────────────────────────────

def load_data():
    opt = pd.read_csv(DATA / "clean_options_chain.csv")
    und = pd.read_csv(DATA / "clean_VELVETFRUIT_EXTRACT.csv").rename(
        columns={"mid_price": "S"})
    df  = opt.merge(und, on="global_ts").sort_values("global_ts").reset_index(drop=True)
    gts = df["global_ts"].to_numpy()
    S   = df["S"].to_numpy()
    T   = data_tte(gts)
    print(f"[load] rows={len(df)}, TTE [{T.min():.3f}, {T.max():.3f}]")
    return df, gts, S, T


def compute_greeks(df, S, T):
    IV, DELTA, GAMMA, THETA, VEGA = {}, {}, {}, {}, {}
    for K in ATM_STRIKES:
        prices = df[f"VEV_{K}"].to_numpy()
        iv_k   = bs_call_iv(S, K, T, prices)
        iv_use = np.where(np.isfinite(iv_k), iv_k, smile_iv(S, K, T))
        IV[K]    = iv_k
        DELTA[K], GAMMA[K], THETA[K], VEGA[K] = bs_greeks(S, K, T, iv_use)
    return IV, DELTA, GAMMA, THETA, VEGA


def compute_vol_estimates(S):
    log_ret        = np.diff(np.log(S), prepend=np.nan)
    sigma_garch    = np.concatenate([[np.nan], garch_vol_series(log_ret[1:])])
    rv_wins        = {w: rolling_rv(log_ret, w) for w in [10, 50, 200]}
    return log_ret, sigma_garch, rv_wins


def sec1_iv_summary(IV):
    print("\n=== §1 IV per strike ===")
    print(f"  {'K':>6}  {'mean IV':>9}  {'std IV':>9}  {'valid%':>8}")
    for K in ATM_STRIKES:
        iv = IV[K]
        print(f"  {K:>6}  {np.nanmean(iv):>9.5f}  {np.nanstd(iv):>9.5f}  "
              f"{np.isfinite(iv).mean():>8.1%}")


def sec2_greeks_summary(GAMMA, THETA, VEGA):
    print("\n=== §2 BS Greeks ===")
    print(f"  {'K':>6}  {'Γ_mean':>10}  {'Θ/day_mean':>12}  {'Vega_mean':>11}")
    for K in ATM_STRIKES:
        print(f"  {K:>6}  {np.nanmean(GAMMA[K]):>10.6f}  "
              f"{np.nanmean(THETA[K]):>12.4f}  {np.nanmean(VEGA[K]):>11.1f}")


def sec3_vol_summary(sigma_garch, rv_wins):
    print("\n=== §3 Realized vol estimates ===")
    print(f"  {'estimator':>14}  {'mean':>9}  {'std':>9}")
    print(f"  {'GARCH(1,1)-t':>14}  {np.nanmean(sigma_garch):>9.5f}  "
          f"{np.nanstd(sigma_garch):>9.5f}")
    for w, rv in rv_wins.items():
        print(f"  {'RV(w='+str(w)+')':>14}  {np.nanmean(rv):>9.5f}  "
              f"{np.nanstd(rv):>9.5f}")


def sec4_core_condition(IV, sigma_garch):
    print("\n=== §4 σ_rv vs σ_iv — core condition ===")
    print(f"  {'K':>6}  {'mean IV':>9}  {'GARCH σ':>9}  "
          f"{'spread':>8}  {'%GARCH>IV':>11}  {'rv/iv':>7}")
    for K in ATM_STRIKES:
        iv_k   = IV[K]
        valid  = np.isfinite(iv_k) & np.isfinite(sigma_garch)
        mi, mr = np.nanmean(iv_k), np.nanmean(sigma_garch)
        pct    = (sigma_garch[valid] > iv_k[valid]).mean()
        print(f"  {K:>6}  {mi:>9.5f}  {mr:>9.5f}  "
              f"{mr-mi:>8.5f}  {pct:>11.1%}  {mr/mi:>7.3f}x")


def sec5_harvest_vs_theta(IV, GAMMA, THETA, S, sigma_garch):
    """Expected gamma harvest vs theta bleed per tick."""
    print("\n=== §5 Gamma harvest vs theta bleed (expected) ===")
    print(f"  {'K':>6}  {'harvest/tick':>14}  {'theta/tick':>12}  "
          f"{'net/tick':>12}  {'H/T ratio':>11}")
    dt = 1.0 / TPD
    results = {}
    for K in ATM_STRIKES:
        iv_k     = np.where(np.isfinite(IV[K]), IV[K], smile_iv(S, K, data_tte(np.zeros(1))))
        rv2      = sigma_garch ** 2 * dt          # σ²_rv per tick
        iv2      = np.where(np.isfinite(IV[K]), IV[K], smile_iv(S, K, np.full(len(S), 0.5))) ** 2 * dt
        harvest  = 0.5 * GAMMA[K] * S**2 * rv2    # expected
        theta_t  = -THETA[K] * dt                   # positive bleed per tick
        valid    = np.isfinite(harvest) & np.isfinite(theta_t)
        eh, et   = np.nanmean(harvest[valid]), np.nanmean(theta_t[valid])
        results[K] = (eh - et, eh / et if et > 0 else np.nan)
        print(f"  {K:>6}  {eh:>14.7f}  {et:>12.7f}  "
              f"{eh-et:>12.7f}  {eh/et:>11.3f}x")
    return results


def sec6_realized_pnl(IV, GAMMA, THETA, S, gts):
    """Realized gamma PnL per tick using actual ΔS."""
    print("\n=== §6 Realized gamma PnL (continuous, no txn cost) ===")
    print(f"  {'K':>6}  {'cum PnL':>10}  {'mean/tick':>11}  {'sharpe':>10}  {'%pos':>8}")
    dt  = 1.0 / TPD
    dS  = np.diff(S, prepend=np.nan)
    pnl = {}
    for K in ATM_STRIKES:
        g_pnl    = 0.5 * GAMMA[K] * dS**2         # ½Γ(ΔS)²
        t_bleed  = THETA[K] * dt                    # < 0 per tick
        net      = g_pnl + t_bleed
        valid    = np.isfinite(net)
        cum      = np.nansum(net[valid])
        mn, std  = np.nanmean(net[valid]), np.nanstd(net[valid])
        sharpe   = mn / std if std > 0 else 0.0
        pct_pos  = (net[valid] > 0).mean()
        pnl[K]   = net
        print(f"  {K:>6}  {cum:>10.2f}  {mn:>11.7f}  {sharpe:>10.5f}  {pct_pos:>8.1%}")
    return pnl


def sec7_discrete_hedge(df, DELTA, S, gts):
    """Discrete delta-hedge simulation at multiple frequencies."""
    print("\n=== §7 Discrete hedge simulation ===")
    print(f"  PnL = ΔC − Δ_old·ΔS − |ΔΔ|·UND_HALFSPREAD per interval")
    results = {K: {} for K in ATM_STRIKES}
    for K in ATM_STRIKES:
        C_prices = df[f"VEV_{K}"].to_numpy()
        for h in HEDGE_FREQS:
            pnl_list = []
            i = h
            while i < len(S):
                i0  = i - h
                dC  = C_prices[i] - C_prices[i0]
                dS_h = S[i] - S[i0]
                d_old   = DELTA[K][i0]
                d_new   = DELTA[K][i]
                raw_pnl = dC - d_old * dS_h
                txn     = abs(d_new - d_old) * UND_HALFSPREAD
                if np.isfinite(raw_pnl) and np.isfinite(txn):
                    pnl_list.append(raw_pnl - txn)
                i += h
            arr = np.array(pnl_list)
            mn, st = arr.mean(), arr.std()
            results[K][h] = dict(cum=arr.sum(), mean=mn, std=st,
                                  sharpe=mn / st if st > 0 else 0.0)

    print(f"\n  {'K':>6}  {'h':>5}  {'cum PnL':>10}  {'mean':>9}  {'sharpe':>9}")
    for K in ATM_STRIKES:
        for h in HEDGE_FREQS:
            r = results[K][h]
            print(f"  {K:>6}  {h:>5}  {r['cum']:>10.2f}  "
                  f"{r['mean']:>9.4f}  {r['sharpe']:>9.5f}")
    return results


def sec8_txn_breakeven(IV, GAMMA, S, sigma_garch):
    """Compute breakeven underlying half-spread for gamma scalping."""
    print("\n=== §8 Transaction cost breakeven ===")
    print(f"  Breakeven UND half-spread = S·(σ²_rv−σ²_iv)/(2·σ_rv·√TPD)")
    print(f"  {'K':>6}  {'harvest/tick':>14}  {'txn/tick':>12}  "
          f"{'net/tick':>12}  {'BE spread':>11}")
    dt     = 1.0 / TPD
    rv_m   = np.nanmean(sigma_garch)
    S_m    = np.nanmean(S)
    e_abs_dS = S_m * rv_m / np.sqrt(TPD)   # E[|ΔS|] per tick ≈ S·σ/√TPD
    for K in ATM_STRIKES:
        Gm   = np.nanmean(GAMMA[K])
        iv_m = np.nanmean(IV[K]) if np.isfinite(np.nanmean(IV[K])) else A
        h_pt = 0.5 * Gm * S_m**2 * (rv_m**2 - iv_m**2) * dt
        txn  = Gm * e_abs_dS * UND_HALFSPREAD
        net  = h_pt - txn
        be   = h_pt / (Gm * e_abs_dS) if Gm * e_abs_dS > 0 else np.nan
        print(f"  {K:>6}  {h_pt:>14.7f}  {txn:>12.7f}  {net:>12.7f}  {be:>11.4f}")


def sec9_vega_risk(IV, GAMMA, VEGA, S, sigma_garch):
    """Vega risk per tick vs expected gamma harvest."""
    print("\n=== §9 Vega risk ===")
    print(f"  {'K':>6}  {'Vega_mean':>11}  {'Δσ_iv std/tick':>16}  "
          f"{'vega risk/tick':>16}  {'vs harvest':>12}")
    dt   = 1.0 / TPD
    rv_m = np.nanmean(sigma_garch)
    S_m  = np.nanmean(S)
    for K in ATM_STRIKES:
        iv_k    = IV[K]
        d_iv    = np.diff(iv_k, prepend=np.nan)
        v_m     = np.nanmean(VEGA[K])
        d_iv_s  = np.nanstd(d_iv)
        vr      = v_m * d_iv_s
        Gm      = np.nanmean(GAMMA[K])
        iv_m    = np.nanmean(iv_k) if np.isfinite(np.nanmean(iv_k)) else A
        harvest = 0.5 * Gm * S_m**2 * (rv_m**2 - iv_m**2) * dt
        ratio   = vr / abs(harvest) if harvest != 0 else np.nan
        print(f"  {K:>6}  {v_m:>11.1f}  {d_iv_s:>16.7f}  {vr:>16.4f}  {ratio:>12.2f}x")


def sec10_ranking(gamma_edge, IV, sigma_garch):
    """Rank strikes for gamma scalping by composite score."""
    print("\n=== §10 Strike ranking ===")
    print(f"  {'K':>6}  {'net edge':>10}  {'H/T':>7}  {'noise/sig':>11}  "
          f"{'score':>9}  {'verdict':>22}")
    scores = {}
    for K in ATM_STRIKES:
        net, ht = gamma_edge[K]
        ns      = NOISE_SIG_RATIO[K]
        # Higher score = higher H/T, lower noise-to-signal
        score   = ht / (1 + ns) if np.isfinite(ns) else ht
        scores[K] = score
        verdict = ("STRONG" if ht > 1.5 and ns < 3 else
                   "OK"     if ht > 1.0 and ns < 5 else
                   "NOISY"  if ns > 5   else "WEAK")
        print(f"  {K:>6}  {net:>10.6f}  {ht:>7.3f}  {ns:>11.2f}  "
              f"{score:>9.4f}  {verdict:>22}")
    best = max(scores, key=scores.get)
    print(f"\n  Best gamma-scalping strike: K={best}")
    return scores


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_rv_vs_iv(gts, sigma_garch, rv_wins, IV):
    fig, axes = plt.subplots(len(ATM_STRIKES), 1, figsize=(16, 18), sharex=True)
    fig.suptitle("σ_rv vs σ_iv per near-ATM strike", fontsize=12)
    col_map = {10: "orange", 50: "green", 200: "royalblue"}
    for ax, K in zip(axes, ATM_STRIKES):
        ax.plot(gts, sigma_garch, lw=0.7, color="crimson",
                label="GARCH(1,1)-t", alpha=0.85)
        for w, rv in rv_wins.items():
            ax.plot(gts, rv, lw=0.5, color=col_map[w], label=f"RV(w={w})", alpha=0.7)
        ax.plot(gts, IV[K], lw=0.9, color="black", label=f"IV K={K}")
        ax.fill_between(gts, IV[K], sigma_garch,
                        where=np.nan_to_num(sigma_garch) > np.nan_to_num(IV[K]),
                        alpha=0.12, color="green", label="σ_rv > σ_iv")
        ax.axhline(0, color="k", lw=0.4)
        for b in [1_000_000, 2_000_000]:
            ax.axvline(b, color="gray", lw=0.5, ls=":")
        ax.set_ylabel(f"K={K}", fontsize=8)
        ax.legend(fontsize=5, ncol=6, loc="upper right")
        ax.grid(True, alpha=0.2)
    axes[-1].set_xlabel("global_ts")
    fig.tight_layout()
    fig.savefig(str(IMG / "gamma_rv_vs_iv.png"), dpi=140)
    plt.close(fig)
    print("  → images/gamma_rv_vs_iv.png")


def plot_cumulative_pnl(gts, realized_pnl):
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = plt.cm.tab10(np.linspace(0, 0.7, len(ATM_STRIKES)))
    for K, col in zip(ATM_STRIKES, colors):
        pnl = realized_pnl[K]
        cum = np.nancumsum(np.where(np.isfinite(pnl), pnl, 0.0))
        ax.plot(gts, cum, lw=1.2, color=col, label=f"K={K}")
    ax.axhline(0, color="k", lw=0.6)
    for b in [1_000_000, 2_000_000]:
        ax.axvline(b, color="gray", lw=0.5, ls=":")
    ax.set_xlabel("global_ts")
    ax.set_ylabel("Cumulative PnL (pts)")
    ax.set_title("Realized gamma PnL — continuous hedging, no transaction cost")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(str(IMG / "gamma_pnl_cumulative.png"), dpi=140)
    plt.close(fig)
    print("  → images/gamma_pnl_cumulative.png")


def plot_discrete_hedge(discrete_results):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Discrete hedge cumulative PnL by frequency (incl. txn cost)", fontsize=12)
    for ax, K in zip(axes.flat[:len(ATM_STRIKES)], ATM_STRIKES):
        cum_vals = [discrete_results[K][h]["cum"] for h in HEDGE_FREQS]
        bars = ax.bar(range(len(HEDGE_FREQS)), cum_vals, color="steelblue",
                      edgecolor="white")
        ax.set_xticks(range(len(HEDGE_FREQS)))
        ax.set_xticklabels([str(h) for h in HEDGE_FREQS])
        ax.set_xlabel("Hedge freq (ticks)")
        ax.set_ylabel("Cum PnL (pts)")
        ax.set_title(f"K={K}")
        ax.axhline(0, color="k", lw=0.6)
        ax.grid(True, alpha=0.2)
        for bar, v in zip(bars, cum_vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (max(cum_vals) - min(cum_vals)) * 0.02,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    for ax in axes.flat[len(ATM_STRIKES):]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(str(IMG / "gamma_discrete_hedge.png"), dpi=140)
    plt.close(fig)
    print("  → images/gamma_discrete_hedge.png")


def plot_strike_ranking(scores, gamma_edge):
    ks  = ATM_STRIKES
    s_v = [scores[k] for k in ks]
    n_v = [gamma_edge[k][0] for k in ks]
    h_v = [gamma_edge[k][1] for k in ks]
    ns_v = [NOISE_SIG_RATIO[k] for k in ks]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, vals, title, color, hl in [
        (axes[0], s_v, "Composite score",      "steelblue",  None),
        (axes[1], n_v, "Net edge / tick",       "darkorange", 0),
        (axes[2], h_v, "H/T ratio",             "green",      1),
        (axes[3], ns_v,"Noise/signal ratio",    "crimson",    None),
    ]:
        ax.bar(range(len(ks)), vals, color=color, edgecolor="white")
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels([str(k) for k in ks], fontsize=8)
        ax.set_xlabel("Strike")
        ax.set_title(title)
        if hl is not None:
            ax.axhline(hl, color="k", lw=0.8, ls="--")
        ax.grid(True, alpha=0.2)
    fig.suptitle("Strike ranking for gamma scalping", fontsize=12)
    fig.tight_layout()
    fig.savefig(str(IMG / "gamma_strike_ranking.png"), dpi=140)
    plt.close(fig)
    print("  → images/gamma_strike_ranking.png")


def plot_pnl_distribution(realized_pnl):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle("Per-tick gamma PnL distribution (continuous hedging)", fontsize=12)
    for ax, K in zip(axes.flat, ATM_STRIKES):
        pnl  = realized_pnl[K]
        data = pnl[np.isfinite(pnl)]
        bins = 80
        ax.hist(data, bins=bins, color="steelblue", edgecolor="none", alpha=0.8)
        ax.axvline(np.mean(data), color="crimson", lw=1.4, ls="--",
                   label=f"mean={np.mean(data):.2e}")
        ax.axvline(0, color="k", lw=0.7)
        ax.set_title(f"K={K}")
        ax.set_xlabel("PnL / tick")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)
    for ax in axes.flat[len(ATM_STRIKES):]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(str(IMG / "gamma_pnl_distribution.png"), dpi=140)
    plt.close(fig)
    print("  → images/gamma_pnl_distribution.png")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  GAMMA SCALPING VIABILITY ANALYSIS — VEV OPTIONS (Round 3)")
    print("=" * 65)

    df, gts, S, T = load_data()
    IV, DELTA, GAMMA, THETA, VEGA = compute_greeks(df, S, T)
    log_ret, sigma_garch, rv_wins  = compute_vol_estimates(S)

    sec1_iv_summary(IV)
    sec2_greeks_summary(GAMMA, THETA, VEGA)
    sec3_vol_summary(sigma_garch, rv_wins)
    sec4_core_condition(IV, sigma_garch)
    gamma_edge    = sec5_harvest_vs_theta(IV, GAMMA, THETA, S, sigma_garch)
    realized_pnl  = sec6_realized_pnl(IV, GAMMA, THETA, S, gts)
    discrete_res  = sec7_discrete_hedge(df, DELTA, S, gts)
    sec8_txn_breakeven(IV, GAMMA, S, sigma_garch)
    sec9_vega_risk(IV, GAMMA, VEGA, S, sigma_garch)
    scores        = sec10_ranking(gamma_edge, IV, sigma_garch)

    print("\n=== Generating plots ===")
    plot_rv_vs_iv(gts, sigma_garch, rv_wins, IV)
    plot_cumulative_pnl(gts, realized_pnl)
    plot_discrete_hedge(discrete_res)
    plot_strike_ranking(scores, gamma_edge)
    plot_pnl_distribution(realized_pnl)

    print("\n" + "=" * 65)
    print("  DONE. All outputs in images/")
    print("=" * 65)


if __name__ == "__main__":
    main()
