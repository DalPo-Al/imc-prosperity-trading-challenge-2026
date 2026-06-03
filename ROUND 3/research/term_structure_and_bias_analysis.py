"""
Priority 1 + 2 analyses for OptionTrader launch:

1. Term-structure check: regress per-tick smile coefficients (a, b, c) on TTE.
   Project a, b, c to live-trading TTE range (5 -> 2).
   Determine whether the static A=0.012535 will systematically misprice options
   during days 3-5.

2. Per-strike bias: compute mean(market - BS(static smile)) per strike on
   historical data. Use as additive correction in surface_arbitrage_orders().

All plots saved to images/.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(ROOT, "..", "..", "data", "round3")
IMG_DIR   = os.path.join(ROOT, "images")
OPT_CSV   = os.path.join(DATA_DIR, "clean_options_chain.csv")
UND_CSV   = os.path.join(DATA_DIR, "clean_VELVETFRUIT_EXTRACT.csv")
COEF_CSV  = os.path.join(ROOT, "iv_smile_coeffs.csv")    # per-ts a,b,c

os.makedirs(IMG_DIR, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
A_OLD, B_OLD, C_OLD = 0.012535, 0.002252, 0.56799
TTE_AT_DATA_ROW0    = 8
TICKS_PER_DAY       = 10_000
TICKS_PER_TS        = 100

ATM_STRIKES   = [5000, 5100, 5200, 5300, 5400, 5500]
ALL_STRIKES   = [4000, 4500] + ATM_STRIKES + [6000, 6500]
LIVE_TTE_GRID = [5.0, 4.0, 3.0, 2.0]


# ── Helpers ───────────────────────────────────────────────────────────────────
def data_tte(global_ts):
    return TTE_AT_DATA_ROW0 - (global_ts // TICKS_PER_TS) / TICKS_PER_DAY


def smile_iv(S, K, T, a, b, c):
    m = np.log(K / S) / np.sqrt(T)
    return a + m * (b + c * m)


def bs_call(S, K, T, iv):
    sT = iv * np.sqrt(T)
    d1 = (np.log(S / K) + 0.5 * iv ** 2 * T) / sT
    d2 = d1 - sT
    return S * norm.cdf(d1) - K * norm.cdf(d2)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading …")
    coef = pd.read_csv(COEF_CSV)
    opt  = pd.read_csv(OPT_CSV)
    und  = pd.read_csv(UND_CSV).rename(columns={"mid_price": "S"})
    df   = opt.merge(und, on="global_ts", how="inner").sort_values("global_ts").reset_index(drop=True)

    # Align coef to df by global_ts
    coef = coef.merge(df[["global_ts"]], on="global_ts", how="inner")

    gts = coef["global_ts"].to_numpy()
    T   = data_tte(gts)
    print(f"  N coef rows = {len(coef)}    TTE range = [{T.min():.4f}, {T.max():.4f}]")

    # Filter out invalid coefficients (NaN or extreme outliers)
    valid = (
        np.isfinite(coef["a"]) & np.isfinite(coef["b"]) & np.isfinite(coef["c"])
        & (np.abs(coef["a"] - coef["a"].median()) < 10 * coef["a"].mad() if hasattr(coef["a"], "mad") else True)
    )
    a = coef["a"].to_numpy()
    b = coef["b"].to_numpy()
    c = coef["c"].to_numpy()
    fin = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
    print(f"  Finite coef rows: {fin.sum()} / {len(coef)}")

    # Trim 1% tails on each coefficient to suppress fitting outliers
    def trim_mask(x, q=0.005):
        lo, hi = np.nanquantile(x, q), np.nanquantile(x, 1 - q)
        return (x >= lo) & (x <= hi)

    keep = fin & trim_mask(a) & trim_mask(b) & trim_mask(c)
    print(f"  After 0.5% tail trim: {keep.sum()} rows")

    Tk, ak, bk, ck = T[keep], a[keep], b[keep], c[keep]

    # ═══════════════════════════════════════════════════════════════════════════
    # PRIORITY 1: TERM-STRUCTURE OF SMILE COEFFICIENTS
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 72)
    print("PRIORITY 1 — Term structure of smile coefficients")
    print("═" * 72)

    def linfit(x, y):
        x = np.asarray(x); y = np.asarray(y)
        X = np.column_stack([np.ones_like(x), x])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        yhat = X @ beta
        ss_res = ((y - yhat) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        # Std error of slope (OLS)
        n = len(y)
        s_err = np.sqrt(ss_res / max(n - 2, 1))
        x_var = ((x - x.mean()) ** 2).sum()
        se_slope = s_err / np.sqrt(x_var) if x_var > 0 else np.nan
        t_stat = beta[1] / se_slope if se_slope > 0 else np.nan
        return beta[0], beta[1], se_slope, t_stat, r2

    print(f"\n{'coeff':<6}{'intercept':>14}{'slope/T':>14}{'SE slope':>14}{'t-stat':>10}{'R²':>10}")
    fits = {}
    for name, y in [("a", ak), ("b", bk), ("c", ck)]:
        intercept, slope, se, t, r2 = linfit(Tk, y)
        fits[name] = (intercept, slope, se, t, r2)
        sig = "  ***" if abs(t) > 3 else ("  *" if abs(t) > 2 else "")
        print(f"{name:<6}{intercept:>14.6f}{slope:>14.6e}{se:>14.6e}{t:>10.2f}{r2:>10.4f}{sig}")

    print(f"\nProjection to live-trading TTE grid [5, 4, 3, 2]:")
    print(f"{'TTE':>6}{'a_proj':>12}{'b_proj':>12}{'c_proj':>12}{'a / A_OLD':>12}")
    proj_table = {}
    for T_live in LIVE_TTE_GRID:
        a_p = fits["a"][0] + fits["a"][1] * T_live
        b_p = fits["b"][0] + fits["b"][1] * T_live
        c_p = fits["c"][0] + fits["c"][1] * T_live
        proj_table[T_live] = (a_p, b_p, c_p)
        print(f"{T_live:>6.1f}{a_p:>12.6f}{b_p:>12.6f}{c_p:>12.6f}{a_p / A_OLD:>12.4f}")

    # Average a over live-trading window
    T_live_mean = np.mean(LIVE_TTE_GRID)
    A_LIVE = fits["a"][0] + fits["a"][1] * T_live_mean
    B_LIVE = fits["b"][0] + fits["b"][1] * T_live_mean
    C_LIVE = fits["c"][0] + fits["c"][1] * T_live_mean
    print(f"\nLive-trading mean (TTE = {T_live_mean:.1f}):")
    print(f"  A_LIVE = {A_LIVE:.6f}   (vs A_OLD = {A_OLD},  ratio {A_LIVE / A_OLD:.4f})")
    print(f"  B_LIVE = {B_LIVE:.6f}")
    print(f"  C_LIVE = {C_LIVE:.6f}")

    # PLOT 1: a, b, c vs TTE with regression
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    fig.suptitle("Smile coefficients vs TTE — historical (TTE 5→8) and projected (TTE 2→5)",
                 fontsize=11)
    coef_data = [("a (level)", ak, fits["a"], "steelblue"),
                 ("b (skew)",  bk, fits["b"], "darkgreen"),
                 ("c (curvature)", ck, fits["c"], "darkorange")]
    T_grid_full = np.linspace(2, 8, 200)
    for ax, (lbl, y, fit, col) in zip(axes, coef_data):
        ax.scatter(Tk, y, s=1.5, alpha=0.15, color=col, label="per-tick fit")
        intercept, slope = fit[0], fit[1]
        ax.plot(T_grid_full, intercept + slope * T_grid_full,
                lw=2, color="crimson",
                label=f"OLS  slope={slope:.3e}  t={fit[3]:.2f}  R²={fit[4]:.4f}")
        ax.axvspan(2, 5, alpha=0.10, color="orange", label="live-trade TTE")
        ax.axvspan(5, 8, alpha=0.10, color="steelblue", label="historical TTE")
        if lbl.startswith("a"):
            ax.axhline(A_OLD, color="black", lw=1, ls=":",
                       label=f"hardcoded A={A_OLD}")
        ax.set_ylabel(lbl, fontsize=10)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("TTE (days)")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "term_structure_abc.png"), dpi=140)
    plt.close(fig)
    print(f"\n  → images/term_structure_abc.png")

    # ═══════════════════════════════════════════════════════════════════════════
    # PRIORITY 2: PER-STRIKE BIAS UNDER STATIC A,B,C
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 72)
    print("PRIORITY 2 — Per-strike bias of static smile (OLD A,B,C)")
    print("═" * 72)

    S = df["S"].to_numpy()
    Td = data_tte(df["global_ts"].to_numpy())

    bias_static = {}
    bias_live   = {}
    rows = []
    for K in ALL_STRIKES:
        market = df[f"VEV_{K}"].to_numpy()
        # Static OLD
        iv_old = smile_iv(S, K, Td, A_OLD, B_OLD, C_OLD)
        theo_old = bs_call(S, K, Td, iv_old)
        d_old = market - theo_old
        # Term-structure-projected (LIVE)
        iv_live = smile_iv(S, K, Td, A_LIVE, B_LIVE, C_LIVE)
        theo_live = bs_call(S, K, Td, iv_live)
        d_live = market - theo_live
        bias_static[K] = float(np.mean(d_old))
        bias_live[K]   = float(np.mean(d_live))
        rows.append({
            "K": K,
            "bias_OLD":  float(np.mean(d_old)),
            "std_OLD":   float(np.std(d_old)),
            "bias_LIVE": float(np.mean(d_live)),
            "std_LIVE":  float(np.std(d_live)),
            "se_mean_OLD": float(np.std(d_old) / np.sqrt(len(d_old))),
        })
    bias_df = pd.DataFrame(rows)
    print("\n" + bias_df.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    # PLOT 2: per-strike bias bars
    fig, ax = plt.subplots(figsize=(11, 5))
    xs = np.arange(len(ALL_STRIKES))
    w  = 0.4
    bias_o = [bias_static[K] for K in ALL_STRIKES]
    bias_l = [bias_live[K]   for K in ALL_STRIKES]
    se_o   = [r["se_mean_OLD"] for r in rows]
    ax.bar(xs - w/2, bias_o, w, yerr=se_o, color="steelblue", edgecolor="white",
           label=f"OLD A={A_OLD}", capsize=3)
    ax.bar(xs + w/2, bias_l, w, color="darkorange", edgecolor="white",
           label=f"LIVE A={A_LIVE:.5f} (term-struct projected)")
    ax.axhline(0, color="k", lw=0.6)
    ax.axhline(+1, color="red", lw=0.6, ls="--", alpha=0.5, label="±1 pt threshold")
    ax.axhline(-1, color="red", lw=0.6, ls="--", alpha=0.5)
    ax.set_xticks(xs); ax.set_xticklabels([str(K) for K in ALL_STRIKES])
    ax.set_xlabel("Strike"); ax.set_ylabel("Mean (market − BS(smile))  [pts]")
    ax.set_title("Per-strike bias of static smile — OLD vs term-structure-projected")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "per_strike_bias.png"), dpi=140)
    plt.close(fig)
    print(f"\n  → images/per_strike_bias.png")

    # ═══════════════════════════════════════════════════════════════════════════
    # OUTPUT: drop-in dict for OptionTrader.py
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 72)
    print("DROP-IN VALUES for OptionTrader.py")
    print("═" * 72)
    print(f"\n# Term-structure-projected smile (live TTE 2-5):")
    print(f"A = {A_LIVE:.6f}")
    print(f"B = {B_LIVE:.6f}")
    print(f"C = {C_LIVE:.6f}")
    print(f"\n# Per-strike additive bias (OLD smile):")
    print("STRIKE_BIAS_STATIC = {")
    for K in ALL_STRIKES:
        print(f"    {K}: {bias_static[K]:+.4f},")
    print("}")
    print(f"\n# Per-strike bias (LIVE smile):")
    print("STRIKE_BIAS_LIVE = {")
    for K in ALL_STRIKES:
        print(f"    {K}: {bias_live[K]:+.4f},")
    print("}")

    # Save findings JSON for downstream use
    out = {
        "term_structure": {
            "a": {"intercept": fits["a"][0], "slope": fits["a"][1],
                  "se_slope": fits["a"][2], "t_stat": fits["a"][3], "r2": fits["a"][4]},
            "b": {"intercept": fits["b"][0], "slope": fits["b"][1],
                  "se_slope": fits["b"][2], "t_stat": fits["b"][3], "r2": fits["b"][4]},
            "c": {"intercept": fits["c"][0], "slope": fits["c"][1],
                  "se_slope": fits["c"][2], "t_stat": fits["c"][3], "r2": fits["c"][4]},
        },
        "live_projection": {"A": A_LIVE, "B": B_LIVE, "C": C_LIVE,
                            "T_mean": T_live_mean,
                            "by_TTE": {str(k): v for k, v in proj_table.items()}},
        "strike_bias_static": bias_static,
        "strike_bias_live":   bias_live,
    }
    with open(os.path.join(ROOT, "term_structure_findings.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  → term_structure_findings.json")
    print("\nDone.")


if __name__ == "__main__":
    main()
