"""
iv_deep_analysis.py — 13-section deep-dive on VEV smile dynamics.

Requires:
    iv_smile_coeffs.csv          (output of iv_smile_analysis.py)
    data/round3/clean_options_chain.csv
    data/round3/clean_VELVETFRUIT_EXTRACT.csv
    data/round3/prices_round_3_day_{0,1,2}.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox

# ── config ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA = ROOT.parent / "data" / "round3"
COEFFS_FILE = ROOT / "iv_smile_coeffs.csv"

TTE_DAY0_DAYS = 8
DAYS_PER_YEAR = 1
ROWS_PER_DAY = 10000
STRIKE_WHITELIST = {5000, 5100, 5200, 5300, 5400, 5500}
RV_WINDOW = 100  # ts for realized-vol rolling window
EMA_WINDOW = 100  # matches current strategy THEO_NORM_WINDOW
LOW_VEGA_THR = 1.0  # matches strategy LOW_VEGA_THR_ADJ trigger

# ── load ──────────────────────────────────────────────────────────────────────
opt = pd.read_csv(DATA / "clean_options_chain.csv")
und = pd.read_csv(DATA / "clean_VELVETFRUIT_EXTRACT.csv").rename(
    columns={"mid_price": "S"}
)
df = (
    opt.merge(und, on="global_ts", how="inner")
    .sort_values("global_ts")
    .reset_index(drop=True)
)
coef_df = pd.read_csv(COEFFS_FILE)

strike_cols_all = [c for c in opt.columns if c.startswith("VEV_")]
strike_cols = [c for c in strike_cols_all if int(c.split("_")[1]) in STRIKE_WHITELIST]
strikes = np.array([int(c.split("_")[1]) for c in strike_cols], dtype=float)
K_arr = strikes[None, :]

N = len(df)
S_arr = df["S"].to_numpy()[:, None]
C_arr = df[strike_cols].to_numpy().astype(float)
S_flat = df["S"].to_numpy()

day_idx = (np.arange(N) // ROWS_PER_DAY).clip(max=2)
tte_days = TTE_DAY0_DAYS - day_idx
T_arr = (tte_days)[:, None]
sqrtT = np.sqrt(T_arr)

coeffs = coef_df[["a", "b", "c"]].to_numpy()  # (N, 3)
a_flat = coeffs[:, 0]


# ── helpers ───────────────────────────────────────────────────────────────────
def bs_call_iv(S, K, T, C, sigma0=0.01, iters=100, tol=1e-8):
    intrinsic = np.maximum(S - K, 0.0)
    valid = (C > intrinsic + 1e-9) & (C < S) & (T > 0)
    sigma = np.full_like(C, sigma0, dtype=float)
    for _ in range(iters):
        sT = sigma * np.sqrt(T)
        d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / sT
        price = S * norm.cdf(d1) - K * norm.cdf(d1 - sT)
        vega = S * norm.pdf(d1) * np.sqrt(T)
        diff = price - C
        with np.errstate(divide="ignore", invalid="ignore"):
            step = np.where(vega > 1e-12, diff / vega, 0.0)
        sigma = np.clip(sigma - step, 1e-6, 5.0)
        if np.nanmax(np.abs(diff[valid])) < tol:
            break
    sigma[~valid] = np.nan
    return sigma


def ar1_ols(x):
    x = x[np.isfinite(x)]
    y = x[1:]
    Xr = sm.add_constant(x[:-1])
    r = sm.OLS(y, Xr).fit()
    return r.params[1], r.bse[1], r.pvalues[1]


def section(n, title):
    bar = "=" * 60
    print(f"\n{bar}\nSECTION {n}: {title}\n{bar}")


# ── compute IV matrix (same logic as first script) ────────────────────────────
print("Computing IV matrix …")
IV = bs_call_iv(S_arr, K_arr, T_arr, C_arr)  # (N, Ks)
M = np.log(K_arr / S_arr) / sqrtT  # standardized log-moneyness
valid = np.isfinite(IV) & np.isfinite(M)
print("Done.\n")

# fitted IV surface from per-ts coefficients
a_ = coeffs[:, 0:1]
b_ = coeffs[:, 1:2]
c_ = coeffs[:, 2:3]
IV_fit = a_ + b_ * M + c_ * M**2  # (N, Ks)
resid = IV - IV_fit  # IV-space residuals (N, Ks)


# ══════════════════════════════════════════════════════════════════════════════
section(1, "PER-STRIKE IV TIME SERIES")
# ══════════════════════════════════════════════════════════════════════════════
print(
    f"{'Strike':>8} {'Mean IV':>10} {'Std IV':>10} {'Min IV':>10} {'Max IV':>10} {'Valid%':>8}"
)
for j, k in enumerate(strikes.astype(int)):
    v = IV[:, j][np.isfinite(IV[:, j])]
    print(
        f"{k:>8} {v.mean():>10.5f} {v.std():>10.5f} {v.min():>10.5f} {v.max():>10.5f} {len(v)/N:>7.1%}"
    )


# ══════════════════════════════════════════════════════════════════════════════
section(2, "PER-STRIKE SMILE FIT RESIDUALS")
# ══════════════════════════════════════════════════════════════════════════════
print(f"{'Strike':>8} {'Mean resid':>12} {'Std resid':>12} {'t-stat':>10} {'Bias?':>8}")
for j, k in enumerate(strikes.astype(int)):
    r = resid[:, j][valid[:, j]]
    if len(r) < 2:
        print(f"{k:>8} {'N/A':>12}")
        continue
    t = r.mean() / (r.std() / np.sqrt(len(r)))
    sig = "YES" if abs(t) > 2 else "no"
    print(f"{k:>8} {r.mean():>12.6f} {r.std():>12.6f} {t:>10.2f} {sig:>8}")


# ══════════════════════════════════════════════════════════════════════════════
section(3, "PER-DAY COEFFICIENT BREAKDOWN")
# ══════════════════════════════════════════════════════════════════════════════
print(
    f"{'Day':>5} {'TTE_d':>7} {'a_mean':>10} {'a_std':>8} {'b_mean':>10} {'b_std':>8} {'c_mean':>10} {'c_std':>8}"
)
day_stats = []
for d in range(3):
    sl = slice(d * ROWS_PER_DAY, (d + 1) * ROWS_PER_DAY)
    ca = coeffs[sl]
    row = (np.nanmean(ca, axis=0), np.nanstd(ca, axis=0))
    day_stats.append(row)
    print(
        f"{d:>5} {TTE_DAY0_DAYS - d:>7} "
        f"{row[0][0]:>10.5f} {row[1][0]:>8.5f} "
        f"{row[0][1]:>10.5f} {row[1][1]:>8.5f} "
        f"{row[0][2]:>10.5f} {row[1][2]:>8.5f}"
    )

print("\nDay-to-day Δa (in σ units of day 0):")
sig0 = day_stats[0][1][0]
for d in range(1, 3):
    delta = day_stats[d][0][0] - day_stats[d - 1][0][0]
    nsig = delta / sig0 if sig0 > 0 else np.nan
    flag = "  ← FLAG (>2σ)" if abs(nsig) > 2 else ""
    print(f"  Day {d-1} → Day {d}: Δa = {delta:+.5f}  ({nsig:+.2f}σ){flag}")


# ══════════════════════════════════════════════════════════════════════════════
section(4, "REGRESS a_t ON REALIZED VOL OF UNDERLYING")
# ══════════════════════════════════════════════════════════════════════════════
# RV in return space → annualized. Steps per year = ROWS_PER_DAY * DAYS_PER_YEAR.
steps_per_year = ROWS_PER_DAY * DAYS_PER_YEAR
log_ret = np.diff(np.log(S_flat), prepend=np.log(S_flat[0]))
rv_series = (pd.Series(log_ret).rolling(RV_WINDOW).std().to_numpy()) * np.sqrt(
    steps_per_year
)
rv_series[:RV_WINDOW] = np.nan  # burn-in

mask4 = np.isfinite(a_flat) & np.isfinite(rv_series)
X4 = sm.add_constant(rv_series[mask4])
res4 = sm.OLS(a_flat[mask4], X4).fit()

print(f"RV window: {RV_WINDOW} ts  |  annualization: ×√({steps_per_year:.0e})")
print(f"OLS: a_t = α + β·RV_t  (n={mask4.sum()})")
print(
    f"  α = {res4.params[0]:.6f}  (t={res4.tvalues[0]:.2f},  p={res4.pvalues[0]:.2e})"
)
print(
    f"  β = {res4.params[1]:.6f}  (t={res4.tvalues[1]:.2f},  p={res4.pvalues[1]:.2e})"
)
print(f"  R² = {res4.rsquared:.5f}")
acf1_r4 = np.corrcoef(res4.resid[:-1], res4.resid[1:])[0, 1]
print(f"  Lag-1 residual autocorr = {acf1_r4:.4f}")
verdict4 = (
    "a PREDICTABLE from S dynamics → model a_t as f(RV)"
    if res4.pvalues[1] < 0.05
    else "no significant vol-spot relation → RV not a driver"
)
print(f"  → {verdict4}")


# ══════════════════════════════════════════════════════════════════════════════
section(5, "HALF-LIFE OF a SHOCKS")
# ══════════════════════════════════════════════════════════════════════════════
# Full-sample AR(1)
phi_full, _, pv_full = ar1_ols(a_flat)
hl_full = (
    -np.log(2) / np.log(abs(phi_full)) if abs(phi_full) not in (0.0, 1.0) else np.inf
)

# Rolling AR(1): re-use 1000-ts windows (subsample every 100 for speed)
roll_phis = []
WIN = 1000
for start in range(0, N - WIN, 100):
    phi_w, _, _ = ar1_ols(a_flat[start : start + WIN])
    if np.isfinite(phi_w):
        roll_phis.append(phi_w)
roll_phis = np.array(roll_phis)
phi_roll = roll_phis.mean()
hl_roll = (
    -np.log(2) / np.log(abs(phi_roll)) if abs(phi_roll) not in (0.0, 1.0) else np.inf
)

# OU: fit da_t = κ(μ - a_{t-1})dt + σ dW  via OLS on differences
a_clean = a_flat[np.isfinite(a_flat)]
da = np.diff(a_clean)
X_ou = sm.add_constant(a_clean[:-1])
res_ou = sm.OLS(da, X_ou).fit()
kappa = -res_ou.params[1]
mu_ou = res_ou.params[0] / kappa if kappa > 0 else np.nan
sigma_ou = res_ou.resid.std()
hl_ou = np.log(2) / kappa if kappa > 0 else np.inf

print(f"Full-sample AR(1):  φ = {phi_full:.5f}  →  half-life = {hl_full:.1f} ts")
print(f"Rolling mean AR(1): φ = {phi_roll:.5f}  →  half-life = {hl_roll:.2f} ts")
print(
    f"Implied optimal EMA window (3×τ):  full = {3*hl_full:.0f} ts  |  rolling = {3*hl_roll:.1f} ts"
)
print(f"\nOU fit (on differences):")
print(f"  κ (mean-reversion speed) = {kappa:.6f}")
print(f"  μ (long-run level)       = {mu_ou:.5f}")
print(f"  σ (innovation noise)     = {sigma_ou:.6f}  per step")
print(f"  OU half-life             = {hl_ou:.1f} ts")


# ══════════════════════════════════════════════════════════════════════════════
section(6, "PCA ON IV MATRIX ACROSS STRIKES")
# ══════════════════════════════════════════════════════════════════════════════
all_valid_rows = valid.all(axis=1)
IV_pca = IV[all_valid_rows]  # keep fully valid rows
IV_center = IV_pca - IV_pca.mean(axis=1, keepdims=True)  # remove per-ts mean
_, s_pca, Vt = np.linalg.svd(IV_center, full_matrices=False)
var_exp = s_pca**2 / (s_pca**2).sum()

print(
    f"Rows used (all 6 strikes valid): {all_valid_rows.sum()} / {N} ({all_valid_rows.mean():.1%})"
)
print(f"\n{'PC':>4} {'Var explained':>15} {'Cumulative':>12}")
cum = 0.0
for i, v in enumerate(var_exp[:6]):
    cum += v
    print(f"{i+1:>4} {v:>14.2%} {cum:>11.2%}")

print(f"\nPC1 loadings per strike:")
for j, k in enumerate(strikes.astype(int)):
    print(f"  K={k}: {Vt[0, j]:+.4f}")

flag6 = (
    "PC1 > 95% → only a matters; b,c negligible → hardcode b=0, c=mean"
    if var_exp[0] > 0.95
    else f"Multiple factors ({var_exp[0]:.0%} in PC1) → smile shape dynamic"
)
print(f"\n→ {flag6}")


# ══════════════════════════════════════════════════════════════════════════════
section(7, "RESIDUAL AUTOCORRELATION (IV SPACE)")
# ══════════════════════════════════════════════════════════════════════════════
print(
    f"{'Strike':>8} {'LB-stat(10)':>12} {'LB-pval':>10} {'Lag-1 ACF':>11} {'Signal?':>9}"
)
for j, k in enumerate(strikes.astype(int)):
    r = resid[:, j][valid[:, j]]
    if len(r) < 30:
        print(f"{k:>8}  N/A")
        continue
    lb = acorr_ljungbox(r, lags=[10], return_df=True)
    stat = lb["lb_stat"].values[0]
    pval = lb["lb_pvalue"].values[0]
    acf1 = np.corrcoef(r[:-1], r[1:])[0, 1]
    sig = "YES" if pval < 0.05 else "no"
    print(f"{k:>8} {stat:>12.2f} {pval:>10.3e} {acf1:>11.4f} {sig:>9}")

print(
    "\n→ 'YES' = autocorrelation in IV residuals → predictable mispricings → scalping edge real"
)


# ══════════════════════════════════════════════════════════════════════════════
section(8, "REALIZED VOL vs ATM IMPLIED VOL")
# ══════════════════════════════════════════════════════════════════════════════
rv_ann = (pd.Series(log_ret).rolling(RV_WINDOW).std().to_numpy()) * np.sqrt(
    steps_per_year
)
atm_iv = a_flat.copy()
mask8 = np.isfinite(atm_iv) & np.isfinite(rv_ann) & (rv_ann > 0)
ratio = atm_iv[mask8] / rv_ann[mask8]
corr8 = np.corrcoef(atm_iv[mask8], rv_ann[mask8])[0, 1]

print(
    f"RV window: {RV_WINDOW} ts  |  annualized via ×√(steps_per_year={steps_per_year:.0e})"
)
print(f"Mean ATM IV (a_t) = {atm_iv[mask8].mean():.5f}")
print(f"Mean RV ann       = {rv_ann[mask8].mean():.5f}")
print(f"Mean IV/RV ratio  = {ratio.mean():.4f}  (std = {ratio.std():.4f})")
print(f"Corr(IV, RV)      = {corr8:.4f}")
if ratio.mean() > 1.10:
    print("→ Options RICH (IV/RV > 1.10) → sell-vol bias justified")
elif ratio.mean() < 0.90:
    print("→ Options CHEAP (IV/RV < 0.90) → buy-vol bias justified")
else:
    print("→ Options FAIRLY PRICED (0.90 ≤ IV/RV ≤ 1.10) → no directional vol bias")


# ══════════════════════════════════════════════════════════════════════════════
section(9, "VOUCHER SPREAD & NOISE FLOOR (RAW BOOK)")
# ══════════════════════════════════════════════════════════════════════════════
price_dfs = []
for d in range(3):
    p = pd.read_csv(DATA / f"prices_round_3_day_{d}.csv", sep=";")
    p["day"] = d
    price_dfs.append(p)
prices_raw = pd.concat(price_dfs, ignore_index=True)

vev = prices_raw[prices_raw["product"].str.match(r"^VEV_\d+$")].copy()
vev["strike"] = vev["product"].str.split("_").str[-1].astype(int)
vev = vev[vev["strike"].isin(STRIKE_WHITELIST)]
vev["spread"] = vev["ask_price_1"] - vev["bid_price_1"]
vev = vev.dropna(subset=["spread", "bid_price_1", "ask_price_1"])
vev = vev[vev["spread"] >= 0]

print(
    f"{'Strike':>8} {'Mean spread':>13} {'Median':>10} {'Min':>8} {'Max':>8} {'n rows':>8}"
)
spread_by_k = {}
for k in sorted(STRIKE_WHITELIST):
    sub = vev[vev["strike"] == k]["spread"]
    spread_by_k[k] = sub.mean() if len(sub) > 0 else np.nan
    if len(sub) == 0:
        print(f"{k:>8} {'no data':>13}")
        continue
    print(
        f"{k:>8} {sub.mean():>13.3f} {sub.median():>10.3f} {sub.min():>8.3f} {sub.max():>8.3f} {len(sub):>8}"
    )

# Noise floor: half-spread / vega, in IV units
med_S = float(np.nanmedian(S_flat))
med_T = (TTE_DAY0_DAYS - 1) / DAYS_PER_YEAR  # day 1 as median
med_iv = float(np.nanmedian(a_flat))
print(f"\nNoise floor estimate (half-spread / vega) in IV units:")
print(f"Reference: S={med_S:.0f}, TTE={med_T:.4f}yr, ATM IV={med_iv:.4f}")
print(
    f"{'Strike':>8} {'Half-spread':>13} {'Vega':>10} {'IV noise floor':>16} {'vs resid std':>14}"
)
for j, k in enumerate(strikes.astype(int)):
    if np.isnan(spread_by_k.get(k, np.nan)):
        continue
    sT_k = med_iv * np.sqrt(med_T)
    d1_k = (np.log(med_S / k) + 0.5 * med_iv**2 * med_T) / sT_k
    vega_k = med_S * norm.pdf(d1_k) * np.sqrt(med_T)
    noise = (spread_by_k[k] / 2) / vega_k if vega_k > 0 else np.nan
    r_std = resid[:, j][valid[:, j]].std() if valid[:, j].sum() > 1 else np.nan
    ratio_n = (
        noise / r_std
        if (np.isfinite(noise) and np.isfinite(r_std) and r_std > 0)
        else np.nan
    )
    print(
        f"{k:>8} {spread_by_k[k]/2:>13.3f} {vega_k:>10.4f} {noise:>16.5f} {ratio_n:>13.2f}×"
    )
print(
    "→ IV noise floor > resid std: microstructure dominates → widen THR_OPEN accordingly"
)


# ══════════════════════════════════════════════════════════════════════════════
section(10, "VEGA PROFILE PER STRIKE")
# ══════════════════════════════════════════════════════════════════════════════
# Use per-ts a_t as IV estimate; fill NaN strikes with median IV
iv_for_vega = np.where(valid, IV, np.nanmedian(IV))
sT_mat = iv_for_vega * sqrtT
d1_mat = (np.log(K_arr / S_arr) + 0.5 * iv_for_vega**2 * T_arr) / np.maximum(
    sT_mat, 1e-12
)
vega_mat = S_arr * norm.pdf(d1_mat) * sqrtT  # (N, Ks)

print(f"LOW_VEGA_THR = {LOW_VEGA_THR}")
print(
    f"{'Strike':>8} {'Mean vega':>12} {'Std vega':>12} {'Min vega':>10} {'% below thr':>13}"
)
for j, k in enumerate(strikes.astype(int)):
    vg = vega_mat[:, j][valid[:, j]]
    below = (vg < LOW_VEGA_THR).mean() * 100
    print(
        f"{k:>8} {vg.mean():>12.4f} {vg.std():>12.4f} {vg.min():>10.4f} {below:>12.1f}%"
    )
print("→ strikes with many rows below threshold should get LOW_VEGA_THR_ADJ applied")


# ══════════════════════════════════════════════════════════════════════════════
section(11, "INFORMED TRADER ANALYSIS [SKIPPED BY USER]")
# ══════════════════════════════════════════════════════════════════════════════
print("Skipped.")


# ══════════════════════════════════════════════════════════════════════════════
section(12, "CROSS-CORRELATION a_t vs UNDERLYING MID")
# ══════════════════════════════════════════════════════════════════════════════
mask12 = np.isfinite(a_flat) & np.isfinite(S_flat)
corr_lv = np.corrcoef(a_flat[mask12], S_flat[mask12])[0, 1]

da_s = np.diff(a_flat)
dS_s = np.diff(S_flat)
mask12d = np.isfinite(da_s) & np.isfinite(dS_s)
corr_df = np.corrcoef(da_s[mask12d], dS_s[mask12d])[0, 1]

# rolling 1000-ts correlation (levels)
roll_c = (
    pd.Series(a_flat, dtype=float)
    .rolling(1000)
    .corr(pd.Series(S_flat, dtype=float))
    .dropna()
    .to_numpy()
)

# lag cross-corr: which leads which?
da_f = da_s[mask12d]
dS_f = dS_s[mask12d]
lag_dS_leads = np.corrcoef(da_f[1:], dS_f[:-1])[0, 1]  # ΔS_{t-1} predicts Δa_t
lag_da_leads = np.corrcoef(da_f[:-1], dS_f[1:])[0, 1]  # Δa_{t-1} predicts ΔS_t

print(f"Corr(a, S)      [levels]    = {corr_lv:+.4f}")
print(f"Corr(Δa, ΔS)    [diffs]     = {corr_df:+.4f}")
print(
    f"Rolling 1000-ts corr(a, S): mean = {roll_c.mean():+.4f}  std = {roll_c.std():.4f}"
)
print(f"\nLag structure (first differences):")
print(f"  ΔS_{{t-1}} → Δa_t  (underlying leads IV): {lag_dS_leads:+.4f}")
print(f"  Δa_{{t-1}} → ΔS_t  (IV leads underlying): {lag_da_leads:+.4f}")

if abs(corr_df) > 0.3:
    lead = (
        "underlying leads IV"
        if abs(lag_dS_leads) > abs(lag_da_leads)
        else "IV leads underlying"
    )
    print(
        f"→ |Corr(Δa, ΔS)| > 0.3: vol-spot dependency detected ({lead}) → vanna hedge term needed"
    )
else:
    print("→ No significant vol-spot dependency → no vanna correction needed")


# ══════════════════════════════════════════════════════════════════════════════
section(13, "FIXED-WINDOW THEO ERROR (EMA window = 20)")
# ══════════════════════════════════════════════════════════════════════════════
alpha_ema = 2 / (EMA_WINDOW + 1)
print(f"EMA α = {alpha_ema:.4f}  (window = {EMA_WINDOW})")
print(
    f"\n{'Strike':>8} {'MSE raw':>12} {'MSE ema-adj':>14} "
    f"{'Lag1 raw':>10} {'Lag1 adj':>10} {'WN?':>6}"
)

for j, k in enumerate(strikes.astype(int)):
    r = resid[:, j][valid[:, j]]
    if len(r) < 50:
        print(f"{k:>8} N/A")
        continue

    # build EMA of IV residuals (mirrors strategy's mean_theo_diffs per option)
    ema = np.empty_like(r)
    ema[0] = r[0]
    for t in range(1, len(r)):
        ema[t] = alpha_ema * r[t] + (1 - alpha_ema) * ema[t - 1]
    r_adj = r - ema

    mse_raw = float(np.mean(r**2))
    mse_adj = float(np.mean(r_adj**2))
    acf_raw = float(np.corrcoef(r[:-1], r[1:])[0, 1])
    acf_adj = float(np.corrcoef(r_adj[:-1], r_adj[1:])[0, 1])
    lb_pval = acorr_ljungbox(r_adj, lags=[10], return_df=True)["lb_pvalue"].values[0]
    wn = "YES" if lb_pval > 0.05 else "no"

    print(
        f"{k:>8} {mse_raw:>12.6f} {mse_adj:>14.6f} {acf_raw:>10.4f} {acf_adj:>10.4f} {wn:>6}"
    )

print(
    "\nWN=YES  → EMA(20) adequately de-trends IV residuals; scalping threshold calibration valid"
)
print("WN=no   → autocorr persists after EMA → raise window or adopt dynamic a_t refit")
