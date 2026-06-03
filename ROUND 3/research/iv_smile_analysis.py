import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------- config ----------
ROOT = Path(__file__).parent
DATA = ROOT.parent / "data" / "round3"
OPT_FILE = DATA / "clean_options_chain.csv"
UND_FILE = DATA / "clean_VELVETFRUIT_EXTRACT.csv"
OUT_COEFFS = ROOT / "iv_smile_coeffs.csv"

TTE_DAY0_DAYS = 8  # TTE in days at row 0
ROWS_PER_DAY = 10000
ROLL_WIN = 100
NEWTON_ITERS = 50
NEWTON_TOL = 1e-8

# Deep ITM (<=4500): vega->0, Newton unstable. Deep OTM (>=6000): tick-floor biases IV.
STRIKE_WHITELIST = {4500, 5100, 5200, 5300, 5400, 5500, 6000, 6500}

# ---------- load ----------
opt = pd.read_csv(OPT_FILE)
und = pd.read_csv(UND_FILE).rename(columns={"mid_price": "S"})
df = (
    opt.merge(und, on="global_ts", how="inner")
    .sort_values("global_ts")
    .reset_index(drop=True)
)

strike_cols_all = [c for c in opt.columns if c.startswith("VEV_")]
strike_cols = [c for c in strike_cols_all if int(c.split("_")[1]) in STRIKE_WHITELIST]
strikes = np.array([int(c.split("_")[1]) for c in strike_cols], dtype=float)
K = strikes[None, :]
print(
    f"[filter] strikes kept: {strikes.astype(int).tolist()} "
    f"(dropped {sorted(set(int(c.split('_')[1]) for c in strike_cols_all) - STRIKE_WHITELIST)})"
)

N = len(df)
S = df["S"].to_numpy()[:, None]
C = df[strike_cols].to_numpy().astype(float)

# continuous TTE: decays 1 day every ROWS_PER_DAY rows
tte_raw = np.clip(TTE_DAY0_DAYS * ROWS_PER_DAY - np.arange(N), 1, None)
T = (tte_raw / ROWS_PER_DAY)[:, None]  # days, shape (N, 1)
sqrtT = np.sqrt(T)


# ---------- BS IV via Newton ----------
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


print(f"[load] N={N} timestamps, Ks={len(strikes)} strikes")
print("[BS]  computing IVs ...")
IV = bs_call_iv(S, K, T, C)

M = np.log(K / S) / sqrtT
valid = np.isfinite(IV) & np.isfinite(M)
print(f"[BS]  valid IV points: {valid.sum()} / {IV.size} ({valid.mean():.1%})")


# ---------- 1. GLOBAL deg-2 fit ----------
m_all = M[valid]
iv_all = IV[valid]
X_g = np.column_stack([np.ones_like(m_all), m_all, m_all**2])
ols_g = OLS(iv_all, X_g).fit()
beta_g, se_g = ols_g.params, ols_g.bse

print("\n================ GLOBAL SMILE FIT (deg-2) ================")
print(f"IV(m) = a + b*m + c*m^2,    m = log(K/S)/sqrt(T)")
print(f"  a (level)     = {beta_g[0]: .6f}  ± {se_g[0]:.6f}")
print(f"  b (skew)      = {beta_g[1]: .6f}  ± {se_g[1]:.6f}")
print(f"  c (curvature) = {beta_g[2]: .6f}  ± {se_g[2]:.6f}")
print(f"  R^2 = {ols_g.rsquared:.5f}   n = {len(iv_all)}")


# ---------- helpers: batched WLS and coefficient std ----------
def batched_ols(basis, IV_, w):
    """Batched WLS across N timestamps. Returns (coeffs, XtWX, nvalid, ok)."""
    Wb = w[..., None] * basis
    XtWX = np.einsum("nki,nkj->nij", Wb, basis)
    XtWy = np.einsum("nki,nk->ni", Wb, IV_)
    nvalid = w.sum(axis=1)
    d = basis.shape[-1]
    ok = nvalid >= (d + 1)
    coeffs = np.full((len(IV_), d), np.nan)
    if ok.any():
        try:
            coeffs[ok] = np.linalg.solve(XtWX[ok], XtWy[ok, :, None]).squeeze(-1)
        except (np.linalg.LinAlgError, ValueError):
            for i in np.where(ok)[0]:
                try:
                    coeffs[i] = np.linalg.solve(XtWX[i], XtWy[i])
                except np.linalg.LinAlgError:
                    pass
    return coeffs, XtWX, nvalid, ok


def coeff_stds(XtWX, coeffs, IV_, basis, w, nvalid, ok):
    resid = (IV_ - np.einsum("nkj,nj->nk", basis, np.nan_to_num(coeffs, nan=0.0))) * w
    sigma2 = (resid**2).sum(axis=1) / np.maximum(nvalid - basis.shape[-1], 1)
    d = basis.shape[-1]
    cov_diag = np.full((len(IV_), d), np.nan)
    for i in np.where(ok)[0]:
        try:
            cov_diag[i] = sigma2[i] * np.diag(np.linalg.inv(XtWX[i]))
        except np.linalg.LinAlgError:
            pass
    return np.sqrt(np.clip(cov_diag, 0, None))


# ---------- 2. PER-TS deg-2 fit ----------
w = valid.astype(float)
M_ = np.nan_to_num(M, nan=0.0)
IV_ = np.nan_to_num(IV, nan=0.0)
ones = np.ones_like(M_)

basis2 = np.stack([ones, M_, M_**2], axis=-1)
coeffs2, XtWX2, nvalid, ok2 = batched_ols(basis2, IV_, w)
std2 = coeff_stds(XtWX2, coeffs2, IV_, basis2, w, nvalid, ok2)

coef_df = pd.DataFrame(
    {
        "global_ts": df["global_ts"].to_numpy(),
        "a": coeffs2[:, 0],
        "b": coeffs2[:, 1],
        "c": coeffs2[:, 2],
        "a_std": std2[:, 0],
        "b_std": std2[:, 1],
        "c_std": std2[:, 2],
        "n_valid": nvalid.astype(int),
    }
)
coef_df.to_csv(OUT_COEFFS, index=False)

good2 = ok2 & np.isfinite(coeffs2).all(axis=1)
print("\n================ PER-TS COEFFICIENT STATISTICS ================")
print(f"{'coeff':<6} {'mean':>12} {'std':>12} {'CV':>10} {'min':>12} {'max':>12}")
for j, name in enumerate(["a", "b", "c"]):
    x = coeffs2[good2, j]
    cv = x.std() / abs(x.mean()) if x.mean() != 0 else np.nan
    print(
        f"{name:<6} {x.mean():12.6f} {x.std():12.6f} {cv:10.4f} {x.min():12.6f} {x.max():12.6f}"
    )
print(f"ts with valid fit: {good2.sum()} / {N} ({good2.mean():.1%})")
print(f"per-ts coeffs saved -> {OUT_COEFFS}")


# ---------- 3. DRIFT: linear fit coeff(t) = alpha + beta * global_ts ----------
def fit_drift(series, ts):
    """Fit series = alpha + beta*ts. Returns (alpha, beta, se_beta, r2)."""
    mask = np.isfinite(series)
    if mask.sum() < 4:
        return np.nan, np.nan, np.nan, np.nan
    X = np.column_stack([np.ones(mask.sum()), ts[mask]])
    res = OLS(series[mask], X).fit()
    return res.params[0], res.params[1], res.bse[1], res.rsquared


ts_arr = coef_df["global_ts"].to_numpy()
print("\n================ DRIFT OF SMILE COEFFICIENTS ================")
print(f"  coeff(t) = alpha + beta * global_ts")
print(f"{'coeff':<6} {'alpha':>14} {'beta':>16} {'se_beta':>14} {'R2':>8}")
drift = {}
for j, name in enumerate(["a", "b", "c"]):
    alpha, beta, se_beta, r2 = fit_drift(coeffs2[:, j], ts_arr)
    drift[name] = dict(alpha=alpha, beta=beta, se_beta=se_beta, r2=r2)
    print(f"{name:<6} {alpha:14.6f} {beta:16.4e} ± {se_beta:.4e}   R2={r2:.4f}")


# ---------- 4. AR structure ----------
def ar1_fit(x):
    x = x[np.isfinite(x)]
    if len(x) < 4:
        return np.nan, np.nan, np.nan, np.nan
    y = x[1:]
    X = np.column_stack([np.ones(len(x) - 1), x[:-1]])
    res = OLS(y, X).fit()
    return res.params[1], res.params[0], res.bse[1], res.pvalues[1]


def rolling_ar1(series, win=ROLL_WIN):
    n = len(series)
    phis = np.full(n, np.nan)
    for start in range(0, n - win + 1):
        phis[start + win - 1] = ar1_fit(series[start : start + win])[0]
    return phis


print("\n================ AR(1) STRUCTURE OF COEFFICIENTS ================")
ar_summary = {}
for j, name in enumerate(["a", "b", "c"]):
    s = coeffs2[:, j].copy()
    phi, _, se_phi, pv = ar1_fit(s)
    s_clean = s[np.isfinite(s)]
    adf_p = adfuller(s_clean, autolag="AIC")[1] if len(s_clean) > 20 else np.nan
    ar_summary[name] = dict(phi=phi, se=se_phi, p=pv, adf_p=adf_p)
    has_ar = (abs(phi) > 0.1) and (pv < 0.05)
    print(
        f"[{name}] AR(1): phi={phi: .5f} ± {se_phi:.5f}  "
        f"(p={pv:.2e})  ADF p={adf_p:.2e}  AR1={'YES' if has_ar else 'no'}"
    )

print(f"\n[rolling AR(1), window = {ROLL_WIN} ts]")
for j, name in enumerate(["a", "b", "c"]):
    phis = rolling_ar1(coeffs2[:, j])
    phis_ok = phis[np.isfinite(phis)]
    if len(phis_ok):
        print(
            f"  {name}: phi mean={phis_ok.mean(): .4f} std={phis_ok.std():.4f} "
            f"min={phis_ok.min(): .4f} max={phis_ok.max():.4f}  "
            f"|phi|>0.5 in {(np.abs(phis_ok) > 0.5).mean():.1%} of windows"
        )


# ---------- 5. STABILITY VERDICT ----------
def _verdict(phi, adf_p, cv):
    stationary = adf_p < 0.05
    if stationary and abs(phi) < 0.3 and cv < 0.10:
        return "HARDCODE-OK"
    if abs(phi) > 0.7:
        return "MODEL-EVOLUTION (high persistence)"
    if not stationary:
        return "MODEL-EVOLUTION (non-stationary)"
    return "MODEL-EVOLUTION (large dispersion)"


print("\n================ STABILITY VERDICT ================")
verdicts = []
for j, name in enumerate(["a", "b", "c"]):
    x = coeffs2[good2, j]
    mean, std = x.mean(), x.std()
    cv = std / abs(mean) if mean != 0 else np.inf
    phi, adf_p = ar_summary[name]["phi"], ar_summary[name]["adf_p"]
    beta = drift[name]["beta"]
    v = _verdict(phi, adf_p, cv)
    verdicts.append(v)
    print(
        f"  {name}: mean={mean: .5f} std={std:.5f} CV={cv:.3f} "
        f"phi={phi: .3f} ADFp={adf_p:.1e} drift_beta={beta:.3e} -> {v}"
    )

print("\nOVERALL:")
if all("MODEL" not in v for v in verdicts):
    print(
        "  All coefficients stable & mean-reverting.  -> Safe to HARDCODE from global fit."
    )
else:
    print("  At least one coefficient needs dynamic modelling.")
    print("  -> Price options off rolling/recent-ts smile fit, not fixed.")
    print("  -> Mispricing signal = IV_market - IV_fit(m) from *current* smile.")


# ---------- 6. PLOT: coefficient evolution with drift line ----------
def plot_coeff_evolution(
    coef_df, drift, roll_win=ROLL_WIN, out_path=ROOT / "iv_coeff_evolution.png"
):
    ts = coef_df["global_ts"].to_numpy()
    day_boundaries = [
        ts[i * ROWS_PER_DAY] for i in range(1, 3) if i * ROWS_PER_DAY < len(ts)
    ]
    labels = {"a": "a  (level)", "b": "b  (skew)", "c": "c  (curvature)"}

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fmt = mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    for ax, name in zip(axes, ["a", "b", "c"]):
        raw = coef_df[name].to_numpy()
        rm = (
            coef_df[name].rolling(roll_win, min_periods=roll_win // 2).mean().to_numpy()
        )
        rs = coef_df[name].rolling(roll_win, min_periods=roll_win // 2).std().to_numpy()
        ax.plot(ts, raw, lw=0.4, alpha=0.35, color="steelblue", label="per-ts")
        ax.plot(ts, rm, lw=1.2, color="steelblue", label=f"roll mean ({roll_win})")
        ax.fill_between(
            ts, rm - rs, rm + rs, alpha=0.18, color="steelblue", label="±1σ"
        )
        d = drift[name]
        if np.isfinite(d["beta"]):
            drift_line = d["alpha"] + d["beta"] * ts
            ax.plot(
                ts,
                drift_line,
                lw=1.5,
                color="crimson",
                ls="--",
                label=f"drift  β={d['beta']:.3e}  R²={d['r2']:.3f}",
            )
        for b in day_boundaries:
            ax.axvline(b, color="grey", ls="--", lw=0.7, alpha=0.6)
        ax.set_ylabel(labels[name], fontsize=10)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.25)
        ax.xaxis.set_major_formatter(fmt)
    axes[0].set_title(
        f"IV smile coefficients  (blue=rolling, red=linear drift,  roll={roll_win})"
    )
    axes[2].set_xlabel("global_ts")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[plot] coeff evolution saved -> {out_path}")


plot_coeff_evolution(coef_df, drift)


# ---------- 7. PLOT: 3-D IV surface ----------
def plot_iv_surface(
    IV, M, global_ts, out_path=ROOT / "iv_surface.png", subsample=3_000
):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    step = max(1, len(global_ts) // subsample)
    idx = np.arange(0, len(global_ts), step)
    iv_s, m_s, ts_s = IV[idx], M[idx], global_ts[idx]
    m_cols = np.nanmean(m_s, axis=0)
    X = np.tile(m_cols, (len(idx), 1))
    Y = np.tile(ts_s[:, None], (1, len(m_cols)))
    Z = np.ma.masked_invalid(iv_s)
    xlim = (float(np.nanmin(m_cols)) - 0.02, float(np.nanmax(m_cols)) + 0.02)

    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        X,
        Y,
        Z,
        cmap="viridis",
        lw=0.2,
        antialiased=True,
        alpha=0.88,
        rcount=subsample,
        ccount=len(m_cols),
    )
    fig.colorbar(surf, ax=ax, shrink=0.45, pad=0.10, label="IV")
    ax.set_xlabel("log-moneyness  m = log(K/S)/√T")
    ax.set_ylabel("global_ts")
    ax.set_zlabel("IV")
    ax.set_title("IV surface  (near-ATM strikes, viridis)")
    ax.set_xlim(xlim)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] IV surface saved -> {out_path}")


plot_iv_surface(IV, M, df["global_ts"].to_numpy())
