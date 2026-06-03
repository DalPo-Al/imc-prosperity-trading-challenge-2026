# Research

## Purpose

Round 3 volatility-surface research on the `VELVETFRUIT_EXTRACT` (VEV) option chain: can a tradeable edge be extracted from the IV smile / term structure across strikes 4000–6500? Underlying VEV plus 10 European-style call vouchers; near-ATM block 5000–5500 is the focus.

## Methodology

Logical order (not file order):

- Invert market prices to per-tick implied vol and fit the smile `IV(m) = a + b·m + c·m²`, `m = log(K/S)/√T`, per timestamp; persist coefficients — `iv_smile_analysis.py` → `iv_smile_coeffs.csv`.
- Characterise per-strike IV level/noise, smile residual bias, PCA factor structure, vol–spot dependence (vanna) — `iv_deep_analysis.py`.
- Per-strike IV time series with EMA overlay to expose slow modes — `iv_per_strike_ema.py`.
- Smile-coefficient term structure vs TTE and per-strike static bias; project to live TTE 5→2 — `term_structure_and_bias_analysis.py` → `term_structure_findings.json`.
- De-trended smile-deviation reversion timing per strike — `smile_reversion_analysis.py`.
- Deep ITM/OTM checks: intrinsic-value violations, tick-floor artefacts — `deep_strikes_analysis.py`.
- Gamma-scalping viability (σ_rv vs σ_iv, discrete-hedge PnL, strike ranking) using hardcoded GARCH(1,1)-t params — `gamma_scalping_analysis.py`.

## Key Findings

- **The one robust result: very fast mean reversion in single components of the surface.** Across the ~8 usable strikes, de-trended per-strike IV/price deviations revert almost instantly — ACF half-life ≈ 1 tick, deviations are white noise at tick+1 (`smile_reversion_analysis.py`, FINDINGS §6). This is the only finding worth carrying forward.
- **Everything else is weak.** Treat the items below as context, not signal:
  - IV sits ~0.9 %/√day below GARCH conditional vol persistently (mean IV/RV ≈ 0.565) — structural, but never turned into reliable PnL.
  - Smile-level coefficient `a` drifts slowly (AR(1) φ ≈ 0.97, half-life ≈ 27 ticks); `b`, `c` dominated by noise (CV 115–562 %).
  - Term structure of `a`, `b`, `c` vs TTE: slopes significant but R² < 2 %; linear projection to TTE 5→2 is unreliable (made per-strike bias worse — keep OLD coefficients).
  - Persistent per-strike polynomial bias (e.g. K=5400 ≈ −2.2, K=5300 ≈ +1.3) — artefact of a degree-2 fit, not tradeable; would fire continuous one-sided orders if uncorrected.
  - Deep ITM quoted below intrinsic 10 % (K=4000) / 29 % (K=4500) of ticks; deep OTM pinned at the 0.5 tick floor. Both excluded from smile fitting.
- **Outcome: no satisfying implementation.** No team member could turn any of this into a strategy that performed acceptably in the live arena. The research is retained as a negative result plus the one mean-reversion observation.

## File Reference

| File                                        | Role            | Notes                                                                                       |
| ------------------------------------------- | --------------- | ------------------------------------------------------------------------------------------- |
| `FINDINGS.md`                               | Primary writeup | Full Round 3 empirical log (GARCH, smile, deep strikes, reversion, term structure).         |
| `iv_smile_analysis.py`                      | Analysis script | Per-tick IV inversion + smile fit; produces `iv_smile_coeffs.csv` (not in repo).            |
| `iv_deep_analysis.py`                       | Analysis script | 13-section smile dynamics: residual bias, PCA, vanna, noise floor.                          |
| `iv_per_strike_ema.py`                      | Analysis script | Per-strike IV time series with EMA overlay.                                                 |
| `term_structure_and_bias_analysis.py`       | Analysis script | Coefficient term structure + per-strike static bias; writes `term_structure_findings.json`. |
| `smile_reversion_analysis.py`               | Analysis script | De-trended deviation reversion timing (source of the 1-tick half-life result).              |
| `deep_strikes_analysis.py`                  | Analysis script | Deep ITM/OTM intrinsic-violation and tick-floor checks.                                     |
| `gamma_scalping_analysis.py`                | Analysis script | Gamma-scalping viability; GARCH params hardcoded (fitted elsewhere).                        |
| `hyperopt_params.py`                        | Config          | Discrete search grid for downstream `OptionTrader.py`. Not imported by other files here.    |
| `term_structure_findings.json`              | Output data     | Term-structure regression + per-strike bias dicts.                                          |
| `images/acf_detrended.png`                  | Artefact        | ACF of de-trended per-strike deviations; `smile_reversion_analysis.py`.                     |
| `images/reversion_durations.png`            | Artefact        | Reversion episode duration histogram; `smile_reversion_analysis.py`.                        |
| `images/per_strike_bias.png`                | Artefact        | Per-strike systematic smile bias; `term_structure_and_bias_analysis.py`.                    |
| `images/term_structure_abc.png`             | Artefact        | a,b,c vs TTE regression; `term_structure_and_bias_analysis.py`.                             |
| `images/detrended_deviation_timeseries.png` | Artefact        | De-trended deviation time series; `smile_reversion_analysis.py`.                            |
| `images/summary_bars.png`                   | Artefact        | Reversion summary bar chart; `smile_reversion_analysis.py`.                                 |
| `images/iv(relevant)/`                      | Artefact dir    | IV visualisations: coeff evolution, per-strike subplots (plain, Kalman, combined).          |
| `images/weak/`                              | Artefact dir    | Weak/exploratory plots: gamma PnL, discrete hedge, deviation comparisons (not cited).       |

## Dependencies

- `numpy` — numerics throughout.
- `pandas` — data loading / tabular manipulation.
- `scipy` — `norm` for Black-Scholes, `brentq` for IV root-finding.
- `matplotlib` — all plots (Agg backend).
- `statsmodels` — ADF, OLS, Ljung-Box (`iv_smile_analysis.py`, `iv_deep_analysis.py`).

GARCH(1,1)-t parameters are hardcoded from a fit done outside this directory (no `arch` dependency here).

## Replication

1. Place the Round 3 inputs under `../data/round3/` (NOT in this repo): `clean_options_chain.csv`, `clean_VELVETFRUIT_EXTRACT.csv`, `prices_round_3_day_{0,1,2}.csv`.
2. `python iv_smile_analysis.py` → regenerates `iv_smile_coeffs.csv`.
3. `python term_structure_and_bias_analysis.py` → regenerates `term_structure_findings.json` + plots.
4. `python smile_reversion_analysis.py` → reproduces the 1-tick reversion result + `images/`.
5. Remaining scripts (`iv_deep_analysis.py`, `deep_strikes_analysis.py`, `gamma_scalping_analysis.py`, `iv_per_strike_ema.py`) are independent and read the same data + coeffs.

Missing dependency: the `../data/round3/` source files are not present in this directory; all scripts fail without them.

## Author

Giorgio Cottini — sole author of the code in this folder.

Many findings regarding term structure of implied volatility were independently confirmed by Harish Jawahar.
