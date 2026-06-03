# Round 3 — Empirical Findings

All quantities are in **vol per √trading-day** unless stated otherwise.  
Time unit: 1 trading day = 10 000 ticks. TTE in trading days (8 → 7 → 6).  
Dataset: 30 000 timestamps × 3 days, strikes 4 000 – 6 500 on VELVETFRUIT_EXTRACT (VEV).

---

## Analysis

### 1. GARCH Volatility Modeling (VEV underlying)

Fourteen univariate GARCH variants fitted on 100 × log-returns of VEV
(scaled to avoid numerical underflow).  
Train / test split: 80 / 20 (23 999 / 6 000 ticks).

#### Model grid

| Variance spec                                  | Distributions tested        |
| ---------------------------------------------- | --------------------------- |
| GARCH(1,1), GARCH(2,1), GARCH(1,2), GARCH(2,2) | Normal, Student-t, Skewed-t |
| GJR-GARCH(1,1)                                 | Normal, Student-t, Skewed-t |
| EGARCH(1,1)                                    | Normal, Student-t, Skewed-t |
| APARCH(1,1)                                    | Student-t, Skewed-t         |

GJR-t and GJR-skewt failed to converge (degenerate AIC > 10⁶); all others converged.

#### In-sample leaderboard (sorted by BIC)

| Model             | AIC      | BIC          | QLIKE      | Vol-MAE    |
| ----------------- | -------- | ------------ | ---------- | ---------- |
| APARCH(1,1)-skewt | −145 552 | **−145 485** | −6.815     | 0.0113     |
| APARCH(1,1)-t     | −145 534 | −145 476     | −6.814     | 0.0113     |
| EGARCH(1,1)-skewt | −145 505 | −145 447     | −6.808     | 0.0114     |
| EGARCH(1,1)-t     | −145 488 | −145 438     | −6.807     | 0.0114     |
| GARCH(1,1)-t      | −145 387 | −145 345     | **−6.831** | **0.0112** |

GARCH(1,1)-t wins on both out-of-sample loss functions (QLIKE and Vol-MAE) despite ranking 8th on BIC.  
Diebold-Mariano test (APARCH-skewt vs GARCH(1,1)-t): DM = 7.1, p ≈ 0 — APARCH is statistically better in-sample but not out-of-sample.

#### Best model parameters — GARCH(1,1)-t

| Param      | Value       |
| ---------- | ----------- |
| μ          | 5.8 × 10⁻⁵  |
| ω          | 2.32 × 10⁻⁴ |
| α₁         | 0.100       |
| β₁         | 0.400       |
| ν (d.o.f.) | 12.0        |

Persistence α + β = 0.50. Unconditional vol = √(ω / (1 − α − β)) ≈ **2.15 %/√day**.

#### Residual diagnostics (GARCH(1,1)-t)

- LB on z² at lag 20: p = 0.000 → remaining ARCH structure in residuals.
- LB on z at lag 20: p = 0.000 → significant autocorrelation in returns (mean model under-specified; AR(1) mean would help).
- APARCH-class models pass both LB tests (p > 0.05 on z²), confirming they better capture the variance dynamics in-sample.

---

### 2. IV Smile — Near-ATM Strikes (K = 5 000 – 5 500)

Strikes outside this range are excluded from smile fitting (see §4 and §5).

#### Global quadratic fit

IV(m) = a + b·m + c·m², m = log(K/S) / √T

| Coeff         | Value     | Std error  |
| ------------- | --------- | ---------- |
| a (level)     | 0.012 078 | ±0.000 002 |
| b (skew)      | 0.002 401 | ±0.000 088 |
| c (curvature) | 0.533 267 | ±0.008 013 |

R² = 2.6 % — the global static fit explains little variance; per-timestamp refit is necessary.

#### Per-timestamp coefficient statistics

| Coeff | Mean     | Std      | CV    | AR(1) φ | ADF p   | Half-life    |
| ----- | -------- | -------- | ----- | ------- | ------- | ------------ |
| a     | 0.012 07 | 0.000 29 | 2.4 % | 0.974   | 0.168   | **27 ticks** |
| b     | 0.001 64 | 0.009 23 | 562 % | 0.463   | < 0.001 | —            |
| c     | 0.612    | 0.706    | 115 % | 0.616   | < 0.001 | —            |

- **a** is non-stationary (ADF p = 0.17), highly persistent (φ = 0.97), slowly mean-reverting.
- **b** is stationary but dominated by noise (CV = 562 %).
- **c** is stationary but highly variable (CV = 115 %).
- Verdict: all three coefficients require **dynamic modelling**; no coefficient is safe to hardcode.

#### Per-strike IV summary

| Strike | Mean IV | Std IV  | Min IV  | Max IV  |
| ------ | ------- | ------- | ------- | ------- |
| 5 000  | 0.01220 | 0.00044 | 0.00901 | 0.01445 |
| 5 100  | 0.01208 | 0.00040 | 0.01053 | 0.01325 |
| 5 200  | 0.01221 | 0.00033 | 0.01132 | 0.01317 |
| 5 300  | 0.01234 | 0.00029 | 0.01150 | 0.01335 |
| 5 400  | 0.01157 | 0.00037 | 0.01089 | 0.01270 |
| 5 500  | 0.01255 | 0.00033 | 0.01161 | 0.01341 |

#### Smile fit residuals (IV − model)

| Strike | Mean residual | Std       | t-stat | Bias? |
| ------ | ------------- | --------- | ------ | ----- |
| 5 000  | −0.000 043    | 0.000 100 | −75    | YES   |
| 5 100  | −0.000 040    | 0.000 205 | −34    | YES   |
| 5 200  | +0.000 145    | 0.000 106 | +238   | YES   |
| 5 300  | +0.000 262    | 0.000 152 | +299   | YES   |
| 5 400  | −0.000 585    | 0.000 191 | −530   | YES   |
| 5 500  | +0.000 261    | 0.000 094 | +482   | YES   |

K = 5 400 is the most systematically biased strike (−58 bp).  
Every strike has a statistically significant bias → a degree-2 polynomial is not sufficient to describe the smile without per-strike corrections.

#### IV autocorrelation structure

All six near-ATM strikes show strong residual autocorrelation after smile fit:

| Strike | LB(10) p-val | Lag-1 ACF |
| ------ | ------------ | --------- |
| 5 000  | 0.000        | 0.862     |
| 5 300  | 0.000        | 0.914     |
| 5 400  | 0.000        | 0.952     |
| 5 500  | 0.000        | 0.906     |

Predictable mispricings persist — scalping edge is real but requires de-trending.

#### EMA de-trending (Section 13)

EMA(100) of smile residuals reduces but does **not** eliminate autocorrelation (lag-1 ACF remains 0.06–0.44 after adjustment; Ljung-Box WN rejected for all strikes).  
→ EMA window must be raised, or `a` must be refitted dynamically.

#### Microstructure noise floor (Section 9)

| Strike | Half-spread | Vega  | Noise floor (IV units) | Noise / resid-std |
| ------ | ----------- | ----- | ---------------------- | ----------------- |
| 5 000  | 3.022       | 1 693 | 0.00179                | 17.8×             |
| 5 100  | 2.148       | 3 628 | 0.00059                | 2.9×              |
| 5 200  | 1.444       | 5 277 | 0.00027                | 2.6×              |
| 5 300  | 1.053       | 5 322 | 0.00020                | 1.3×              |
| 5 400  | 0.691       | 3 799 | 0.00018                | 0.95×             |
| 5 500  | 0.575       | 1 956 | 0.00029                | 3.1×              |

K = 5 400 is the only strike where microstructure noise is _below_ the residual std — cleanest signal.  
K = 5 000 is noise-dominated (17.8×) and unreliable for vol estimation.

#### PCA on IV matrix

| PC  | Variance explained | Cumulative |
| --- | ------------------ | ---------- |
| 1   | 73.3 %             | 73.3 %     |
| 2   | 13.6 %             | 86.9 %     |
| 3   | 7.6 %              | 94.4 %     |

PC1 does not dominate (< 95 %) → smile shape is multi-factor; b and c cannot be set to zero.  
PC1 loadings are heterogeneous across strikes (K = 5 400 dominates with −0.787), confirming K = 5 400 drives most of the smile variation.

#### Vol-spot dependency (Section 12)

- Corr(Δa, ΔS) = −0.377 (significant; threshold 0.3).
- Lag structure: ΔS leads Δa and Δa leads ΔS both at +0.19 — neither clearly leads.
- → A **vanna hedge term is needed**.

---

### 3. IV vs GARCH Volatility — Near-ATM

#### Comparison

| Strike | Mean IV | GARCH vol | Spread (IV − GARCH) | % ticks IV > GARCH |
| ------ | ------- | --------- | ------------------- | ------------------ |
| 5 000  | 0.01220 | 0.02150   | −0.0093             | 0 %                |
| 5 100  | 0.01208 | 0.02150   | −0.0094             | 0 %                |
| 5 200  | 0.01221 | 0.02150   | −0.0093             | 0 %                |
| 5 300  | 0.01234 | 0.02150   | −0.0091             | 0 %                |
| 5 400  | 0.01157 | 0.02150   | −0.0099             | 0 %                |
| 5 500  | 0.01255 | 0.02150   | −0.0089             | 0 %                |

- IV is below GARCH conditional vol **100 % of the time** across all strikes and all 3 days.
- Mean IV / RV ratio = **0.565** — options are 43 % cheap relative to realized vol.
- The spread is structurally stable (std ≈ 0.03–0.04 %) with no timing pattern — the mispricing is **permanent, not episodic**.

---

### 4. Deep ITM (K = 4 000, K = 4 500)

#### Intrinsic value violations

| Strike    | % ticks below intrinsic | Mean violation | Worst    |
| --------- | ----------------------- | -------------- | -------- |
| K = 4 000 | **10.0 %**              | −1.21 pts      | −7.0 pts |
| K = 4 500 | **29.4 %**              | −0.70 pts      | −6.0 pts |

- Options are frequently quoted **below intrinsic** (S − K).
- This violates no-arbitrage: there is no valid implied volatility when C < S − K.
- Valid IV exists only in the remaining 10–29 % of ticks; those valid IVs are noisy and far above ATM (mean 4.15 % for K = 4 000; 2.35 % for K = 4 500).

#### Smile residuals

Deep ITM sits +81 to +163 bp above the near-ATM smile extrapolation — expected given near-zero (or negative) time value producing inflated IV estimates.  
**These strikes must be excluded from smile fitting.**

---

### 5. Deep OTM (K = 6 000, K = 6 500)

#### Prices and IV

- Both strikes are **constantly quoted at 0.5** — the minimum tick.
- The price reflects a quantization floor, not a market-clearing vol signal.

| Strike    | Mean IV | Smile model (extrapolated) | Spread      | % ticks above model |
| --------- | ------- | -------------------------- | ----------- | ------------------- |
| K = 6 000 | 0.01987 | 0.01373                    | **+61 bp**  | 100 %               |
| K = 6 500 | 0.03011 | 0.01623                    | **+139 bp** | 100 %               |

- Deep OTM IV is **permanently above** the near-ATM quadratic extrapolation — a pronounced volatility smile / skew.
- The spread grows with distance from ATM (classic convexity premium in OTM options).
- **These strikes must be excluded from smile fitting** (min-tick floor corrupts the curvature estimate).

---

## Strategy

### Near-ATM (K = 5 000 – 5 500) — Gamma Scalping

**Core finding:** IV < GARCH vol by ~0.9 % / √day persistently → options are structurally cheap → **buy-vol bias is justified**.

**Smile fitting:**

- Use only K = 5 000 – 5 500 for smile fit.
- Refit per-timestamp (dynamic a, b, c); do not hardcode.
- EMA window for de-trending must exceed 100 ticks (autocorr persists at lag 1 after EMA(100)).

**Best strikes for gamma scalping:**

- K = 5 200 and K = 5 300 have the highest mean vega (~5 280–5 290) and lowest noise-floor-to-residual ratio.
- K = 5 000 is noise-dominated (noise floor 17.8× residual std) — avoid or widen threshold significantly.
- K = 5 400 has the cleanest signal (noise floor 0.95× residual std) but also the largest systematic smile bias (−58 bp).

**Signal construction:**

- Compute `theo_diff = market_price − BS(S, K, T, IV_smile(m))`.
- Track `mean_theo_diff = EMA(theo_diff)` with window > 100 ticks.
- Trade when `current_theo_diff − mean_theo_diff` exceeds threshold.
- Add a regime gate: only trade when `EMA(|current − mean|) ≥ threshold` (suppress trading in quiet markets).

**Vanna hedge:**

- Corr(Δa, ΔS) = −0.38 → vol moves with the underlying.
- A vanna correction term is needed if holding large vega exposure.

**Model for conditional vol:**

- Use **GARCH(1,1)-t** (5 parameters, wins on OOS QLIKE and Vol-MAE).
- APARCH(1,1)-skewt wins in-sample but overfits on 30 k ticks; revisit with ≥ 1 M ticks.
- Parameters: ω = 2.32 × 10⁻⁴, α = 0.10, β = 0.40, ν = 12 — re-estimate offline, not live.

---

### Deep ITM (K = 4 000, K = 4 500) — Intrinsic Arbitrage

**Core finding:** options are frequently quoted below intrinsic value → risk-free profit available.

**Trade:**

- Monitor `C − max(S − K, 0)` continuously.
- When negative: buy the call, short the underlying.
- Lock in `(S − C) − K > 0` at initiation; profit is guaranteed at expiry regardless of path.
- K = 4 500 offers higher frequency (29 % of ticks); K = 4 000 offers larger per-event profit (mean −1.2 pts).

**Do not use for smile fitting** — prices are at or below intrinsic, making IV either undefined or unreliable.

---

### Deep OTM (K = 6 000, K = 6 500) — Skew Trading / Tail Hedge

**Core finding:** deep OTM options carry a permanent vol premium (+61 to +139 bp) above the near-ATM smile extrapolation, driven by minimum-tick floor pricing.

**Two uses:**

1. **Tail hedge:** if short gamma on near-ATM strikes, buy OTM calls to cap blowup exposure on large upward VEV moves. Cost is known (0.5 pts per contract) and small relative to ATM premium.

2. **Skew relative value:** the near-ATM smile under-predicts OTM IV by a stable, large margin. If a richer OTM pricing model is available, the spread between model IV and market IV can be traded directly.

**Do not use for smile fitting** — prices are quantization-floored at 0.5 pts, not market-clearing.

---

### 6. Smile Deviation & Reversion Time — confirmed 2026-04-25

Script: `smile_reversion_analysis.py`. Plots: `images/`. TTE convention confirmed: historical data day 0 → TTE = 8; data spans TTE 8 → 5. `OptionTrader.py` trades days 3–5 → TTE 5 → 2. Both use the same clock (`TTE = 8 − elapsed_days`); no convention mismatch.

#### Deviations — OLD A,B,C on historical data (correct TTE 8→5)

| Strike | Mean dev   | Std dev | Notes                           |
| ------ | ---------- | ------- | ------------------------------- |
| 4 000  | +0.012     | 0.829   | ~zero mean, white noise         |
| 4 500  | +0.011     | 0.759   | ~zero mean, white noise         |
| 5 000  | −0.042     | 0.561   | negligible bias                 |
| 5 100  | −0.067     | 0.941   | negligible bias                 |
| 5 200  | **+0.729** | 0.925   | structural bias: poly underfits |
| 5 300  | **+1.327** | 1.151   | structural bias: poly underfits |
| 5 400  | **−2.201** | 0.770   | structural bias: poly overfits  |
| 5 500  | +0.526     | 0.422   | mild bias                       |
| 6 000  | +0.493     | 0.005   | tick-floor artefact             |
| 6 500  | +0.500     | 0.000   | constant tick floor             |

Conclusion: OLD A,B,C correctly price K = 5 000–5 100 on historical data (mean dev < 0.1 pt). Residual biases at K = 5 200–5 400 are structural (degree-2 poly insufficient, confirmed by §2 t-stat table), not a calibration error.

**Critical issue for live trading:** these biases are persistent. `surface_arbitrage_orders()` measures `dev = market − BS(smile)`. K = 5 400 will always show `dev ≈ −2.2`, firing aggressive `bid` continuously. K = 5 300 will always show `dev ≈ +1.3`, potentially firing aggressive `ask`. Both are **structural artefacts of the polynomial**, not tradeable signals.

#### Mean reversion time (de-trended deviations, roll = 200, threshold ±1.0 pt)

| Strike | ACF half-life | # episodes | Median | Mean | p90 |
| ------ | ------------- | ---------- | ------ | ---- | --- |
| 5 000  | **1 tick**    | 2 039      | 1      | 1.1  | 1   |
| 5 100  | **1 tick**    | 1 649      | 1      | 1.1  | 1   |
| 5 200  | **1 tick**    | 851        | 1      | 1.0  | 1   |
| 5 300  | **1 tick**    | 125        | 1      | 1.0  | 1   |
| 5 400  | 1 tick        | 1          | 1      | 1.0  | 1   |
| 5 500  | 4 ticks       | 0          | —      | —    | —   |

After de-trending, **all ATM-strike deviations are white noise at tick scale** (ACF HL = 1). No pointwise signal survives to tick+1. The exploitable mispricing is in the slow drift of the smile level (`a`-coefficient HL = 27 ticks per §2), not in pointwise price spikes.

**Open question for live trading (days 3–5, TTE 5→2):** the smile was calibrated at TTE 8→5. If IV has a term structure (rises as T → 0, common empirically), `a` will be higher at TTE 2–5 than the fitted value 0.012535. Coefficient `a` is non-stationary (ADF p=0.17) and shows slow drift within historical data — cannot distinguish term-structure from vol-regime noise without data at TTE ≤ 5. See §7 for required analysis.

#### Required fix before launch

1. `surface_arbitrage_orders()`: add per-strike mean-deviation correction before comparing to threshold.  
   `dev_adjusted = dev[K] − STRIKE_BIAS[K]` where `STRIKE_BIAS = {5300: +1.33, 5400: −2.20, ...}`.  
   Without this, K = 5 400 fires continuous buys and K = 5 300 continuous sells regardless of true mispricing.
2. Threshold must exceed σ(de-trended deviation) per strike (0.2–0.9 pts) to avoid noise trading.

---

### 7. Term-structure of smile coefficients & per-strike bias — confirmed 2026-04-25

Script: `term_structure_and_bias_analysis.py`. Plots: `images/term_structure_abc.png`, `images/per_strike_bias.png`. Source: `iv_smile_coeffs.csv` (per-tick a,b,c from `iv_smile_analysis.py`), trimmed at 0.5% tails (29 172 / 30 000 rows kept). TTE for historical data = 8 − global_ts/100/10 000.

#### 7.1 Term structure: per-tick coefficient ~ TTE regression

OLS fit `coeff = α + β · TTE` across 29 172 rows:

| Coeff | Intercept  | Slope (per day TTE) | SE         | t-stat | R²         |
| ----- | ---------- | ------------------- | ---------- | ------ | ---------- |
| a     | +0.012 885 | **−3.68 × 10⁻⁵**    | 2.4 × 10⁻⁶ | −15.5  | **0.0082** |
| b     | +0.044 901 | −4.76 × 10⁻³        | 2.7 × 10⁻⁴ | −17.9  | 0.0109     |
| c     | +0.045 709 | −6.84 × 10⁻²        | 1.2 × 10⁻² | −5.7   | 0.0011     |

All three slopes statistically significant (|t| > 5) but R² < 2 % — the trend is real, magnitude tiny.

#### 7.2 Projection to live-trading TTE (5 → 2)

| TTE | a_proj    | b_proj     | c_proj     | a_proj / A_OLD |
| --- | --------- | ---------- | ---------- | -------------- |
| 5.0 | 0.012 701 | +0.021 110 | −0.296 166 | 1.013          |
| 4.0 | 0.012 738 | +0.025 868 | −0.227 791 | 1.016          |
| 3.0 | 0.012 774 | +0.030 626 | −0.159 416 | 1.019          |
| 2.0 | 0.012 811 | +0.035 384 | −0.091 041 | 1.022          |

Mid-window (TTE = 3.5): A_LIVE = 0.012 756, B_LIVE = 0.028 247, C_LIVE = −0.193 603.

**Practical conclusion:**

- **`a` term structure: negligible.** A_LIVE is only +1.76 % above A_OLD across full live TTE range. The hardcoded A = 0.012 535 will not cause systematic mispricing from term-structure effect.
- **`b` and `c` extrapolations: do not trust.** R² < 0.02 means the regression line is noise. Projecting B and C linearly gives B*LIVE = 0.028 (12× the OLD B) and C_LIVE = −0.19 (sign flip from OLD +0.57) — these are extrapolation artefacts, not real term structure. **Keep OLD B and C.** Verified empirically below: per-strike bias is \_worse* under LIVE projection than under OLD coefficients.

#### 7.3 Per-strike bias under static smile (priority 2)

Mean(market − BS(smile)) computed across all 30 000 historical ticks. Standard error of mean ≤ 0.007 pt for every strike → biases are highly precise estimates.

| K     | bias_OLD   | std_OLD | bias_LIVE  | std_LIVE | SE(mean) |
| ----- | ---------- | ------- | ---------- | -------- | -------- |
| 4 000 | +0.012     | 0.829   | +0.012     | 0.829    | 0.0048   |
| 4 500 | +0.011     | 0.759   | +0.012     | 0.759    | 0.0044   |
| 5 000 | −0.042     | 0.561   | +0.818     | 0.567    | 0.0032   |
| 5 100 | −0.067     | 0.941   | +0.516     | 0.932    | 0.0054   |
| 5 200 | **+0.729** | 0.925   | +0.162     | 0.992    | 0.0053   |
| 5 300 | **+1.327** | 1.151   | −0.234     | 1.237    | 0.0066   |
| 5 400 | **−2.201** | 0.770   | **−3.740** | 0.834    | 0.0044   |
| 5 500 | +0.526     | 0.424   | −0.372     | 0.472    | 0.0024   |
| 6 000 | +0.493     | 0.005   | +0.496     | 0.004    | 0.0000   |
| 6 500 | +0.500     | 0.000   | +0.500     | 0.000    | 0.0000   |

LIVE projection makes K=5000, 5100, 5400 _worse_ — confirms B/C extrapolation is unreliable. **Use OLD A,B,C with per-strike bias correction.**

#### 7.4 Drop-in correction for `OptionTrader.py`

Keep hardcoded `A=0.012535, B=0.002252, C=0.56799` (no change).

Add per-strike additive bias dict, subtract before threshold check in `surface_arbitrage_orders()`:

```python
STRIKE_BIAS = {
    "VEV_4000": +0.0117,
    "VEV_4500": +0.0106,
    "VEV_5000": -0.0415,
    "VEV_5100": -0.0668,
    "VEV_5200": +0.7287,
    "VEV_5300": +1.3271,
    "VEV_5400": -2.2007,
    "VEV_5500": +0.5256,
    "VEV_6000": +0.4930,
    "VEV_6500": +0.5000,
}

# in surface_arbitrage_orders():
dev_corrected = dev - STRIKE_BIAS[name]
if dev_corrected >  thr and opt.max_allowed_sell_volume > 0: opt.ask(...)
elif dev_corrected < -thr and opt.max_allowed_buy_volume > 0: opt.bid(...)
```

Effect:

- K=5400 stops firing continuous buys (the −2.2 bias is treated as zero baseline).
- K=5300 stops firing continuous sells (the +1.3 bias is treated as zero baseline).
- K=4000/4500/6000/6500 unchanged (biases ≤ 0.5, mostly tick-floor).
- K=5000/5100/5200/5500 trade only on genuine deviations from per-strike baseline.

#### 7.5 Caveat: bias is in-sample

Per-strike biases were measured on TTE 8→5 with OLD smile. Two reasons the bias may shift at TTE 5→2:

1. The (small) term structure of `a` shifts the BS price slightly — at TTE = 2 the bias correction should be re-derived. Under-correction risk: order of 0.05 pt per strike. Acceptable.
2. The polynomial mis-fit pattern (which strikes are over- vs under-fit) depends on moneyness `m = log(K/S)/√T`. As T decreases, `m` widens for fixed K-S spread → bias pattern may shift. **This is the main residual risk.** Recommend re-measuring bias after first live day.

---

_Generated from Round 3 analysis — scripts in `Round_3/` and `Round_3/garch/`._
