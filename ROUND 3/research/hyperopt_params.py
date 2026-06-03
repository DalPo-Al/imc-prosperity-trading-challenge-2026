"""
Hyperparameter search space for OptionTrader.py.

All values are discrete. The search program will receive this file
and iterate over combinations (or sample them).

Parameter rationale:
  SURFACE_ARB_MULTIPLIER — controls edge threshold; bias residual std ~0.5 pts,
                            so values below 1.0 may overtrade noise.
  MAX_TRADE_SIZE         — caps per-tick position delta; prevents instant blowup
                            if bias calibration is off.
  STRIKE_BIAS per strike — TTE 5→6 calibration has std 0.4–0.9 pts; search ±1 pt
                            around calibrated values in 0.25 steps.
  INTRINSIC_BUFFER       — 0 = pure no-arb; 0.5 = current; 1.0 = conservative.
"""

from itertools import product as iterproduct

# ── Primary strategy knobs ────────────────────────────────────────────────────

SURFACE_ARB_MULTIPLIER_VALUES = [0.5, 1.0, 1.5, 2.0, 3.0]

MAX_TRADE_SIZE_VALUES = [10, 20, 30, 50, 100]

INTRINSIC_BUFFER_VALUES = [0.0, 0.25, 0.5, 1.0]

# ── Per-strike STRIKE_BIAS search ─────────────────────────────────────────────
# Calibrated centres (TTE 5→6 window). Search ±1.0 in 0.25 steps.

import numpy as np

def _range(centre, half=1.0, step=0.25):
    vals = np.arange(centre - half, centre + half + step/2, step)
    return [round(float(v), 4) for v in vals]

STRIKE_BIAS_VALUES = {
    "VEV_5000": _range(+0.0316),   # centre ~0, market near fair
    "VEV_5100": _range(-0.6400),   # model overprices K=5100 at live TTE
    "VEV_5200": _range(+0.7654),   # model underprices K=5200
    "VEV_5300": _range(+1.9245),   # largest positive bias, most active strike
    "VEV_5400": _range(-1.9138),   # large negative bias
    "VEV_5500": _range(+0.6151),
}

# ── Summary ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    n_primary = (len(SURFACE_ARB_MULTIPLIER_VALUES)
                 * len(MAX_TRADE_SIZE_VALUES)
                 * len(INTRINSIC_BUFFER_VALUES))
    n_bias_per_strike = [len(v) for v in STRIKE_BIAS_VALUES.values()]
    n_bias_total = 1
    for n in n_bias_per_strike: n_bias_total *= n

    print("=== Hyperopt search space ===")
    print(f"SURFACE_ARB_MULTIPLIER : {SURFACE_ARB_MULTIPLIER_VALUES}")
    print(f"MAX_TRADE_SIZE         : {MAX_TRADE_SIZE_VALUES}")
    print(f"INTRINSIC_BUFFER       : {INTRINSIC_BUFFER_VALUES}")
    print()
    for k, v in STRIKE_BIAS_VALUES.items():
        print(f"STRIKE_BIAS[{k}] : {v}")
    print()
    print(f"Primary knob combos : {n_primary}")
    print(f"Bias combos         : {n_bias_total:,}")
    print(f"Full grid size      : {n_primary * n_bias_total:,}")
    print()
    print("Recommendation: fix MAX_TRADE_SIZE=30, sweep SURFACE_ARB_MULTIPLIER")
    print("and per-strike STRIKE_BIAS independently first (sequential search).")
    print("Full grid is too large for exhaustive search — use random/Bayesian.")
