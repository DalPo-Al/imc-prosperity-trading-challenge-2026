# Support

## Purpose

Per-instrument order-book microstructure snapshots. Each PNG is a 4-panel diagnostic of one instrument's tick series, used to read price behaviour, spread regime, and order-flow signals before strategy design.

## Panels

Every image has the same four stacked panels (x-axis = tick index):

- **Mid price** — top-of-book mid over time.
- **Spread** — best-ask minus best-bid.
- **Top level imbalance** — `(bid_vol - ask_vol) / (bid_vol + ask_vol)` at L1; sign = pressure direction.
- **Microprice - Mid** — imbalance-weighted price minus mid; short-horizon fair-value skew.

## File Reference

| File                       | Instrument           | Notes                                                                                    |
| -------------------------- | -------------------- | ---------------------------------------------------------------------------------------- |
| `ASH_COATED_OSMIUM.png`    | ASH_COATED_OSMIUM    | Mid mean-reverts ~10000; spread mostly 5–21; imbalance and microprice noisy, symmetric.  |
| `INTARIAN_PEPPER_ROOT.png` | INTARIAN_PEPPER_ROOT | Two regimes seen: strong linear uptrend (10000→13000) and a mean-reverting window ~5000. |
| `EMERALDS.png`             | EMERALDS             | Microstructure diagnostic, same 4-panel layout.                                          |
| `TOMATOES.png`             | TOMATOES             | Microstructure diagnostic, same 4-panel layout.                                          |

## Replication

Plots produced by upstream analysis notebook from raw order-book tick data (not in this directory). Regenerate via `matplotlib` with bid/ask price+volume per tick.

## Author

Giorgio Cottini.
