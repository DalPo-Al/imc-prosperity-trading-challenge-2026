# Round 1 - Trading Groundwork

## Overview

This folder contains the implementation of algorithmic trading strategies developed for Round 1 of the IMC Prosperity Trading Challenge, set on the fictional market of Intara.

The objective of this round is to develop and deploy a trading algorithm capable of generating a net profit of **200,000 XIRECs or more** before the start of the third trading day. In parallel, participants may also trade manually in the Exchange Auction to enhance overall performance.

## Market Environment

Two primary tradable instruments are available:

- **ASH_COATED_OSMIUM**  
  A relatively volatile asset with suspected latent structure and exploitable price patterns.

- **INTARIAN_PEPPER_ROOT**  
  A stable, slow-moving asset with low volatility and predictable behavior.

Both assets have a position limit of **80 units**.

## Strategy Overview

The implemented approach is based on a dual-regime interpretation of the two assets:

- **INTARIAN_PEPPER_ROOT**: treated as a low-volatility mean-reverting instrument, suitable for market making or range-bound strategies.
- **ASH_COATED_OSMIUM**: treated as a higher-volatility asset where statistical patterns and short-term inefficiencies are exploited.

The system combines:
- signal generation based on short-term price dynamics
- inventory-aware execution logic
- risk constraints aligned with position limits

## Repository Structure

- `round_1/strategies/` → core trading logic
- `round_1/utils/` → helper functions for pricing, execution, and risk management
- `round_1/notebooks/` → exploratory analysis and strategy validation
- `round_1/backtesting/` → simulation environment for strategy evaluation

## Key Assumptions

- Market prices are assumed to exhibit short-term inefficiencies that can be exploited.
- Pepper Root behaves as a stable mean-reverting asset.
- Osmium contains exploitable stochastic structure despite apparent randomness.
- Transaction costs and position limits are binding constraints on profitability.

## Performance

The strategy was evaluated in simulation under competition conditions. Results are dependent on market assumptions and parameter calibration.

## Notes

This implementation prioritizes:
- robustness over overfitting
- risk control over aggressive positioning
- interpretability of trading signals

---
