# Round 3 - Gloves Off

## Overview

This folder contains the implementation of algorithmic trading strategies developed for Round 3 of the IMC Prosperity Trading Challenge, marking the start of the **Great Orbital Ascension Trials (GOAT)** phase.

Unlike previous rounds, the leaderboard is reset, and all teams begin with zero PnL. This round introduces a more complex multi-asset environment including derivatives and structured payoff instruments.

## Market Environment

Trading is conducted on three asset classes:

### 1. Underlying Assets (Delta-1 Instruments)

- **HYDROGEL_PACK**
- **VELVETFRUIT_EXTRACT**

These assets behave as standard spot instruments with continuous price dynamics.

### 2. Derivative Instruments (Options)

- **VELVETFRUIT_EXTRACT_VOUCHER (VEV_4000 → VEV_6500)**

These are call-style options on VELVETFRUIT_EXTRACT with varying strike prices and a fixed expiry structure.

Key characteristics:
- 7-day total time to expiry at the start of Round 1
- 1-day decay per round
- independent trading from the underlying asset
- position limit: 300 per voucher

### 3. Manual Trading Opportunity

- **ORNAMENTAL_BIO_PODS**
- Two manual limit orders can be submitted to interact with market participants
- Acquired inventory is automatically liquidated into final PnL

## Strategy Overview

This round extends previous market-making and signal-based frameworks into a multi-layered system:

### Underlying Assets
- continuation of mean-reversion and short-term signal strategies
- refined execution logic based on prior round performance

### Options (Vouchers)
- relative value analysis between strike prices and underlying price
- implied volatility intuition and mispricing detection
- structured view of payoff asymmetry across strikes and time-to-expiry

### Execution Layer
- position-aware risk control across correlated instruments
- improved inventory balancing under tighter constraints
- separation of spot vs derivative risk exposure

## Repository Structure

- `round_3/strategies/` → trading logic for spot and option instruments
- `round_3/options/` → voucher pricing logic and analysis
- `round_3/utils/` → shared utilities for pricing, risk, and execution
- `round_3/notebooks/` → exploratory analysis and strategy validation
- `round_3/backtesting/` → simulation environment

## Key Assumptions

- underlying assets retain short-term exploitable inefficiencies
- option prices reflect incomplete or noisy implied valuation
- cross-strike relationships contain relative value opportunities
- time decay is a critical driver of voucher dynamics

## Notes

This round introduces a shift in complexity:
- from single-market trading → multi-asset portfolio reasoning
- from linear strategies → structured payoff interpretation
- from execution-only thinking → pricing-aware decision making

---
