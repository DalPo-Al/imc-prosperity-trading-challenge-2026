# Round 2 - Growing Your Outpost

## Overview

This folder contains the implementation of refined algorithmic trading strategies developed for Round 2 of the IMC Prosperity Trading Challenge.

This round represents the final opportunity to reach the qualifying threshold of a net PnL of **200,000 XIRECs or more** before the leaderboard reset for Phase 2.

Market activity has increased significantly compared to Round 1, introducing additional complexity in both execution and strategy optimization.

## Market Environment

The same two assets from Round 1 are traded:

- **ASH_COATED_OSMIUM**  
  A volatile asset with exploitable short-term inefficiencies and structural patterns.

- **INTARIAN_PEPPER_ROOT**  
  A low-volatility, mean-reverting asset with stable price dynamics.

Position limits remain unchanged:
- ASH_COATED_OSMIUM: 80  
- INTARIAN_PEPPER_ROOT: 80  

## New Mechanism: Market Access Fee (MAF)

Round 2 introduces a **Market Access Fee (MAF)** mechanism that determines access to additional order book liquidity.

Participants may submit a bid for increased market access:
- Top 50% of bids are accepted
- Accepted bids grant **+25% additional market access**
- The bid amount is subtracted from final Round 2 PnL for accepted participants only

The bidding mechanism is implemented through a `bid()` function in the trading class.

## Strategy Enhancements

Compared to Round 1, this iteration introduces:

- Improved signal refinement based on Round 1 performance analysis
- Adjusted execution logic to account for increased market depth
- Inventory-aware position management under tighter competition
- Integration of bid strategy for optimal trade-off between cost and market access

## Market Access Considerations

The bidding mechanism introduces a game-theoretic trade-off:

- Higher bids increase probability of obtaining additional liquidity
- Lower bids reduce cost but risk exclusion from expanded market access
- Optimal bidding depends on expected marginal value of additional flow

## Repository Structure

- `round_2/strategies/` → improved trading logic and new bidding strategy
- `round_2/utils/` → updated helper functions and execution utilities
- `round_2/notebooks/` → performance analysis of Round 1 and strategy improvements
- `round_2/backtesting/` → simulation and validation framework

## Key Assumptions

- Market inefficiencies persist but are less stable than Round 1
- Additional market access provides measurable but non-linear value
- Strategy performance is sensitive to execution constraints and liquidity depth

## Notes

This round emphasizes:
- adaptation to evolving market conditions
- optimization under strategic constraints (MAF mechanism)
- balancing execution quality and cost efficiency

---
