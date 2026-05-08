# Round 5 - The Final Stretch

## Overview

This folder contains the final algorithmic trading system developed for Round 5 of the IMC Prosperity Trading Challenge.

This round represents the final stage of the competition, where the objective is to maximize cumulative PnL across a newly introduced universe of assets while adapting to a complete reset of tradable instruments.

All previous round assets have been removed. The strategy must be fully redefined for a new and significantly broader market structure.

In addition to algorithmic trading, a secondary opportunity is available in the Ignith market, where external informational signals (Ashflow Alpha) may be used to guide discretionary trading decisions.

## Market Environment

### Algorithmic Trading Universe

A total of **50 new tradable instruments** are introduced, grouped into 10 categories of 5 assets each:

- Galaxy Sounds Recorders  
- Vertical Sleeping Pods  
- Organic Microchips  
- Purification Pebbles  
- Domestic Robots  
- UV-Visors  
- Instant Translators  
- Construction Panels  
- Liquid Breath Oxygen Shakes  
- Protein Snack Packs  

Each instrument has a strict position limit of **10 units**.

This round significantly increases the dimensionality of the asset selection problem, requiring prioritization and filtering of exploitable structures.

## Strategy Overview

The trading system is designed around a **selection-first architecture**, where the primary challenge is not execution, but identifying which instruments contain persistent inefficiencies.

### 1. Asset Selection Layer
- identification of instruments with stable statistical structure
- elimination of low-signal or noise-dominated assets
- clustering of similar behavioral patterns within categories

### 2. Strategy Allocation Layer
- assignment of appropriate trading logic per asset class
- mean reversion strategies for range-bound instruments
- momentum or breakout logic for structured trend assets
- avoidance of overfitting across highly heterogeneous products

### 3. Risk and Position Control
- strict adherence to low position limits (±10)
- diversification across categories
- reduced concentration risk due to small inventory constraints

## External Market: Ignith

A parallel trading opportunity exists in the Ignith market, consisting of 9 tradable goods.

Trading decisions may incorporate:
- Ashflow Alpha informational feed
- market sentiment signals
- cross-market structural observations

This component is treated separately from the algorithmic system.

## Repository Structure

- `round_5/selection/` → asset filtering and product ranking logic
- `round_5/strategies/` → trading logic per selected asset group
- `round_5/utils/` → shared utilities for execution and analysis
- `round_5/notebooks/` → exploratory analysis and feature evaluation
- `round_5/backtesting/` → simulation framework for final evaluation
- `round_5/ignith/` → external market trading logic (Ashflow Alpha-based)

## Key Assumptions

- not all assets contain exploitable structure; selection is critical
- cross-sectional filtering improves performance more than per-asset optimization
- strict position limits require high turnover efficiency
- informational edge in Ignith is separate from algorithmic edge in core market

## Notes

This final round shifts the problem from:
- strategy design → **strategy selection**
- execution optimization → **information filtering**
- individual asset modeling → **portfolio of micro-strategies**

The primary challenge is identifying where not to trade.

---
