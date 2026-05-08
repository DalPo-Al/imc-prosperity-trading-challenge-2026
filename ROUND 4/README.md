# Round 4 - The More The Merrier

## Overview

This folder contains the implementation of algorithmic trading strategies developed for Round 4 of the IMC Prosperity Trading Challenge, part of the Great Orbital Ascension Trials (GOAT).

This round introduces a new dimension: **counterparty visibility**, allowing identification of market participants through trade data. This enables behavioral analysis of other traders and opens the door to strategy refinement based on participant-specific patterns.

In addition to algorithmic trading, a manual trading opportunity is available in **Aether Crystal** and associated exotic option contracts.

## Market Environment

The algorithmic trading universe remains unchanged from Round 3:

### Spot Assets
- **HYDROGEL_PACK**
- **VELVETFRUIT_EXTRACT**

### Derivatives
- **VELVETFRUIT_EXTRACT_VOUCHER (VEV_4000 → VEV_6500)**

These option instruments retain their structure, with time decay continuing from prior rounds (TTE decreases by one day per round).

Position limits remain:
- HYDROGEL_PACK: 200  
- VELVETFRUIT_EXTRACT: 200  
- VELVETFRUIT_EXTRACT_VOUCHER: 300 per strike  

## New Feature: Counterparty Information

For the first time, trade data includes identifiable counterparties:

- `buyer`: participant executing the buy side
- `seller`: participant executing the sell side

This allows:
- analysis of trading behavior per participant
- detection of recurring liquidity providers or aggressive takers
- potential strategy adaptation based on counterparty profiles

## Strategy Enhancements

This round extends previous frameworks with:

### 1. Behavioral Analysis Layer
- extraction of counterparty trade patterns
- identification of consistent market behavior across participants
- classification of liquidity vs aggressive trading profiles

### 2. Adaptive Execution Logic
- optional adjustment of execution based on observed counterparties
- refinement of entry/exit timing using behavioral signals

### 3. Continued Multi-Asset Trading
- spot strategies (mean reversion + microstructure signals)
- option strategies (relative value across strikes + time decay dynamics)

## External Trading Opportunity

In addition to algorithmic trading, manual participation is available in:
- **Aether Crystal**
- related exotic option instruments

These are treated independently from the algorithmic portfolio and require separate decision-making.

## Repository Structure

- `round_4/strategies/` → core algorithmic trading logic
- `round_4/counterparty_analysis/` → behavioral and participant analysis
- `round_4/options/` → voucher and derivative modeling
- `round_4/utils/` → shared utilities (pricing, execution, risk)
- `round_4/notebooks/` → exploratory and analytical work
- `round_4/backtesting/` → simulation framework

## Key Assumptions

- market microstructure patterns persist across participants
- counterparty behavior is partially stable and classifiable
- option dynamics remain driven by underlying + time decay
- informational advantage can be extracted from trade-level visibility

## Notes

This round represents a transition from purely price-based modeling to **participant-aware trading systems**.

Key emphasis areas:
- behavioral signal extraction
- adaptive execution logic
- integration of market microstructure and counterparty data

---
