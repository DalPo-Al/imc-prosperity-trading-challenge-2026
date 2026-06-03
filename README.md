# IMC Prosperity 4 — Alpha Hunters Padua

Quantitative trading strategies and research for the **IMC Prosperity Trading Challenge 2026**, a global algorithmic + manual trading competition run by IMC Trading on a series of fictional markets.

## Results

| Ranking           | Placement        | Field           |
| ----------------- | ---------------- | --------------- |
| Overall           | **208 / 18,803** | all teams       |
| Algorithmic track | **184 / 18,803** | algo-only score |
| Italy             | **2 / 61**       | Italian teams   |

Field: 30,000+ participants, 1,549 universities, 117 countries.

## Format

Five rounds, each opening a new market with its own instruments, position limits, and mechanics. Every round is scored independently on cumulative PnL; phases 2+ reset the leaderboard. Each round combines:

- an **algorithmic** track — a single `Strategy.py` submitted against the IMC `datamodel` simulator (`TradingState` in, `Order` lists out), backtested on provided historical price/trade data;
- a **manual** track — discrete decision problems scored separately.

This repo holds the submitted strategy plus the research that justified it for each round.

## Rounds

| Round | Codename             | Market                                                  | Detail                        |
| ----- | -------------------- | ------------------------------------------------------- | ----------------------------- |
| 1     | Trading Groundwork   | 2 spot assets (ASH_COATED_OSMIUM, INTARIAN_PEPPER_ROOT) | [README](ROUND%201/README.md) |
| 2     | Growing Your Outpost | Same 2 assets + Market Access Fee auction               | [README](ROUND%202/README.md) |
| 3     | Gloves Off           | Spot + VELVETFRUIT_EXTRACT options (VEV_4000–6500)      | [README](ROUND%203/README.md) |
| 4     | The More The Merrier | Same universe + counterparty visibility                 | [README](ROUND%204/README.md) |
| 5     | The Final Stretch    | 50 new assets, 10 categories × 5                        | [README](ROUND%205/README.md) |

Per-round instruments, mechanics, and strategy rationale live in each round's README. Research subfolders (where present) carry the analysis behind the strategy, with their own READMEs.

## Repository structure

```txt
.
├── ROUND 1/          # spot market making + signal exploitation
│   ├── research/     # price-structure analysis (per-asset plots)
│   ├── README.md
│   └── Strategy
├── ROUND 2/          # refined R1 strategy + access-fee bidding
│   ├── README.md
│   └── Strategy.py
├── ROUND 3/          # options: IV smile, term structure, gamma scalping
│   ├── research/     # smile/reversion/deep-strike analysis + findings
│   ├── README.md
│   └── Strategy.py
├── ROUND 4/          # R3 universe + counterparty analysis
│   ├── research/
│   ├── README.md
│   └── Strategy.py
├── ROUND 5/          # 50-asset selection: lead-lag + fingerprinting
│   ├── research/     # cluster lead-lag pipeline + category fingerprints
│   ├── README.md
│   └── Strategy.py
├── README.md
└── LICENSE
```

## Tech stack

Python. Strategies target the IMC Prosperity `datamodel` API (single-file submission, no external deps at runtime). Research uses:

- `numpy`, `pandas` — data handling and numerics
- `scipy` — statistics, optimisation, hierarchical clustering
- `statsmodels` — VAR, ARMA, ACF/PACF, ADF, GARCH-adjacent diagnostics
- `scikit-learn` — scaling, Lasso, SVD
- `matplotlib` — figures

## Per-round results

|   Round |      Final PnL |
| ------: | -------------: |
| Round 1 |  96,632 XERICS |
| Round 2 |  89,294 XERICS |
| Round 3 |  61,613 XERICS |
| Round 4 | 219,004 XERICS |
| Round 5 |  33,264 XERICS |

## Team & contributions

This repository contains work developed collaboratively during the challenge. Contributions were distributed across research, implementation, strategy design, manual trading, and review.

| Team member             | Main contribution                                                                                                    |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------- |
| **Alessio Dal Pozzolo** | Strategy design and experimental implementation, translating theoretical ideas into executable trading decisions.    |
| **Enrico Berto**        | Sole contributor to manual trading, with responsibility for discretionary execution and live strategy review.        |
| **Giorgio Cottini**     | Data analysis and research, signal discovery, market-structure analysis, and parameters optimisation for strategies. |
| **Harish Jawahar**      | Data analysis and research, strategy development and implementation, and review of trading ideas.                    |

## License

See [LICENSE](LICENSE).
