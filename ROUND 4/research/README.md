# Research

## Competition Structure

The two option rounds contained an important structural feature: the trading horizon of the competition was shorter than the full contractual life of the available option instruments.

That distinction mattered.

A standard option analysis usually focuses on the terminal payoff at expiry. In this setting, however, the relevant objective was not the option's final settlement value, but the mark-to-market P&L realised within the finite evaluation window of the challenge. We therefore decided to treat the round as a finite-horizon trading problem rather than a full-maturity option pricing problem.

The key insight was that some positions had a substantially different risk profile when evaluated over the competition horizon than they would have had at maturity. In particular, the long-term obligations embedded in certain option positions were not fully realised inside the round's timeframe, while their short-horizon mark-to-market behaviour still contributed directly to the team's score.

This created a timing asymmetry between:

- the contractual maturity of the option,
- the period over which positions affected the leaderboard,
- and the point at which P&L was effectively crystallised.

## Key Insight

The strategy was built around this boundary condition. Rather than interpreting the opportunity as a conventional market inefficiency, we treated it as a consequence of the game's accounting structure: the option's full maturity existed in the contract specification, but the relevant risk window was truncated by the challenge itself.

From this perspective, the trade was not based on predicting the terminal payoff of the option. It was based on recognising that, within the finite horizon of the simulation, the position's effective exposure was governed by short-term mark-to-market dynamics rather than by the full expiry payoff.

This made the round especially interesting because the main source of edge did not come from a more accurate volatility estimate, a better directional forecast, or a faster execution model. It came from correctly identifying the boundary condition imposed by the rules of the environment and adapting the pricing logic accordingly.

In practice, the strategy exploited the mismatch between the payoff window and the evaluation window. The competition ended before the full long-term risk of the option position could materialise, while the corresponding P&L impact was already reflected during the scoring period.

## Favourable Conditions

An additional lucky condition was that the price of the underlying always oscillated between two fixed extremes, and at the beginning of round 4, it was very close to the upper one. This fact amplified the P&L of selling deep-ITM options at round start.

## Authors

There was no leading actor in the creation of this strategy; some of us independently and accidentally stumbled into strategies that fixed selling positions on deep-ITM options.
In a joint effort to abandon this method, we failed to find a strategy with a better risk/reward ratio and ended up accepting this risk.
