from datamodel import (
    Listing,
    Observation,
    Order,
    OrderDepth,
    ProsperityEncoder,
    Symbol,
    Trade,
    TradingState,
)
import json
from typing import Any


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(
        self,
        state: TradingState,
        orders: dict[Symbol, list[Order]],
        conversions: int,
        trader_data: str,
    ) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )
        max_item_length = (self.max_log_length - base_length) // 3
        print(
            self.to_json(
                [
                    self.compress_state(
                        state, self.truncate(state.traderData, max_item_length)
                    ),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        return [[l.symbol, l.product, l.denomination] for l in listings.values()]

    def compress_order_depths(
        self, order_depths: dict[Symbol, OrderDepth]
    ) -> dict[Symbol, list[Any]]:
        return {s: [od.buy_orders, od.sell_orders] for s, od in order_depths.items()}

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        return [
            [t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
            for arr in trades.values()
            for t in arr
        ]

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
            ]
        return [getattr(observations, "plainObservations", {}), conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        return [[o.symbol, o.price, o.quantity] for arr in orders.values() for o in arr]

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        return value if len(value) <= max_length else value[: max_length - 3] + "..."


logger = Logger()


# ── Instruments & Limits ──────────────────────────────────────────────────────
VELVETFRUIT = "VELVETFRUIT_EXTRACT"
HYDROGEL_PACK = "HYDROGEL_PACK"
OPT_STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
OPT_SYMBOLS = [f"VEV_{k}" for k in OPT_STRIKES]
PRODUCTS = [VELVETFRUIT, HYDROGEL_PACK]

POSITION_LIMITS = {
    VELVETFRUIT: 200,
    HYDROGEL_PACK: 200,
    **{s: 300 for s in OPT_SYMBOLS},
}

# ── Strategy Parameters ───────────────────────────────────────────────────────
VEV_THRESH = 1.0
HYDRO_THRESH = 1.0 
STRATEGY_PARAMS = {
    VELVETFRUIT: {"mean": 5250.0, "std": 20.0, "z_mid_penny_threshold": VEV_THRESH, "z_score_open_threshold": VEV_THRESH-0.1},
    # HYDROGEL_PACK: {"mean": 9995.0, "std": 34.6}
    HYDROGEL_PACK: {"mean": 10_000, "std": 35, "z_mid_penny_threshold": HYDRO_THRESH, "z_score_open_threshold": HYDRO_THRESH-0.1},
}

Z_SCORE_MAX_THRESHOLD = 4.0  # Z-score for max trade size
Z_SCORE_CLOSE_THRESHOLD = 0.5  # Z-score to close the entire position
MIN_TRADE_SIZE = 10  # Minimum number of units to trade in a single order
MAX_TRADE_SIZE = 20  # Maximum number of units to trade in a single order #in practice no trades are being offered at this volume lol
SCALING_EXPONENT = 2.0  # Exponent for volume scaling. >1 for exponential, 1 for linear.


# ── Per-product order helper ──────────────────────────────────────────────────
class ProductTrader:
    def __init__(self, name: str, state: TradingState):
        self.name = name
        self.orders = []
        self.position_limit = POSITION_LIMITS.get(name, 0)
        self.initial_position = state.position.get(name, 0)
        self.max_allowed_buy_volume = self.position_limit - self.initial_position
        self.max_allowed_sell_volume = self.position_limit + self.initial_position
        self.mkt_buy_orders, self.mkt_sell_orders = self._parse_book(state)
        self.best_bid, self.best_ask = self._best_prices()
        self.mid_price = (
            (self.best_bid + self.best_ask) / 2
            if self.best_bid is not None and self.best_ask is not None
            else None
        )

    def _parse_book(self, state):
        if self.name not in state.order_depths:
            return {}, {}
        od: OrderDepth = state.order_depths[self.name]
        buy_orders = {p: abs(v) for p, v in sorted(od.buy_orders.items(), reverse=True)}
        sell_orders = {p: abs(v) for p, v in sorted(od.sell_orders.items())}
        return buy_orders, sell_orders

    def _best_prices(self):
        return (
            max(self.mkt_buy_orders) if self.mkt_buy_orders else None,
            min(self.mkt_sell_orders) if self.mkt_sell_orders else None,
        )

    def bid(self, price, volume):
        vol = min(abs(int(volume)), self.max_allowed_buy_volume)
        if vol > 0:
            self.max_allowed_buy_volume -= vol
            self.orders.append(Order(self.name, int(price), vol))

    def ask(self, price, volume):
        vol = min(abs(int(volume)), self.max_allowed_sell_volume)
        if vol > 0:
            self.max_allowed_sell_volume -= vol
            self.orders.append(Order(self.name, int(price), -vol))

    def get_orders(self):
        return {self.name: self.orders}


# ── Mean Reversion Logic ──────────────────────────────────────────────────────
class MeanReversionTrader:
    def __init__(self, state: TradingState, product_name: str):
        self.product = ProductTrader(product_name, state)
        self.params = STRATEGY_PARAMS[product_name]
        self.is_velvet = product_name == VELVETFRUIT
        if self.is_velvet:
            self.options = [ProductTrader(s, state) for s in OPT_SYMBOLS]

    def _trade(self):
        # We need both bid and ask to make decisions
        if self.product.best_bid is None or self.product.best_ask is None:
            return

        # If the price is close to the mean, switch to market making.
        if self.product.mid_price is not None:
            z_mid = (self.product.mid_price - self.params["mean"]) / self.params["std"]
            penny_threshold = self.params["z_mid_penny_threshold"]
            # When z-score is small (price is near the mean), penny the spread.
            if abs(z_mid) < penny_threshold:
                bid_price = self.product.best_bid + 1
                ask_price = self.product.best_ask - 1
                # Ensure our quotes don't cross the spread
                if bid_price < ask_price:
                    self.product.bid(bid_price, MAX_TRADE_SIZE)
                    self.product.ask(ask_price, MAX_TRADE_SIZE)
                return  # We've placed market making orders, so we are done for this tick.

        z_bid = (self.product.best_bid - self.params["mean"]) / self.params["std"]
        z_ask = (self.product.best_ask - self.params["mean"]) / self.params["std"]
        pos = self.product.initial_position

        open_threshold = self.params["z_score_open_threshold"]

        # --- Opening logic: Open short if bid price is high ---
        if z_bid > open_threshold:
            # Price is high -> Sell. The z-score to use for scaling is z_bid.
            z_abs = z_bid
            # Scale trade size based on how far z-score is above the open threshold
            linear_scaling_factor = min(
                (z_abs - open_threshold) / (Z_SCORE_MAX_THRESHOLD - open_threshold),
                1.0,
            )
            exponential_scaling_factor = linear_scaling_factor**SCALING_EXPONENT
            trade_size = int(
                MIN_TRADE_SIZE + exponential_scaling_factor * (MAX_TRADE_SIZE - MIN_TRADE_SIZE)
            )
            if self.is_velvet:
                # Trade the underlying VELVETFRUIT
                vol_to_sell_underlying = min(self.product.max_allowed_sell_volume, trade_size)
                if vol_to_sell_underlying > 0:
                    self.product.ask(self.product.best_bid, vol_to_sell_underlying)

                # Trade each option individually
                for opt in self.options:
                    if opt.best_bid is not None and opt.best_bid != 0: # Check if there's a non-zero bid to sell into
                        vol_available_to_sell_option = opt.max_allowed_sell_volume
                        vol_offered_in_market = opt.mkt_buy_orders.get(opt.best_bid, 0)
                        final_trade_size_option = min(trade_size, vol_available_to_sell_option, vol_offered_in_market)
                        if final_trade_size_option > 0:
                            opt.ask(opt.best_bid, final_trade_size_option)

            else:
                vol_to_sell = self.product.max_allowed_sell_volume
                final_trade_size = min(vol_to_sell, trade_size)
                if final_trade_size > 0:
                    self.product.ask(self.product.best_bid, final_trade_size)

        # --- Opening logic: Open long if ask price is low ---
        elif z_ask < -open_threshold:
            # Price is low -> Buy. The z-score to use for scaling is abs(z_ask).
            z_abs = abs(z_ask)
            # Scale trade size based on how far z-score is above the open threshold
            linear_scaling_factor = min(
                (z_abs - open_threshold) / (Z_SCORE_MAX_THRESHOLD - open_threshold),
                1.0,
            )
            exponential_scaling_factor = linear_scaling_factor**SCALING_EXPONENT
            trade_size = int(
                MIN_TRADE_SIZE + exponential_scaling_factor * (MAX_TRADE_SIZE - MIN_TRADE_SIZE)
            )
            if self.is_velvet:
                # Trade the underlying VELVETFRUIT
                vol_to_buy_underlying = min(self.product.max_allowed_buy_volume, trade_size)
                if vol_to_buy_underlying > 0:
                    self.product.bid(self.product.best_ask, vol_to_buy_underlying)

                # Trade each option individually
                for opt in self.options:
                    if opt.best_ask is not None and opt.best_bid is not None and opt.best_bid != 0: # Check if there's an ask to buy from and a valid non-zero bid
                        vol_available_to_buy_option = opt.max_allowed_buy_volume
                        vol_offered_in_market = opt.mkt_sell_orders.get(opt.best_ask, 0)
                        final_trade_size_option = min(trade_size, vol_available_to_buy_option, vol_offered_in_market)
                        if final_trade_size_option > 0:
                            opt.bid(opt.best_ask, final_trade_size_option)

            else:
                vol_to_buy = self.product.max_allowed_buy_volume
                final_trade_size = min(vol_to_buy, trade_size)
                if final_trade_size > 0:
                    self.product.bid(self.product.best_ask, final_trade_size)

        # --- Closing logic: close position as price reverts to mean ---
        # elif pos > 0 and z_ask > -Z_SCORE_CLOSE_THRESHOLD:
        #     # We are long, and the buy signal (low z_ask) has disappeared. Close the position.
        #     self.product.ask(self.product.best_bid, pos)
        # elif pos < 0 and z_bid < Z_SCORE_CLOSE_THRESHOLD:
        #     # We are short, and the sell signal (high z_bid) has disappeared. Close the position.
        #     self.product.bid(self.product.best_ask, -pos)

    def get_orders(self) -> dict:
        self._trade()
        orders = self.product.get_orders()
        if self.is_velvet:
            for opt in self.options:
                orders.update(opt.get_orders())
        return orders


class Trader:
    def run(self, state: TradingState):
        result = {}
        for product_name in PRODUCTS:
            if product_name in state.order_depths:
                trader = MeanReversionTrader(state, product_name)
                orders = trader.get_orders()
                result.update(orders)

        # This strategy is stateless, so trader_data is an empty string
        trader_data = ""
        logger.flush(state, result, 0, trader_data)
        return result, 0, trader_data