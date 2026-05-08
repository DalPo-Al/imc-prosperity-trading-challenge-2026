
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

# ── Instruments ───────────────────────────────────────────────────────────────
VELVETFRUIT = "VELVETFRUIT_EXTRACT"
OPT_STRIKES  = [5000, 5100, 5200]
OPT_SYMBOLS  = [f"VEV_{k}" for k in OPT_STRIKES]

POSITION_LIMITS = {VELVETFRUIT: 200, **{s: 300 for s in OPT_SYMBOLS}}

# ── Fair values: OOS mean wall price (days 0+1) ───────────────────────────────
FAIR_VALUE = {
    "VEV_5000": 253.26,
    "VEV_5100": 166.55,
    "VEV_5200":  96.31,
}

# ── Strategy knobs ────────────────────────────────────────────────────────────
PRICE_THR_OPEN  = 2.0   # min edge in price pts to open  [hyperopt: 0.3 – 8.0]
PRICE_THR_CLOSE = 0.5   # close when residual drops here [hyperopt: 0.0 – 4.0, < OPEN]
MAX_TRADE_SIZE  = 20    # max new units per tick          [hyperopt: 5 – 50]
SPOT_JUMP_THR   = 10.0  # skip tick if ΔS > this         [hyperopt: 3.0 – 30.0]


# ── State management ──────────────────────────────────────────────────────────
class StateManager:
    def __init__(self, state: TradingState):
        try:
            self._data: dict = json.loads(state.traderData) if state.traderData else {}
        except:
            self._data = {}
        self._new: dict = {}

    def get(self, key, default=None):
        return self._data.get(key, default)

    def set(self, key, value):
        self._new[key] = value

    def dump(self) -> str:
        return json.dumps(self._new)


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
        self.mid = (
            (self.best_bid + self.best_ask) / 2
            if self.best_bid is not None and self.best_ask is not None
            else None
        )
        wb = self._vwap(self.mkt_buy_orders)
        wa = self._vwap(self.mkt_sell_orders)
        self.wall = (wb + wa) / 2 if wb is not None and wa is not None else self.mid

    def _parse_book(self, state):
        buy_orders = sell_orders = {}
        try:
            od: OrderDepth = state.order_depths[self.name]
            buy_orders = {
                p: abs(v) for p, v in sorted(od.buy_orders.items(), reverse=True)
            }
            sell_orders = {p: abs(v) for p, v in sorted(od.sell_orders.items())}
        except:
            pass
        return buy_orders, sell_orders

    def _best_prices(self):
        return (
            max(self.mkt_buy_orders) if self.mkt_buy_orders else None,
            min(self.mkt_sell_orders) if self.mkt_sell_orders else None,
        )

    @staticmethod
    def _vwap(levels: dict) -> float | None:
        if not levels:
            return None
        total_vol = sum(levels.values())
        if total_vol == 0:
            return None
        return sum(p * v for p, v in levels.items()) / total_vol

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


# ── Option trading logic ──────────────────────────────────────────────────────
class OptionTrader:
    def __init__(self, state: TradingState, sm: StateManager):
        self.sm = sm
        self.underlying = ProductTrader(VELVETFRUIT, state)
        self.options = {f"VEV_{k}": ProductTrader(f"VEV_{k}", state) for k in OPT_STRIKES}

    def _trade(self, opt: ProductTrader, name: str) -> None:
        if opt.best_bid is None or opt.best_ask is None or opt.wall is None:
            return
        fair   = FAIR_VALUE[name]
        sell_sig = opt.best_bid - fair   # > 0: bid above fair → option rich → sell
        buy_sig  = opt.best_ask - fair   # < 0: ask below fair → option cheap → buy
        if sell_sig > PRICE_THR_OPEN:
            avail = opt.mkt_buy_orders.get(opt.best_bid, 0)
            close = max(opt.initial_position, 0)
            size  = min(opt.max_allowed_sell_volume,
                        close + min(MAX_TRADE_SIZE, max(avail - close, 0)), avail)
            opt.ask(opt.best_bid, size)
        elif buy_sig < PRICE_THR_CLOSE and opt.initial_position < 0:
            opt.bid(opt.best_ask, -opt.initial_position)
        elif buy_sig < -PRICE_THR_OPEN:
            avail = opt.mkt_sell_orders.get(opt.best_ask, 0)
            close = max(-opt.initial_position, 0)
            size  = min(opt.max_allowed_buy_volume,
                        close + min(MAX_TRADE_SIZE, max(avail - close, 0)), avail)
            opt.bid(opt.best_ask, size)
        elif sell_sig > -PRICE_THR_CLOSE and opt.initial_position > 0:
            opt.ask(opt.best_bid, opt.initial_position)

    def get_orders(self) -> dict:
        S = self.underlying.wall
        if S is None:
            return {}
        S_prev = self.sm.get("S_prev", S)
        self.sm.set("S_prev", S)
        if abs(S - S_prev) > SPOT_JUMP_THR:
            return {}
        out = {}
        for name in OPT_SYMBOLS:
            self._trade(self.options[name], name)
            out.update(self.options[name].get_orders())
        return out


class Trader:
    def run(self, state: TradingState):
        sm = StateManager(state)
        trader = OptionTrader(state, sm)
        result = trader.get_orders()
        trader_data = sm.dump()
        logger.flush(state, result, 0, trader_data)
        return result, 0, trader_data
