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
import math
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
POSITION_LIMIT = 10

# ── Strategy Parameters ───────────────────────────────────────────────────────
PAIRS_TRADING_PARAMS = {
 0: {'a': 'SLEEP_POD_NYLON',
  'b': 'UV_VISOR_RED',
  'mean': -0.14161123988415622,
  'std': 0.03238476097712599},
 1: {'a': 'PANEL_2X4',
  'b': 'ROBOT_DISHES',
  'mean': 0.11933627948649105,
  'std': 0.03469051344166678},
 2: {'a': 'TRANSLATOR_VOID_BLUE',
  'b': 'UV_VISOR_MAGENTA',
  'mean': -0.023489858080427126,
  'std': 0.03214567088724117},
 3: {'a': 'MICROCHIP_TRIANGLE',
  'b': 'PEBBLES_S',
  'mean': 0.0855054826502178,
  'std': 0.04789726455427217},
 4: {'a': 'MICROCHIP_CIRCLE',
  'b': 'OXYGEN_SHAKE_CHOCOLATE',
  'mean': -0.03676560716277447,
  'std': 0.03463583515040852},
 5: {'a': 'SNACKPACK_CHOCOLATE',
  'b': 'SNACKPACK_PISTACHIO',
  'mean': 0.03739509349757599,
  'std': 0.019317346669806158},
 6: {'a': 'SNACKPACK_RASPBERRY',
  'b': 'SNACKPACK_VANILLA',
  'mean': -0.001417079674296865,
  'std': 0.024269956868988172},
 7: {'a': 'GALAXY_SOUNDS_SOLAR_FLAMES',
  'b': 'SNACKPACK_PISTACHIO',
  'mean': 0.15959720937684468,
  'std': 0.03532715830125765},
 8: {'a': 'OXYGEN_SHAKE_GARLIC',
  'b': 'ROBOT_DISHES',
  'mean': 0.17825998911079427,
  'std': 0.04079822615217669},
 9: {'a': 'GALAXY_SOUNDS_SOLAR_WINDS',
  'b': 'PEBBLES_M',
  'mean': 0.01756834377003127,
  'std': 0.04222925804869073},
 10: {'a': 'PEBBLES_M',
  'b': 'ROBOT_MOPPING',
  'mean': -0.07880577741207187,
  'std': 0.04656452401594608},
 11: {'a': 'OXYGEN_SHAKE_CHOCOLATE',
  'b': 'UV_VISOR_RED',
  'mean': -0.15000565577208708,
  'std': 0.04329687315614372},
 12: {'a': 'SNACKPACK_PISTACHIO',
  'b': 'SNACKPACK_RASPBERRY',
  'mean': -0.06170158199519749,
  'std': 0.029711481477987304},
 13: {'a': 'OXYGEN_SHAKE_CHOCOLATE',
  'b': 'PANEL_2X4',
  'mean': -0.16862876767768448,
  'std': 0.04627772862813418},
 14: {'a': 'PANEL_1X2',
  'b': 'UV_VISOR_RED',
  'mean': -0.22206121061230852,
  'std': 0.053565593499014574},
 15: {'a': 'OXYGEN_SHAKE_EVENING_BREATH',
  'b': 'SLEEP_POD_LAMB_WOOL',
  'mean': -0.14799988908799425,
  'std': 0.04317129010047046},
 16: {'a': 'MICROCHIP_CIRCLE',
  'b': 'PANEL_2X4',
  'mean': -0.205394374840459,
  'std': 0.055207179795655245},
 17: {'a': 'PANEL_1X2',
  'b': 'SLEEP_POD_LAMB_WOOL',
  'mean': -0.188570696163849,
  'std': 0.04530796076621919},
 18: {'a': 'MICROCHIP_CIRCLE',
  'b': 'OXYGEN_SHAKE_GARLIC',
  'mean': -0.2643180844647622,
  'std': 0.062339101100185375},
 19: {'a': 'OXYGEN_SHAKE_CHOCOLATE',
  'b': 'OXYGEN_SHAKE_GARLIC',
  'mean': -0.22755247730198774,
  'std': 0.05061782492361493},
 20: {'a': 'SNACKPACK_CHOCOLATE',
  'b': 'SNACKPACK_RASPBERRY',
  'mean': -0.024306488497621494,
  'std': 0.025873883921800643},
 21: {'a': 'MICROCHIP_RECTANGLE',
  'b': 'OXYGEN_SHAKE_MORNING_BREATH',
  'mean': -0.14126570408734745,
  'std': 0.051507565902519474},
 22: {'a': 'PANEL_2X2',
  'b': 'ROBOT_LAUNDRY',
  'mean': -0.026425591735226135,
  'std': 0.03711189365516561},
 23: {'a': 'GALAXY_SOUNDS_BLACK_HOLES',
  'b': 'ROBOT_DISHES',
  'mean': 0.13750569597652157,
  'std': 0.04202984065103812},
 24: {'a': 'MICROCHIP_CIRCLE',
  'b': 'UV_VISOR_RED',
  'mean': -0.18677126293486157,
  'std': 0.05212388542506224},
 25: {'a': 'OXYGEN_SHAKE_MORNING_BREATH',
  'b': 'ROBOT_VACUUMING',
  'mean': 0.0885538404092828,
  'std': 0.03923883242818696},
 26: {'a': 'ROBOT_DISHES',
  'b': 'UV_VISOR_MAGENTA',
  'mean': -0.10576062223263194,
  'std': 0.03949669279443504},
 27: {'a': 'ROBOT_LAUNDRY',
  'b': 'ROBOT_VACUUMING',
  'mean': 0.07074315343796511,
  'std': 0.038921511836220946},
 28: {'a': 'PANEL_2X4',
  'b': 'SLEEP_POD_NYLON',
  'mean': 0.16023435178975368,
  'std': 0.048755793874386194},
 29: {'a': 'ROBOT_DISHES',
  'b': 'TRANSLATOR_VOID_BLUE',
  'mean': -0.08227076415220481,
  'std': 0.03974892070796538},
 30: {'a': 'GALAXY_SOUNDS_SOLAR_WINDS',
  'b': 'OXYGEN_SHAKE_EVENING_BREATH',
  'mean': 0.12258717121752104,
  'std': 0.040805568083334846},
 31: {'a': 'SLEEP_POD_LAMB_WOOL',
  'b': 'SNACKPACK_RASPBERRY',
  'mean': 0.06046762257852749,
  'std': 0.03703249009155746},
 32: {'a': 'OXYGEN_SHAKE_CHOCOLATE',
  'b': 'SLEEP_POD_NYLON',
  'mean': -0.008394415887930844,
  'std': 0.040911450349299355},
 33: {'a': 'GALAXY_SOUNDS_SOLAR_FLAMES',
  'b': 'SNACKPACK_CHOCOLATE',
  'mean': 0.1222021158792687,
  'std': 0.03949979157506741},
 34: {'a': 'SNACKPACK_RASPBERRY',
  'b': 'SNACKPACK_STRAWBERRY',
  'mean': -0.06162890342627323,
  'std': 0.04373906860756546},
 35: {'a': 'GALAXY_SOUNDS_BLACK_HOLES',
  'b': 'OXYGEN_SHAKE_GARLIC',
  'mean': -0.04075429313427271,
  'std': 0.03936100317666244},
 36: {'a': 'GALAXY_SOUNDS_SOLAR_FLAMES',
  'b': 'ROBOT_VACUUMING',
  'mean': 0.1971826983998019,
  'std': 0.05101511525092184},
 37: {'a': 'MICROCHIP_RECTANGLE',
  'b': 'TRANSLATOR_ASTRO_BLACK',
  'mean': -0.07688639875102546,
  'std': 0.051522115605585055},
 38: {'a': 'SNACKPACK_STRAWBERRY',
  'b': 'TRANSLATOR_ECLIPSE_CHARCOAL',
  'mean': 0.08891802459683007,
  'std': 0.03929645757216058},
 39: {'a': 'OXYGEN_SHAKE_EVENING_BREATH',
  'b': 'UV_VISOR_RED',
  'mean': -0.18149040353645376,
  'std': 0.051861022766495325},
 40: {'a': 'SNACKPACK_STRAWBERRY',
  'b': 'TRANSLATOR_VOID_BLUE',
  'mean': -0.013886830172024996,
  'std': 0.033772595721312845},
 41: {'a': 'GALAXY_SOUNDS_SOLAR_FLAMES',
  'b': 'TRANSLATOR_ECLIPSE_CHARCOAL',
  'mean': 0.12518474855220404,
  'std': 0.047523538036685804},
 42: {'a': 'PANEL_2X4',
  'b': 'TRANSLATOR_VOID_BLUE',
  'mean': 0.03706551533428622,
  'std': 0.040957115174379735},
 43: {'a': 'PEBBLES_XL',
  'b': 'ROBOT_DISHES',
  'mean': 0.2793647403913547,
  'std': 0.08767202690813},
 44: {'a': 'PEBBLES_L',
  'b': 'TRANSLATOR_GRAPHITE_MIST',
  'mean': 0.007763062176722697,
  'std': 0.049321300567428186},
 45: {'a': 'SLEEP_POD_COTTON',
  'b': 'SLEEP_POD_POLYESTER',
  'mean': -0.027665878728799304,
  'std': 0.04101624560531991},
 46: {'a': 'PEBBLES_S',
  'b': 'ROBOT_LAUNDRY',
  'mean': -0.1023446222912656,
  'std': 0.0562618152740881},
 47: {'a': 'SLEEP_POD_LAMB_WOOL',
  'b': 'SNACKPACK_VANILLA',
  'mean': 0.05905054290423063,
  'std': 0.036204542349821706},
 48: {'a': 'PANEL_1X2',
  'b': 'SLEEP_POD_NYLON',
  'mean': -0.08044997072815231,
  'std': 0.04315754330054034},
 49: {'a': 'GALAXY_SOUNDS_DARK_MATTER',
  'b': 'TRANSLATOR_ECLIPSE_CHARCOAL',
  'mean': 0.04255338095859698,
  'std': 0.042027966648899164},
 50: {'a': 'PANEL_2X4',
  'b': 'UV_VISOR_MAGENTA',
  'mean': 0.013575657253859091,
  'std': 0.039878957743164016},
 51: {'a': 'SNACKPACK_STRAWBERRY',
  'b': 'UV_VISOR_MAGENTA',
  'mean': -0.03737668825245212,
  'std': 0.037286969467808684},
 52: {'a': 'PEBBLES_S',
  'b': 'ROBOT_IRONING',
  'mean': 0.025118061709715582,
  'std': 0.059975110786580195},
 53: {'a': 'GALAXY_SOUNDS_SOLAR_WINDS',
  'b': 'TRANSLATOR_GRAPHITE_MIST',
  'mean': 0.035289484140477605,
  'std': 0.04255365201549993},
 54: {'a': 'ROBOT_DISHES',
  'b': 'UV_VISOR_RED',
  'mean': -0.10071316758089362,
  'std': 0.046527123776761964},
 55: {'a': 'OXYGEN_SHAKE_MORNING_BREATH',
  'b': 'ROBOT_IRONING',
  'mean': 0.14527337097229887,
  'std': 0.04948937034063898},
 56: {'a': 'PEBBLES_M',
  'b': 'UV_VISOR_RED',
  'mean': -0.07647157608896397,
  'std': 0.057067626394893795},
 57: {'a': 'GALAXY_SOUNDS_DARK_MATTER',
  'b': 'SNACKPACK_RASPBERRY',
  'mean': 0.015264259788040139,
  'std': 0.03719305518344513},
 58: {'a': 'PEBBLES_M',
  'b': 'SLEEP_POD_NYLON',
  'mean': 0.06513966379519225,
  'std': 0.05173716246914589},
 59: {'a': 'GALAXY_SOUNDS_BLACK_HOLES',
  'b': 'OXYGEN_SHAKE_CHOCOLATE',
  'mean': 0.18679818416771501,
  'std': 0.055522826436002065},
 60: {'a': 'OXYGEN_SHAKE_EVENING_BREATH',
  'b': 'SNACKPACK_STRAWBERRY',
  'mean': -0.14916116993573997,
  'std': 0.04856541634172257},
 61: {'a': 'PANEL_1X2',
  'b': 'PANEL_2X4',
  'mean': -0.24068432251790595,
  'std': 0.0739365378120658},
 62: {'a': 'SNACKPACK_CHOCOLATE',
  'b': 'SNACKPACK_VANILLA',
  'mean': -0.02572356817191836,
  'std': 0.037899673105914425},
 63: {'a': 'MICROCHIP_CIRCLE',
  'b': 'PEBBLES_XL',
  'mean': -0.3654228357453227,
  'std': 0.10768716278193816},
 64: {'a': 'SLEEP_POD_POLYESTER',
  'b': 'SLEEP_POD_SUEDE',
  'mean': 0.04015225458102201,
  'std': 0.042193416236059024}}

Z_SCORE_MAX_THRESHOLD = 3.0  # Z-score for max trade size
Z_SCORE_OPEN_THRESHOLD = 1.0  # Z-score to open the entire position
Z_SCORE_CLOSE_THRESHOLD = 0.5  # Z-score to close the entire position
MIN_TRADE_SIZE = 1  # Minimum number of units to trade in a single order
MAX_TRADE_SIZE = 10  # Maximum number of units to trade in a single order #in practice no trades are being offered at this volume lol
SCALING_EXPONENT = 2.0  # Exponent for volume scaling. >1 for exponential, 1 for linear.


# ── Per-product order helper ──────────────────────────────────────────────────
class ProductTrader:
    def __init__(self, name: str, state: TradingState):
        self.name = name
        self.orders = []
        self.position_limit = POSITION_LIMIT
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


# ── Pairs Trading Logic ───────────────────────────────────────────────────────
class PairsTrader:
    def __init__(self, state: TradingState, pair_params: dict):
        self.params = pair_params
        self.product_a = ProductTrader(self.params["a"], state)
        self.product_b = ProductTrader(self.params["b"], state)

    def _trade(self):
        # We need prices to make decisions.
        if (
            self.product_a.best_bid is None
            or self.product_a.best_ask is None
            or self.product_b.best_bid is None
            or self.product_b.best_ask is None
        ):
            return

        # open_threshold = self.params["z_score_open_threshold"]
        # close_threshold = self.params["z_score_close_threshold"]
        open_threshold = Z_SCORE_OPEN_THRESHOLD
        # close_threshold = Z_SCORE_CLOSE_THRESHOLD

        # pos_a = self.product_a.initial_position
        # pos_b = self.product_b.initial_position

        # --- Opening Logic ---
        # When the ratio is high, we consider shorting the spread (Sell A, Buy B).
        # The signal is based on the price to go LONG the spread (Buy A @ ask, Sell B @ bid).
        log_ratio_for_short_signal = math.log(self.product_a.best_ask / self.product_b.best_bid)
        z_score_for_short_signal = (log_ratio_for_short_signal - self.params["mean"]) / self.params["std"]

        # When the ratio is low, we consider longing the spread (Buy A, Sell B).
        # The signal is based on the price to go SHORT the spread (Sell A @ bid, Buy B @ ask).
        log_ratio_for_long_signal = math.log(self.product_a.best_bid / self.product_b.best_ask)
        z_score_for_long_signal = (log_ratio_for_long_signal - self.params["mean"]) / self.params["std"]

        # Ratio is high -> A is overvalued vs B. Sell A, Buy B.
        if z_score_for_short_signal > open_threshold:
            # Scale trade size based on how far z-score is above the open threshold
            linear_scaling_factor = min(
                (z_score_for_short_signal - open_threshold)
                / (Z_SCORE_MAX_THRESHOLD - open_threshold),
                1.0,
            )
            exponential_scaling_factor = linear_scaling_factor**SCALING_EXPONENT
            trade_size = int(
                MIN_TRADE_SIZE + exponential_scaling_factor * (MAX_TRADE_SIZE - MIN_TRADE_SIZE)
            )

            # Sell A at best bid, Buy B at best ask
            self.product_a.ask(self.product_a.best_bid, trade_size)
            self.product_b.bid(self.product_b.best_ask, trade_size)

        # Ratio is low -> A is undervalued vs B. Buy A, Sell B.
        elif z_score_for_long_signal < -open_threshold:
            z_abs = abs(z_score_for_long_signal)
            # Scale trade size based on how far z-score is above the open threshold
            linear_scaling_factor = min(
                (z_abs - open_threshold) / (Z_SCORE_MAX_THRESHOLD - open_threshold),
                1.0,
            )
            exponential_scaling_factor = linear_scaling_factor**SCALING_EXPONENT
            trade_size = int(
                MIN_TRADE_SIZE + exponential_scaling_factor * (MAX_TRADE_SIZE - MIN_TRADE_SIZE)
            )

            # Buy A at best ask, Sell B at best bid
            self.product_a.bid(self.product_a.best_ask, trade_size)
            self.product_b.ask(self.product_b.best_bid, trade_size)

        # --- Closing logic: close position as price reverts to mean ---
        # else:
        #     # For closing, we check the mid-price to see if the theoretical value has reverted.
        #     log_ratio_mid = math.log(self.product_a.mid_price / self.product_b.mid_price)
        #     z_score_mid = (log_ratio_mid - self.params["mean"]) / self.params["std"]

        #     # We are long A, short B, and spread is reverting (z_score is less extreme)
        #     if pos_a > 0 and pos_b < 0 and z_score_mid > -close_threshold:
        #         # Close position: Sell A, Buy B
        #         self.product_a.ask(self.product_a.best_bid, pos_a)
        #         self.product_b.bid(self.product_b.best_ask, abs(pos_b))

        #     # We are short A, long B, and spread is reverting (z_score is less extreme)
        #     elif pos_a < 0 and pos_b > 0 and z_score_mid < close_threshold:
        #         # Close position: Buy A, Sell B
        #         self.product_a.bid(self.product_a.best_ask, abs(pos_a))
        #         self.product_b.ask(self.product_b.best_bid, pos_b)

    def get_orders(self) -> dict:
        self._trade()
        orders = self.product_a.get_orders()
        orders.update(self.product_b.get_orders())
        return orders


class Trader:
    def run(self, state: TradingState):
        result = {}

        # --- Pairs Trading ---
        for _, pair_params in PAIRS_TRADING_PARAMS.items():
            asset_a = pair_params["a"]
            asset_b = pair_params["b"]

            # If both assets in the pair have already had orders generated, skip this pair.
            if asset_a in result and asset_b in result:
                continue

            if asset_a in state.order_depths and asset_b in state.order_depths:
                trader = PairsTrader(state, pair_params)
                orders = trader.get_orders()

                # If one asset in the pair has already been traded, we only
                # execute the trade for the other, untraded asset.
                # WARNING: This breaks the pair's hedge and introduces directional risk.
                if asset_a in result:
                    orders.pop(asset_a, None)
                if asset_b in result:
                    orders.pop(asset_b, None)

                result.update(orders)

        # This strategy is stateless, so trader_data is an empty string
        trader_data = ""
        logger.flush(state, result, 0, trader_data)
        return result, 0, trader_data