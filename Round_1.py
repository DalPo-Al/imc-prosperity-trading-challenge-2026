from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Any, Tuple
import json
import math


# ── Shared utilities ──────────────────────────────────────────────────────────

def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def best_bid(depth: OrderDepth) -> int:
    return max(depth.buy_orders) if depth.buy_orders else 0


def best_ask(depth: OrderDepth) -> int:
    return min(depth.sell_orders) if depth.sell_orders else 0


def l1_bid_volume(depth: OrderDepth) -> int:
    bb = best_bid(depth)
    return depth.buy_orders[bb] if bb else 0


def l1_ask_volume(depth: OrderDepth) -> int:
    ba = best_ask(depth)
    return abs(depth.sell_orders[ba]) if ba else 0


def book_imbalance(depth: OrderDepth) -> float:
    if not depth.buy_orders or not depth.sell_orders:
        return 0.0
    bv = l1_bid_volume(depth)
    av = l1_ask_volume(depth)
    total = bv + av
    return (bv - av) / total if total > 0 else 0.0


def microprice(depth: OrderDepth) -> float:
    bb = best_bid(depth)
    ba = best_ask(depth)
    if not bb or not ba:
        return 0.0
    bv = l1_bid_volume(depth)
    av = l1_ask_volume(depth)
    total = bv + av
    if total <= 0:
        return (bb + ba) / 2.0
    return (ba * bv + bb * av) / total


# ── State management ──────────────────────────────────────────────────────────

class StateManager:
    @staticmethod
    def load(raw: str) -> Dict[str, Any]:
        try:
            data = json.loads(raw) if raw else {}
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def dump(data: Dict[str, Any]) -> str:
        return json.dumps(data, separators=(",", ":"))


# ── ASH_COATED_OSMIUM ─────────────────────────────────────────────────────────

def osmium_strategy_v3(
    state: TradingState,
    position: int,
    limit: int,
    memory: dict
) -> List[Order]:
    PRODUCT = "ASH_COATED_OSMIUM"
    EMA_ALPHA = 0.15
    INVENTORY_SKEW = 0.05

    order_depth = state.order_depths.get(PRODUCT)
    orders: List[Order] = []

    if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders:
        return []

    buy_prices = sorted(order_depth.buy_orders.keys(), reverse=True)
    sell_prices = sorted(order_depth.sell_orders.keys())

    best_bid_px = buy_prices[0]
    best_ask_px = sell_prices[0]
    vol_bid = order_depth.buy_orders[best_bid_px]
    vol_ask = abs(order_depth.sell_orders[best_ask_px])

    total_vol = vol_bid + vol_ask
    micro_price = (
        (best_bid_px * vol_ask + best_ask_px * vol_bid) / total_vol
        if total_vol > 0
        else (best_bid_px + best_ask_px) / 2
    )

    prev_ema = memory.get("osmium_ema", micro_price)
    current_ema = (micro_price * EMA_ALPHA) + (prev_ema * (1 - EMA_ALPHA))
    memory["osmium_ema"] = current_ema

    buy_capacity = limit - position
    sell_capacity = -limit - position

    urgency_multiplier = 1.0 + (abs(position) / limit)
    res_price = current_ema - (INVENTORY_SKEW * urgency_multiplier * position)

    SELF_PRICE = 10_000
    MIN_EDGE = 0.0

    if best_ask_px < SELF_PRICE - MIN_EDGE and buy_capacity > 0:
        vol = min(vol_ask, buy_capacity)
        orders.append(Order(PRODUCT, best_ask_px, int(vol)))
        buy_capacity -= vol

    if best_bid_px > SELF_PRICE + MIN_EDGE and sell_capacity < 0:
        vol = min(vol_bid, abs(sell_capacity))
        orders.append(Order(PRODUCT, best_bid_px, int(-vol)))
        sell_capacity += vol

    theoretical_bid = math.floor(res_price - 0.2)
    theoretical_ask = math.ceil(res_price + 0.2)

    quote_bid = int(min(theoretical_bid, best_bid_px + 1, best_ask_px - 1))
    quote_ask = int(max(theoretical_ask, best_ask_px - 1, best_bid_px + 1))

    if buy_capacity > 0:
        orders.append(Order(PRODUCT, quote_bid, int(buy_capacity)))
    if sell_capacity < 0:
        orders.append(Order(PRODUCT, quote_ask, int(sell_capacity)))

    return orders


# ── INTARIAN_PEPPER_ROOT ──────────────────────────────────────────────────────

class PepperStrategy:
    WINDOW = 24

    IMB_COEFF = 0.8
    MICRO_COEFF = 1
    INV_COEFF = 1

    BASE_SIZE = 20
    MAX_POS = 80

    L1_OFFSET = 8
    L2_OFFSET = 5

    TAKE_EDGE_BUY = 5
    TAKE_EDGE_SELL = 1

    TARGET_POS = 80
    SOFT_CAP = 60
    HARD_CAP = 75

    def run(
        self,
        depth: OrderDepth,
        pos: int,
        limit: int,
        ts: Dict[str, Any],
    ) -> Tuple[List[Order], Dict[str, Any]]:
        orders: List[Order] = []

        bb = best_bid(depth)
        ba = best_ask(depth)
        if not bb or not ba:
            return [], ts

        mid = (bb + ba) / 2.0
        imb = book_imbalance(depth)
        mp = microprice(depth)
        micro_dev = mp - mid
        spread = ba - bb

        mids: List[float] = ts.get("mids", [])
        mids.append(mid)
        mids = mids[-self.WINDOW:]
        new_ts = {"mids": mids}

        if len(mids) < 4:
            return [], new_ts

        sorted_m = sorted(mids)
        n = len(sorted_m)
        median = (
            sorted_m[n // 2]
            if n % 2
            else (sorted_m[n // 2 - 1] + sorted_m[n // 2]) / 2.0
        )

        trend = 0.0
        if len(mids) >= 12:
            trend = (mids[-1] - mids[0]) / max(1, len(mids) - 1)

        fair = (
            median
            + 6.0 * trend
            + self.MICRO_COEFF * micro_dev
            + self.IMB_COEFF * imb
            - self.INV_COEFF * (pos - self.TARGET_POS)
        )

        buy_room = max(0, limit - pos)
        sell_room = max(0, limit + pos)

        if pos < self.TARGET_POS:
            bid_sz = self.BASE_SIZE + 6
            ask_sz = max(2, self.BASE_SIZE - 5)
        elif pos < self.SOFT_CAP:
            bid_sz = self.BASE_SIZE + 2
            ask_sz = max(2, self.BASE_SIZE - 3)
        else:
            bid_sz = max(3, self.BASE_SIZE - 3)
            ask_sz = self.BASE_SIZE + 2

        best_ask_vol = abs(depth.sell_orders[ba])
        best_bid_vol = depth.buy_orders[bb]

        if ba <= fair - self.TAKE_EDGE_BUY and buy_room > 0:
            qty = min(best_ask_vol, min(buy_room, bid_sz + 4))
            if qty > 0:
                orders.append(Order("INTARIAN_PEPPER_ROOT", ba, qty))
                buy_room -= qty

        if pos > 40 and bb >= fair + self.TAKE_EDGE_SELL and micro_dev < 0 and sell_room > 0:
            qty = min(best_bid_vol, min(sell_room, ask_sz + 2, max(0, pos)))
            if qty > 0:
                orders.append(Order("INTARIAN_PEPPER_ROOT", bb, -qty))
                sell_room -= qty

        bid_px = int(round(fair - self.L1_OFFSET))
        ask_px = int(round(fair + self.L1_OFFSET + 1))

        if pos < self.TARGET_POS:
            bid_px += 1
        if pos > self.SOFT_CAP:
            ask_px -= 1
            bid_px -= 1

        mid_anchor = sum(mids) / len(mids)
        if mid > mid_anchor + 100:
            bid_px -= 1
            ask_px += 1
        elif mid < mid_anchor - 100:
            bid_px += 1

        bid_px = min(bid_px, ba - 1)
        ask_px = max(ask_px, bb + 1)

        if bid_px < ask_px:
            qty = min(bid_sz, buy_room)
            if qty > 0:
                orders.append(Order("INTARIAN_PEPPER_ROOT", bid_px, qty))
                buy_room -= qty

            sell_quote_qty = ask_sz if pos > 5 else 0
            qty = min(sell_quote_qty, sell_room, max(0, pos + 5))
            if qty > 0:
                orders.append(Order("INTARIAN_PEPPER_ROOT", ask_px, -qty))
                sell_room -= qty

        if spread >= 4:
            bid_px2 = int(round(fair - self.L2_OFFSET))
            ask_px2 = int(round(fair + self.L2_OFFSET + 1))

            if pos < self.TARGET_POS:
                bid_px2 += 1

            bid_px2 = min(bid_px2, ba - 1)
            ask_px2 = max(ask_px2, bb + 1)

            if bid_px2 < ask_px2:
                qty = min(max(4, bid_sz // 2), buy_room)
                if qty > 0 and bid_px2 != bid_px:
                    orders.append(Order("INTARIAN_PEPPER_ROOT", bid_px2, qty))

                if pos > 20:
                    qty = min(max(2, ask_sz // 2), sell_room, pos)
                    if qty > 0 and ask_px2 != ask_px:
                        orders.append(Order("INTARIAN_PEPPER_ROOT", ask_px2, -qty))

        if pos < -10 and buy_room > 0:
            qty = min(buy_room, 12)
            orders.append(Order("INTARIAN_PEPPER_ROOT", ba, qty))

        return orders, new_ts


# ── Trader ────────────────────────────────────────────────────────────────────

class Trader:
    LIMITS: Dict[str, int] = {
        "ASH_COATED_OSMIUM": 80,
        "INTARIAN_PEPPER_ROOT": 80,
    }

    def __init__(self) -> None:
        self._pepper = PepperStrategy()

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        memory = StateManager.load(state.traderData)
        trader_orders: Dict[str, List[Order]] = {}

        # ASH_COATED_OSMIUM
        if "ASH_COATED_OSMIUM" in state.order_depths:
            ash_memory = memory.setdefault("osmium", {})
            trader_orders["ASH_COATED_OSMIUM"] = osmium_strategy_v3(
                state,
                state.position.get("ASH_COATED_OSMIUM", 0),
                self.LIMITS["ASH_COATED_OSMIUM"],
                ash_memory,
            )

        # INTARIAN_PEPPER_ROOT
        if "INTARIAN_PEPPER_ROOT" in state.order_depths:
            pepper_state = memory.get("pepper", {})
            pepper_orders, pepper_state = self._pepper.run(
                state.order_depths["INTARIAN_PEPPER_ROOT"],
                state.position.get("INTARIAN_PEPPER_ROOT", 0),
                self.LIMITS["INTARIAN_PEPPER_ROOT"],
                pepper_state,
            )
            trader_orders["INTARIAN_PEPPER_ROOT"] = pepper_orders
            memory["pepper"] = pepper_state

        return trader_orders, 0, StateManager.dump(memory)