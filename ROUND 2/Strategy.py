
from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Any, Tuple
import json
import math


# ─────────────────────────────────────────────────────────────
# BASIC ORDER BOOK UTILITIES
# ─────────────────────────────────────────────────────────────

def best_bid(depth: OrderDepth) -> int:
    return max(depth.buy_orders) if depth.buy_orders else 0


def best_ask(depth: OrderDepth) -> int:
    return min(depth.sell_orders) if depth.sell_orders else 0


def l1_bid_volume(depth: OrderDepth) -> int:
    bb = best_bid(depth)
    return depth.buy_orders.get(bb, 0)


def l1_ask_volume(depth: OrderDepth) -> int:
    ba = best_ask(depth)
    return abs(depth.sell_orders.get(ba, 0))


def book_imbalance(depth: OrderDepth) -> float:
    """(BidVol - AskVol) / TotalVol"""
    bv = l1_bid_volume(depth)
    av = l1_ask_volume(depth)
    total = bv + av
    return (bv - av) / total if total > 0 else 0.0


def microprice(depth: OrderDepth) -> float:
    """Volume-weighted mid price"""
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


# ─────────────────────────────────────────────────────────────
# STATE MANAGEMENT
# ─────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────
# ASH_COATED_OSMIUM STRATEGY (UPDATED V4 - AGGRESSIVE L2)
# ─────────────────────────────────────────────────────────────

def osmium_strategy_v3(
    state: TradingState,
    position: int,
    limit: int,
    memory: dict
) -> List[Order]:

    PRODUCT = "ASH_COATED_OSMIUM"

    # --- Parameters ---
    EMA_ALPHA = 0.20  # Increased for faster reaction
    INVENTORY_SKEW = 0.04 # Slightly reduced to allow more aggressive positioning
    
    orders: List[Order] = []
    depth = state.order_depths.get(PRODUCT)

    if not depth or not depth.buy_orders or not depth.sell_orders:
        return []

    # --- Level 1 & Level 2 Data ---
    sorted_buy_prices = sorted(depth.buy_orders.keys(), reverse=True)
    sorted_sell_prices = sorted(depth.sell_orders.keys())

    bb1 = sorted_buy_prices[0]
    bv1 = depth.buy_orders[bb1]
    bb2 = sorted_buy_prices[1] if len(sorted_buy_prices) > 1 else bb1 - 1
    bv2 = depth.buy_orders.get(bb2, 0)

    ba1 = sorted_sell_prices[0]
    av1 = abs(depth.sell_orders[ba1])
    ba2 = sorted_sell_prices[1] if len(sorted_sell_prices) > 1 else ba1 + 1
    av2 = abs(depth.sell_orders.get(ba2, 0))

    # --- Fair Price Calculation ---
    total_vol = bv1 + av1
    micro = ((bb1 * av1 + ba1 * bv1) / total_vol) if total_vol > 0 else (bb1 + ba1) / 2
    
    prev_ema = memory.get("ema", micro)
    ema = EMA_ALPHA * micro + (1 - EMA_ALPHA) * prev_ema
    memory["ema"] = ema

    # Reservation price (Where we want our mid to be based on inventory)
    res_price = ema - (position * INVENTORY_SKEW)

    # Capacity
    buy_cap = limit - position
    sell_cap = -limit - position # negative value

    # =========================================================
    # MARKET TAKING (AGGRESSIVE L1 + L2)
    # =========================================================

    # Taking Sells (Buying)
    for price, vol in [(ba1, av1), (ba2, av2)]:
        if price <= res_price - 0.5 and buy_cap > 0:
            qty = min(buy_cap, vol)
            orders.append(Order(PRODUCT, price, qty))
            buy_cap -= qty

    # Taking Bids (Selling)
    for price, vol in [(bb1, bv1), (bb2, bv2)]:
        if price >= res_price + 0.5 and sell_cap < 0:
            qty = min(abs(sell_cap), vol)
            orders.append(Order(PRODUCT, price, -qty))
            sell_cap += qty

    # =========================================================
    # MARKET MAKING (LAYERED L1 + L2)
    # =========================================================

    # Competitive Bidding
    if buy_cap > 0:
        # Layer 1: Penny the best bid or follow res_price
        bid_l1 = int(min(res_price - 1, bb1 + 1, ba1 - 1))
        qty_l1 = max(buy_cap // 2, 1)
        orders.append(Order(PRODUCT, bid_l1, qty_l1))
        
        # Layer 2: Deeper liquidity at bb2 or slightly below L1
        if buy_cap - qty_l1 > 0:
            bid_l2 = min(bid_l1 - 1, bb2 + 1)
            orders.append(Order(PRODUCT, bid_l2, buy_cap - qty_l1))

    # Competitive Asking
    if sell_cap < 0:
        # Layer 1: Penny the best ask
        ask_l1 = int(max(res_price + 1, ba1 - 1, bb1 + 1))
        qty_l1 = max(abs(sell_cap) // 2, 1)
        orders.append(Order(PRODUCT, ask_l1, -qty_l1))
        
        # Layer 2: Deeper liquidity
        if abs(sell_cap) - qty_l1 > 0:
            ask_l2 = max(ask_l1 + 1, ba2 - 1)
            orders.append(Order(PRODUCT, ask_l2, -(abs(sell_cap) - qty_l1)))

    return orders


# ─────────────────────────────────────────────────────────────
# INTARIAN_PEPPER_ROOT STRATEGY
# ─────────────────────────────────────────────────────────────

class PepperStrategy:

    WINDOW = 5

    IMB_COEFF = 0.8
    MICRO_COEFF = 1
    INV_COEFF = 1 

    BASE_SIZE = 15 #min quantity we place per trade
    MAX_POS = 80

#add +1 to spread

    L1_OFFSET = 8
    L2_OFFSET = 5

    TREND = 6

    TAKE_EDGE_BUY = 5
    TAKE_EDGE_SELL = 1

    SELL_THRESHOLD = 20 

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

        mid = (bb + ba) / 2
        imb = book_imbalance(depth)
        mp = microprice(depth)
        micro_dev = mp - mid

        # --- rolling window of mids ---
        mids = ts.get("mids", [])
        mids.append(mid)
        mids = mids[-self.WINDOW:]
        ts = {"mids": mids}

        if len(mids) < 4:
            return [], ts

        # --- median fair anchor ---
        sorted_m = sorted(mids)
        n = len(sorted_m)
        median = sorted_m[n // 2] if n % 2 else (sorted_m[n//2 - 1] + sorted_m[n//2]) / 2

        # --- simple trend ---
        trend = (mids[-1] - mids[0]) / max(1, len(mids) - 1) if len(mids) >= 12 else 0

        # --- fair price ---
        fair = (
            median
            + self.TREND * trend
            + self.MICRO_COEFF * micro_dev
            + self.IMB_COEFF * imb
            - self.INV_COEFF * (pos - self.MAX_POS)
        )

        buy_room = max(0, limit - pos)
        sell_room = max(0, limit + pos)

        # =========================================================
        # MARKET TAKING
        # =========================================================

        best_ask_vol = abs(depth.sell_orders[ba])
        best_bid_vol = depth.buy_orders[bb]

        if ba <= fair - self.TAKE_EDGE_BUY and buy_room > 0:
            qty = min(best_ask_vol, buy_room, self.BASE_SIZE + 4)
            if qty > 0:
                orders.append(Order("INTARIAN_PEPPER_ROOT", ba, qty))
                buy_room -= qty

        if pos > 40 and bb >= fair + self.TAKE_EDGE_SELL and micro_dev < 0:
            qty = min(best_bid_vol, sell_room, pos)
            if qty > 0:
                orders.append(Order("INTARIAN_PEPPER_ROOT", bb, -qty))
                sell_room -= qty

        # =========================================================
        # MARKET MAKING (L1)
        # =========================================================

        bid_px = int(round(fair - self.L1_OFFSET))
        ask_px = int(round(fair + self.L1_OFFSET + 1))

        bid_px = min(bid_px, ba - 1)
        ask_px = max(ask_px, bb + 1)

        if bid_px < ask_px:
            if buy_room > 0:
                orders.append(Order("INTARIAN_PEPPER_ROOT", bid_px, min(self.BASE_SIZE, buy_room)))

            if pos > 5 and sell_room > 0:
                orders.append(Order("INTARIAN_PEPPER_ROOT", ask_px, -min(self.BASE_SIZE, sell_room, pos)))

        # =========================================================
        # MARKET MAKING (L2)
        # =========================================================

        spread = ba - bb
        if spread >= 4:

            bid_px2 = int(round(fair - self.L2_OFFSET))
            ask_px2 = int(round(fair + self.L2_OFFSET + 1))

            bid_px2 = min(bid_px2, ba - 1)
            ask_px2 = max(ask_px2, bb + 1)

            if bid_px2 < ask_px2:
                if buy_room > 0:
                    orders.append(Order("INTARIAN_PEPPER_ROOT", bid_px2, min(self.BASE_SIZE // 2, buy_room)))

                if pos > self.SELL_THRESHOLD and sell_room > 0:
                    orders.append(Order("INTARIAN_PEPPER_ROOT", ask_px2, -min(self.BASE_SIZE // 2, sell_room, pos)))

        return orders, ts


# ─────────────────────────────────────────────────────────────
# TRADER ENTRY POINT
# ─────────────────────────────────────────────────────────────

class Trader:

    LIMITS = {
        "ASH_COATED_OSMIUM": 80,
        "INTARIAN_PEPPER_ROOT": 80,
    }

    def __init__(self):
        self.pepper = PepperStrategy()
    
    def bid(self):
        return 12500

    def run(self, state: TradingState):

        memory = StateManager.load(state.traderData)
        orders = {}

        # --- Osmium ---
        if "ASH_COATED_OSMIUM" in state.order_depths:
            mem = memory.setdefault("osmium", {})
            orders["ASH_COATED_OSMIUM"] = osmium_strategy_v3(
                state,
                state.position.get("ASH_COATED_OSMIUM", 0),
                self.LIMITS["ASH_COATED_OSMIUM"],
                mem,
            )

        # --- Pepper ---
        if "INTARIAN_PEPPER_ROOT" in state.order_depths:
            ts = memory.get("pepper", {})
            out, ts = self.pepper.run(
                state.order_depths["INTARIAN_PEPPER_ROOT"],
                state.position.get("INTARIAN_PEPPER_ROOT", 0),
                self.LIMITS["INTARIAN_PEPPER_ROOT"],
                ts,
            )
            orders["INTARIAN_PEPPER_ROOT"] = out
            memory["pepper"] = ts

        return orders, 0, StateManager.dump(memory)
