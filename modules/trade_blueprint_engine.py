"""
modules/trade_blueprint_engine.py
─────────────────────────────────────────────────────────────────────────────
Build deterministic trade blueprints from confluence + structure context.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List


@dataclass
class TradeBlueprint:
    direction: str = "NONE"  # LONG | SHORT | NONE
    entry_price: float = 0.0
    stop_loss: float = 0.0
    tp1: float = 0.0
    tp2: float = 0.0
    tp3: float = 0.0
    rr_tp1: float = 0.0
    rr_tp2: float = 0.0
    rr_tp3: float = 0.0
    invalidation_price: float = 0.0
    valid: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TradeBlueprintEngine:
    def __init__(self, min_rr: float = 1.5) -> None:
        self.min_rr = min_rr

    def build(
        self,
        direction: str,
        close: float,
        sr_levels: List[float],
        market_structure: Dict[str, Any],
        min_rr_override: float | None = None,
    ) -> TradeBlueprint:
        if direction not in ("LONG", "SHORT") or close <= 0:
            return TradeBlueprint(direction="NONE")

        # Structural invalidation anchor from latest swing list
        lows = market_structure.get("last_swing_lows", []) or []
        highs = market_structure.get("last_swing_highs", []) or []
        if direction == "LONG":
            sl = float(lows[-1]["price"]) if lows else close * 0.99
            if sl >= close:
                sl = close * 0.99
            risk = close - sl
            tps = self._next_levels(sr_levels, close, above=True)
        else:
            sl = float(highs[-1]["price"]) if highs else close * 1.01
            if sl <= close:
                sl = close * 1.01
            risk = sl - close
            tps = self._next_levels(sr_levels, close, above=False)

        if risk <= 0:
            return TradeBlueprint(direction=direction, entry_price=close, stop_loss=sl)

        tp1 = tps[0] if len(tps) > 0 else (close + risk if direction == "LONG" else close - risk)
        tp2 = tps[1] if len(tps) > 1 else (close + 2 * risk if direction == "LONG" else close - 2 * risk)
        tp3 = tps[2] if len(tps) > 2 else (close + 3 * risk if direction == "LONG" else close - 3 * risk)

        rr1 = self._rr(direction, close, sl, tp1)
        rr2 = self._rr(direction, close, sl, tp2)
        rr3 = self._rr(direction, close, sl, tp3)
        rr_floor = self.min_rr if min_rr_override is None else float(min_rr_override)
        valid = rr1 >= rr_floor

        return TradeBlueprint(
            direction=direction,
            entry_price=close,
            stop_loss=sl,
            tp1=tp1,
            tp2=tp2,
            tp3=tp3,
            rr_tp1=rr1,
            rr_tp2=rr2,
            rr_tp3=rr3,
            invalidation_price=sl,
            valid=valid,
        )

    def _next_levels(self, levels: List[float], close: float, above: bool) -> List[float]:
        uniq = sorted(set(float(x) for x in levels))
        if above:
            return [x for x in uniq if x > close][:3]
        return [x for x in reversed(uniq) if x < close][:3]

    def _rr(self, direction: str, entry: float, sl: float, tp: float) -> float:
        risk = abs(entry - sl)
        if risk <= 0:
            return 0.0
        reward = (tp - entry) if direction == "LONG" else (entry - tp)
        return round(max(0.0, reward / risk), 4)
