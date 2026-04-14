"""
modules/market_structure_engine.py
─────────────────────────────────────────────────────────────────────────────
Deterministic market structure engine:
- swing high/low detection
- macro/micro trend classification
- BOS / CHoCH event tagging
- premium / discount state
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class SwingPoint:
    index: int
    price: float
    kind: str  # SH | SL


@dataclass
class MarketStructureResult:
    macro_trend: str = "RANGING"  # BULLISH | BEARISH | RANGING
    micro_trend: str = "RANGING"
    last_event: str = "NONE"  # BOS | CHOCH | NONE
    last_event_direction: str = "NONE"  # BULLISH | BEARISH | NONE
    last_event_index: int = -1
    last_event_price: float = 0.0
    premium_discount: str = "EQUILIBRIUM"  # PREMIUM | DISCOUNT | EQUILIBRIUM
    eq_price: float = 0.0
    last_swing_highs: List[Dict[str, Any]] = None  # type: ignore[assignment]
    last_swing_lows: List[Dict[str, Any]] = None  # type: ignore[assignment]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["last_swing_highs"] = self.last_swing_highs or []
        d["last_swing_lows"] = self.last_swing_lows or []
        return d


class MarketStructureEngine:
    def __init__(self, swing_window: int = 2) -> None:
        self.swing_window = swing_window

    def analyze(self, df: pd.DataFrame) -> MarketStructureResult:
        if df is None or len(df) < 10:
            return MarketStructureResult()

        swings_h, swings_l = self._find_swings(df, self.swing_window)
        macro = self._classify_trend(swings_h, swings_l, lookback=6)
        micro = self._classify_trend(swings_h, swings_l, lookback=3)
        evt, evt_dir, evt_idx, evt_price = self._detect_structure_event(
            df, swings_h, swings_l, macro
        )
        pd_state, eq = self._premium_discount(df, swings_h, swings_l)

        return MarketStructureResult(
            macro_trend=macro,
            micro_trend=micro,
            last_event=evt,
            last_event_direction=evt_dir,
            last_event_index=evt_idx,
            last_event_price=evt_price,
            premium_discount=pd_state,
            eq_price=eq,
            last_swing_highs=[asdict(s) for s in swings_h[-5:]],
            last_swing_lows=[asdict(s) for s in swings_l[-5:]],
        )

    def _find_swings(
        self, df: pd.DataFrame, w: int
    ) -> Tuple[List[SwingPoint], List[SwingPoint]]:
        highs: List[SwingPoint] = []
        lows: List[SwingPoint] = []
        for i in range(w, len(df) - w):
            h = df["high"].iloc[i]
            l = df["low"].iloc[i]
            left_h = df["high"].iloc[i - w : i]
            right_h = df["high"].iloc[i + 1 : i + 1 + w]
            left_l = df["low"].iloc[i - w : i]
            right_l = df["low"].iloc[i + 1 : i + 1 + w]
            if h > left_h.max() and h > right_h.max():
                highs.append(SwingPoint(index=i, price=float(h), kind="SH"))
            if l < left_l.min() and l < right_l.min():
                lows.append(SwingPoint(index=i, price=float(l), kind="SL"))
        return highs, lows

    def _classify_trend(
        self, highs: List[SwingPoint], lows: List[SwingPoint], lookback: int
    ) -> str:
        if len(highs) < 2 or len(lows) < 2:
            return "RANGING"
        hs = highs[-lookback:]
        ls = lows[-lookback:]
        hh = sum(1 for i in range(1, len(hs)) if hs[i].price > hs[i - 1].price)
        lh = sum(1 for i in range(1, len(hs)) if hs[i].price < hs[i - 1].price)
        hl = sum(1 for i in range(1, len(ls)) if ls[i].price > ls[i - 1].price)
        ll = sum(1 for i in range(1, len(ls)) if ls[i].price < ls[i - 1].price)
        if hh >= max(1, len(hs) // 2) and hl >= max(1, len(ls) // 2):
            return "BULLISH"
        if lh >= max(1, len(hs) // 2) and ll >= max(1, len(ls) // 2):
            return "BEARISH"
        return "RANGING"

    def _detect_structure_event(
        self,
        df: pd.DataFrame,
        highs: List[SwingPoint],
        lows: List[SwingPoint],
        macro_trend: str,
    ) -> Tuple[str, str, int, float]:
        if not highs or not lows:
            return "NONE", "NONE", -1, 0.0
        close = float(df["close"].iloc[-1])
        last_high = highs[-1]
        last_low = lows[-1]

        if close > last_high.price:
            evt = "BOS" if macro_trend != "BEARISH" else "CHOCH"
            return evt, "BULLISH", int(last_high.index), float(last_high.price)
        if close < last_low.price:
            evt = "BOS" if macro_trend != "BULLISH" else "CHOCH"
            return evt, "BEARISH", int(last_low.index), float(last_low.price)
        return "NONE", "NONE", -1, 0.0

    def _premium_discount(
        self, df: pd.DataFrame, highs: List[SwingPoint], lows: List[SwingPoint]
    ) -> Tuple[str, float]:
        if not highs or not lows:
            return "EQUILIBRIUM", 0.0
        hi = highs[-1].price
        lo = lows[-1].price
        if hi <= lo:
            hi = float(df["high"].tail(30).max())
            lo = float(df["low"].tail(30).min())
        eq = (hi + lo) / 2.0
        close = float(df["close"].iloc[-1])
        if close > eq:
            return "PREMIUM", eq
        if close < eq:
            return "DISCOUNT", eq
        return "EQUILIBRIUM", eq

