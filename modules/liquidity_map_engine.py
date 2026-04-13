"""
modules/liquidity_map_engine.py
─────────────────────────────────────────────────────────────────────────────
Deterministic liquidity map engine:
- EQH/EQL clustering
- sweep detection (recent)
- inducement candidate level
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import pandas as pd

from modules.market_structure_engine import SwingPoint


@dataclass
class LiquidityPool:
    side: str  # BUY_SIDE | SELL_SIDE
    level: float
    members: List[int]


@dataclass
class LiquidityMapResult:
    eqh: List[Dict[str, Any]] = None  # type: ignore[assignment]
    eql: List[Dict[str, Any]] = None  # type: ignore[assignment]
    sweep_detected: bool = False
    sweep_side: str = "NONE"  # BUY_SIDE | SELL_SIDE | NONE
    sweep_index: int = -1
    inducement_level: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["eqh"] = self.eqh or []
        d["eql"] = self.eql or []
        return d


class LiquidityMapEngine:
    def __init__(self, tolerance_pct: float = 0.0005) -> None:
        self.tolerance_pct = tolerance_pct

    def analyze(
        self, df: pd.DataFrame, swing_highs: List[SwingPoint], swing_lows: List[SwingPoint]
    ) -> LiquidityMapResult:
        if df is None or df.empty:
            return LiquidityMapResult()

        eqh = self._cluster_levels(swing_highs, side="BUY_SIDE")
        eql = self._cluster_levels(swing_lows, side="SELL_SIDE")
        sweep_detected, sweep_side, sweep_idx = self._recent_sweep(df, eqh, eql)
        inducement = self._inducement_level(df, swing_highs, swing_lows)

        return LiquidityMapResult(
            eqh=[asdict(x) for x in eqh],
            eql=[asdict(x) for x in eql],
            sweep_detected=sweep_detected,
            sweep_side=sweep_side,
            sweep_index=sweep_idx,
            inducement_level=inducement,
        )

    def _cluster_levels(self, swings: List[SwingPoint], side: str) -> List[LiquidityPool]:
        pools: List[LiquidityPool] = []
        for s in swings:
            placed = False
            for p in pools:
                tol = max(1e-8, p.level * self.tolerance_pct)
                if abs(s.price - p.level) <= tol:
                    p.members.append(s.index)
                    p.level = (p.level * (len(p.members) - 1) + s.price) / len(p.members)
                    placed = True
                    break
            if not placed:
                pools.append(LiquidityPool(side=side, level=s.price, members=[s.index]))
        return [p for p in pools if len(p.members) >= 2]

    def _recent_sweep(
        self, df: pd.DataFrame, eqh: List[LiquidityPool], eql: List[LiquidityPool]
    ) -> tuple[bool, str, int]:
        start = max(0, len(df) - 3)
        for i in range(len(df) - 1, start - 1, -1):
            r = df.iloc[i]
            for p in eqh:
                if r["high"] > p.level and r["close"] < p.level:
                    return True, "BUY_SIDE", i
            for p in eql:
                if r["low"] < p.level and r["close"] > p.level:
                    return True, "SELL_SIDE", i
        return False, "NONE", -1

    def _inducement_level(
        self, df: pd.DataFrame, highs: List[SwingPoint], lows: List[SwingPoint]
    ) -> float:
        _ = df
        if highs and lows:
            return float((highs[-1].price + lows[-1].price) / 2.0)
        return 0.0

