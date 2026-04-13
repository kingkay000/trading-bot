"""
modules/ob_fvg_engine.py
─────────────────────────────────────────────────────────────────────────────
Deterministic Order Block (OB) and Fair Value Gap (FVG) detection.
"""

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd


@dataclass
class OrderBlock:
    direction: str  # bullish | bearish
    index: int
    high: float
    low: float
    mitigated: bool


@dataclass
class FairValueGap:
    direction: str  # bullish | bearish
    index: int
    top: float
    bottom: float
    mitigated: bool


class OBFVGEngine:
    def detect_order_blocks(
        self, df: pd.DataFrame, lookahead: int = 3
    ) -> List[OrderBlock]:
        if df is None or len(df) < lookahead + 5:
            return []

        blocks: List[OrderBlock] = []
        atr = df["ATR_14"] if "ATR_14" in df.columns else None

        for i in range(2, len(df) - lookahead):
            row = df.iloc[i]
            impulse_close = float(df["close"].iloc[i + lookahead])
            impulse_open = float(df["open"].iloc[i + 1])

            # bullish OB = last bearish candle before upward displacement
            if row["close"] < row["open"]:
                displacement = impulse_close - impulse_open
                threshold = float((atr.iloc[i] if atr is not None else 0) or 0) * 0.5
                if displacement > max(0.0, threshold):
                    high = float(row["high"])
                    low = float(row["low"])
                    mitigated = self._is_zone_mitigated(df, i + 1, low, high)
                    blocks.append(
                        OrderBlock(
                            direction="bullish",
                            index=i,
                            high=high,
                            low=low,
                            mitigated=mitigated,
                        )
                    )

            # bearish OB = last bullish candle before downward displacement
            if row["close"] > row["open"]:
                displacement = impulse_open - impulse_close
                threshold = float((atr.iloc[i] if atr is not None else 0) or 0) * 0.5
                if displacement > max(0.0, threshold):
                    high = float(row["high"])
                    low = float(row["low"])
                    mitigated = self._is_zone_mitigated(df, i + 1, low, high)
                    blocks.append(
                        OrderBlock(
                            direction="bearish",
                            index=i,
                            high=high,
                            low=low,
                            mitigated=mitigated,
                        )
                    )

        return blocks

    def detect_fvgs(self, df: pd.DataFrame) -> List[FairValueGap]:
        if df is None or len(df) < 5:
            return []

        fvgs: List[FairValueGap] = []
        for i in range(0, len(df) - 2):
            c0 = df.iloc[i]
            c2 = df.iloc[i + 2]

            # bullish FVG: c2 low > c0 high
            if c2["low"] > c0["high"]:
                top = float(c2["low"])
                bottom = float(c0["high"])
                mitigated = self._is_gap_mitigated(df, i + 3, bottom, top, bullish=True)
                fvgs.append(
                    FairValueGap(
                        direction="bullish",
                        index=i + 1,
                        top=top,
                        bottom=bottom,
                        mitigated=mitigated,
                    )
                )

            # bearish FVG: c2 high < c0 low
            if c2["high"] < c0["low"]:
                top = float(c0["low"])
                bottom = float(c2["high"])
                mitigated = self._is_gap_mitigated(
                    df, i + 3, bottom, top, bullish=False
                )
                fvgs.append(
                    FairValueGap(
                        direction="bearish",
                        index=i + 1,
                        top=top,
                        bottom=bottom,
                        mitigated=mitigated,
                    )
                )

        return fvgs

    def _is_zone_mitigated(
        self, df: pd.DataFrame, start_idx: int, zone_low: float, zone_high: float
    ) -> bool:
        for j in range(start_idx, len(df)):
            r = df.iloc[j]
            if r["low"] <= zone_high and r["high"] >= zone_low and zone_low <= r["close"] <= zone_high:
                return True
        return False

    def _is_gap_mitigated(
        self,
        df: pd.DataFrame,
        start_idx: int,
        gap_bottom: float,
        gap_top: float,
        bullish: bool,
    ) -> bool:
        for j in range(start_idx, len(df)):
            r = df.iloc[j]
            # gap considered mitigated when price fully trades into opposite side
            if bullish and r["low"] <= gap_bottom:
                return True
            if not bullish and r["high"] >= gap_top:
                return True
        return False

    def nearest_unmitigated_ob(
        self, blocks: List[OrderBlock], close: float, direction: str
    ) -> Optional[OrderBlock]:
        cands = [
            b
            for b in blocks
            if not b.mitigated and b.direction == ("bullish" if direction == "LONG" else "bearish")
        ]
        if not cands:
            return None
        return min(cands, key=lambda b: min(abs(b.low - close), abs(b.high - close)))

    def has_fvg_overlap(self, block: OrderBlock, fvgs: List[FairValueGap]) -> bool:
        for g in fvgs:
            if g.mitigated:
                continue
            if block.direction != g.direction:
                continue
            overlap = max(0.0, min(block.high, g.top) - max(block.low, g.bottom))
            if overlap > 0:
                return True
        return False

