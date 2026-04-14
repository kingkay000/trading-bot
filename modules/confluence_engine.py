"""
modules/confluence_engine.py
─────────────────────────────────────────────────────────────────────────────
Deterministic confluence scoring engine.

This module computes a rule-based 0–100 confluence score from already
calculated indicators/patterns so that AI can focus on judgement, not math.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import pandas as pd
from modules.liquidity_map_engine import LiquidityMapEngine
from modules.market_structure_engine import MarketStructureEngine
from modules.ob_fvg_engine import OBFVGEngine


@dataclass
class ConfluenceResult:
    direction: str = "NONE"  # LONG | SHORT | NONE
    score: int = 0
    tier: str = "NO_SIGNAL"  # TIER_1 | TIER_2 | TIER_3 | TIER_4 | NO_SIGNAL
    structure_score: int = 0
    liquidity_score: int = 0
    zone_score: int = 0
    momentum_score: int = 0
    volume_score: int = 0
    market_structure: Dict[str, Any] = None  # type: ignore[assignment]
    liquidity_map: Dict[str, Any] = None  # type: ignore[assignment]
    notes: List[str] = None  # type: ignore[assignment]

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["notes"] = self.notes or []
        payload["market_structure"] = self.market_structure or {}
        payload["liquidity_map"] = self.liquidity_map or {}
        return payload


class ConfluenceEngine:
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.structure_engine = MarketStructureEngine()
        self.liquidity_engine = LiquidityMapEngine()
        self.ob_fvg_engine = OBFVGEngine()

    def analyze(
        self,
        df: pd.DataFrame,
        patterns: List[Any],
        sr_levels: List[float],
    ) -> ConfluenceResult:
        if df is None or df.empty:
            return ConfluenceResult(notes=["No dataframe"])

        last = df.iloc[-1]
        notes: List[str] = []
        ms = self.structure_engine.analyze(df)
        swing_highs = ms.last_swing_highs or []
        swing_lows = ms.last_swing_lows or []
        # reconstruct lightweight swings for liquidity engine
        from modules.market_structure_engine import SwingPoint

        highs = [SwingPoint(index=s["index"], price=s["price"], kind="SH") for s in swing_highs]
        lows = [SwingPoint(index=s["index"], price=s["price"], kind="SL") for s in swing_lows]
        lm = self.liquidity_engine.analyze(df, highs, lows)

        # Direction vote using simple deterministic stack.
        long_votes = 0
        short_votes = 0
        if ms.macro_trend == "BULLISH":
            long_votes += 1
        elif ms.macro_trend == "BEARISH":
            short_votes += 1
        if ms.micro_trend == "BULLISH":
            long_votes += 1
        elif ms.micro_trend == "BEARISH":
            short_votes += 1
        if bool(last.get("is_macd_bullish", False)):
            long_votes += 1
        if bool(last.get("is_macd_bearish", False)):
            short_votes += 1
        if bool(last.get("is_liquidity_sweep_bullish", False)):
            long_votes += 1
        if bool(last.get("is_liquidity_sweep_bearish", False)):
            short_votes += 1

        direction = "LONG" if long_votes > short_votes else "SHORT" if short_votes > long_votes else "NONE"

        structure_score = self._score_structure(last, patterns, ms, direction, notes)
        liquidity_score = self._score_liquidity(last, sr_levels, lm, direction, notes)
        blocks = self.ob_fvg_engine.detect_order_blocks(df)
        fvgs = self.ob_fvg_engine.detect_fvgs(df)
        zone_score = self._score_zones(
            close=float(last.get("close", 0.0) or 0.0),
            blocks=blocks,
            fvgs=fvgs,
            direction=direction,
            notes=notes,
        )
        momentum_score = self._score_momentum(last, direction, notes)
        volume_score = self._score_volume(last, notes)

        total = structure_score + liquidity_score + zone_score + momentum_score + volume_score
        tier = self._tier(total)

        if direction == "NONE":
            tier = "NO_SIGNAL"
            total = 0
            notes.append("Directional bias unresolved")

        return ConfluenceResult(
            direction=direction,
            score=int(max(0, min(100, total))),
            tier=tier,
            structure_score=structure_score,
            liquidity_score=liquidity_score,
            zone_score=zone_score,
            momentum_score=momentum_score,
            volume_score=volume_score,
            market_structure=ms.to_dict(),
            liquidity_map=lm.to_dict(),
            notes=notes,
        )

    def _score_structure(
        self,
        last: pd.Series,
        patterns: List[Any],
        ms: Any,
        direction: str,
        notes: List[str],
    ) -> int:
        score = 0
        bullish_bos = any(getattr(p, "pattern_name", "") == "Break of Structure (Bullish)" for p in patterns)
        bearish_bos = any(getattr(p, "pattern_name", "") == "Break of Structure (Bearish)" for p in patterns)
        above_ema50 = bool(last.get("is_above_ema50", False))
        above_ema200 = bool(last.get("is_above_ema200", False))

        if direction == "LONG":
            if ms.macro_trend == "BULLISH":
                score += 5
            if ms.micro_trend == "BULLISH":
                score += 5
            if above_ema50:
                score += 5
            if bullish_bos:
                score += 10
        elif direction == "SHORT":
            if ms.macro_trend == "BEARISH":
                score += 5
            if ms.micro_trend == "BEARISH":
                score += 5
            if not above_ema50:
                score += 5
            if bearish_bos:
                score += 10

        if score > 0:
            notes.append(f"Structure score {score}/25")
        return min(score, 25)

    def _score_liquidity(
        self,
        last: pd.Series,
        sr_levels: List[float],
        lm: Any,
        direction: str,
        notes: List[str],
    ) -> int:
        score = 0
        close = float(last.get("close", 0.0) or 0.0)
        sweep_bull = bool(last.get("is_liquidity_sweep_bullish", False))
        sweep_bear = bool(last.get("is_liquidity_sweep_bearish", False))

        if direction == "LONG" and sweep_bull:
            score += 10
        if direction == "SHORT" and sweep_bear:
            score += 10
        if lm.sweep_detected:
            if direction == "LONG" and lm.sweep_side == "SELL_SIDE":
                score += 10
            if direction == "SHORT" and lm.sweep_side == "BUY_SIDE":
                score += 10
            notes.append(f"Liquidity sweep: {lm.sweep_side} @ {lm.sweep_index}")

        # proximity to nearest S/R level
        if sr_levels and close > 0:
            nearest = min(sr_levels, key=lambda x: abs(x - close))
            dist = abs(nearest - close) / close
            if dist <= 0.003:  # within 0.3%
                score += 5
                notes.append("Near structural level")

        return min(score, 25)

    def _score_zones(
        self,
        close: float,
        blocks: List[Any],
        fvgs: List[Any],
        direction: str,
        notes: List[str],
    ) -> int:
        if direction not in ("LONG", "SHORT") or close <= 0:
            return 0

        score = 0
        best = self.ob_fvg_engine.nearest_unmitigated_ob(blocks, close, direction)
        if best is None:
            notes.append("No unmitigated OB near price")
            return 0

        # Base OB quality
        score += 10
        dist = min(abs(best.low - close), abs(best.high - close)) / close
        if dist <= 0.003:
            score += 5
            notes.append("Price near unmitigated OB")

        if self.ob_fvg_engine.has_fvg_overlap(best, fvgs):
            score += 5
            notes.append("OB/FVG overlap detected")

        return min(score, 20)

    def _score_momentum(self, last: pd.Series, direction: str, notes: List[str]) -> int:
        score = 0
        rsi = float(last.get("RSI_14", 50.0) or 50.0)
        macd_bull = bool(last.get("is_macd_bullish", False))
        macd_bear = bool(last.get("is_macd_bearish", False))
        bull_div = bool(last.get("is_bullish_div", False))
        bear_div = bool(last.get("is_bearish_div", False))

        if direction == "LONG":
            if rsi < 35:
                score += 5
            if bull_div:
                score += 5
            if macd_bull:
                score += 5
        elif direction == "SHORT":
            if rsi > 65:
                score += 5
            if bear_div:
                score += 5
            if macd_bear:
                score += 5

        if bool(last.get("is_adx_trending", False)):
            score += 5
        if score:
            notes.append(f"Momentum score {score}/20")
        return min(score, 20)

    def _score_volume(self, last: pd.Series, notes: List[str]) -> int:
        score = 0
        if bool(last.get("is_high_volume", False)):
            score += 5
        vr = float(last.get("Volume_Ratio", 1.0) or 1.0)
        if vr >= 1.3:
            score += 5
        if score:
            notes.append(f"Volume score {score}/10")
        return min(score, 10)

    def _tier(self, score: int) -> str:
        if score >= 90:
            return "TIER_1"
        if score >= 75:
            return "TIER_2"
        if score >= 60:
            return "TIER_3"
        if score >= 40:
            return "TIER_4"
        return "NO_SIGNAL"
