"""
modules/regime_engine.py
─────────────────────────────────────────────────────────────────────────────
Classifies market regime for adaptive scoring and gating.
"""

from dataclasses import dataclass
from typing import Dict, Any
import pandas as pd


@dataclass
class RegimeState:
    regime: str = "UNKNOWN"  # TRENDING | RANGING | HIGH_VOLATILITY | LOW_VOLATILITY
    atr_ratio: float = 1.0
    adx: float = 0.0
    range_ratio: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime": self.regime,
            "atr_ratio": round(self.atr_ratio, 4),
            "adx": round(self.adx, 4),
            "range_ratio": round(self.range_ratio, 4),
        }


class RegimeEngine:
    def classify(self, df: pd.DataFrame) -> RegimeState:
        if df is None or df.empty:
            return RegimeState()
        atr = float(df.get("ATR_14", pd.Series([0])).dropna().iloc[-1]) if "ATR_14" in df.columns and not df["ATR_14"].dropna().empty else 0.0
        atr_med = float(df["ATR_14"].dropna().median()) if "ATR_14" in df.columns and not df["ATR_14"].dropna().empty else max(atr, 1e-9)
        atr_ratio = (atr / atr_med) if atr_med > 0 else 1.0
        adx = float(df.get("ADX_14", pd.Series([0])).dropna().iloc[-1]) if "ADX_14" in df.columns and not df["ADX_14"].dropna().empty else 0.0
        window = df.tail(20)
        r = float(window["high"].max() - window["low"].min()) if not window.empty else 0.0
        close = float(df["close"].iloc[-1]) if not df.empty else 1.0
        range_ratio = (r / close) if close > 0 else 0.0

        regime = "RANGING"
        if atr_ratio > 1.5:
            regime = "HIGH_VOLATILITY"
        elif atr_ratio < 0.7:
            regime = "LOW_VOLATILITY"
        if adx > 25 and range_ratio > 0.01:
            regime = "TRENDING"
        return RegimeState(regime=regime, atr_ratio=atr_ratio, adx=adx, range_ratio=range_ratio)

