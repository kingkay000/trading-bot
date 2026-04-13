"""
modules/probability_engine.py
─────────────────────────────────────────────────────────────────────────────
Transforms deterministic score into calibrated probability and expectancy.
"""

from dataclasses import dataclass
from math import exp
from typing import Dict, Any


@dataclass
class ProbabilityAssessment:
    p_win: float
    expected_value: float
    confidence_band: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "p_win": round(self.p_win, 4),
            "expected_value": round(self.expected_value, 4),
            "confidence_band": round(self.confidence_band, 4),
        }


class ProbabilityEngine:
    def assess(self, score: int, rr: float, regime: str = "RANGING") -> ProbabilityAssessment:
        # Logistic mapping from confluence score.
        x = (float(score) - 50.0) / 12.0
        p = 1.0 / (1.0 + exp(-x))

        # Regime calibration adjustment.
        if regime == "HIGH_VOLATILITY":
            p *= 0.92
        elif regime == "TRENDING":
            p *= 1.03
        elif regime == "LOW_VOLATILITY":
            p *= 0.97
        p = min(0.99, max(0.01, p))

        ev = (p * rr) - (1.0 - p)
        band = max(0.03, 0.25 - (score / 500.0))
        return ProbabilityAssessment(p_win=p, expected_value=ev, confidence_band=band)

