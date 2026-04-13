"""
modules/calibration.py
─────────────────────────────────────────────────────────────────────────────
Symbol-specific calibration resolver for deterministic thresholds.
"""

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class SymbolCalibration:
    min_rr_ratio: float
    executable_tiers: List[str]


class CalibrationResolver:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config or {}

    def for_symbol(self, symbol: str) -> SymbolCalibration:
        risk_cfg = self.config.get("risk", {})
        sig_cfg = self.config.get("signals", {})
        cal_cfg = (self.config.get("calibration", {}) or {}).get("symbols", {})
        sym_cfg = cal_cfg.get(symbol.upper(), {})
        return SymbolCalibration(
            min_rr_ratio=float(sym_cfg.get("min_rr_ratio", risk_cfg.get("min_rr_ratio", 1.5))),
            executable_tiers=list(
                sym_cfg.get(
                    "executable_tiers",
                    sig_cfg.get("executable_tiers", ["TIER_1", "TIER_2"]),
                )
            ),
        )

