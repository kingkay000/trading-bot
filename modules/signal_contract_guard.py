"""
modules/signal_contract_guard.py
─────────────────────────────────────────────────────────────────────────────
Deterministic contract checks between AI output and deterministic blueprint.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class GuardResult:
    approved: bool
    reason: str = ""


class SignalContractGuard:
    def validate(
        self,
        signal: Any,
        blueprint: Any,
        expected_direction: str,
        min_rr: float,
    ) -> GuardResult:
        ai_dir = str(getattr(signal, "signal", "HOLD")).upper()
        if ai_dir not in ("BUY", "SELL", "HOLD"):
            return GuardResult(False, f"Invalid AI direction: {ai_dir}")

        if expected_direction in ("BUY", "SELL") and ai_dir in ("BUY", "SELL") and ai_dir != expected_direction:
            return GuardResult(False, f"Direction mismatch deterministic={expected_direction} ai={ai_dir}")

        entry = float(getattr(signal, "entry_price", 0.0) or 0.0)
        sl = float(getattr(signal, "stop_loss", 0.0) or 0.0)
        tp1 = float(getattr(signal, "take_profit_1", 0.0) or 0.0)
        rr = float(getattr(signal, "risk_reward_ratio", 0.0) or 0.0)

        if ai_dir == "BUY" and not (sl < entry < tp1):
            return GuardResult(False, "BUY geometry invalid (expected SL < entry < TP1)")
        if ai_dir == "SELL" and not (tp1 < entry < sl):
            return GuardResult(False, "SELL geometry invalid (expected TP1 < entry < SL)")
        if rr < float(min_rr):
            return GuardResult(False, f"AI RR below floor: {rr:.2f} < {min_rr:.2f}")

        # Non-fatal mismatch warning path: keep strict by rejecting for now.
        bp_dir = str(getattr(blueprint, "direction", "NONE"))
        if bp_dir == "LONG" and ai_dir == "SELL":
            return GuardResult(False, "Blueprint LONG but AI SELL")
        if bp_dir == "SHORT" and ai_dir == "BUY":
            return GuardResult(False, "Blueprint SHORT but AI BUY")

        return GuardResult(True, "")

