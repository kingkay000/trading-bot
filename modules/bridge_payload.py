"""
modules/bridge_payload.py
─────────────────────────────────────────────────────────────────────────────
Builds execution-bridge payloads with backward-compatible fields for MQL5 EAs.
"""

from typing import Any, Dict, Optional
import time
import uuid


def normalize_direction(value: Any) -> str:
    raw = str(value or "").upper().strip()
    if raw in ("BUY", "LONG"):
        return "BUY"
    if raw in ("SELL", "SHORT"):
        return "SELL"
    return "HOLD"


def build_signal_payload(signal: Any, fundamental_context: Optional[Any] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "contract_version": "fxguru-v1",
        "signal_uuid": str(getattr(signal, "signal_uuid", "") or uuid.uuid4().hex),
        "symbol": str(getattr(signal, "symbol", "") or ""),
        "direction": normalize_direction(getattr(signal, "signal", "HOLD")),
        "entry_price": float(getattr(signal, "entry_price", 0.0) or 0.0),
        "stop_loss": float(getattr(signal, "stop_loss", 0.0) or 0.0),
        "take_profit": float(getattr(signal, "take_profit_1", 0.0) or 0.0),
        "confidence": float(getattr(signal, "confidence", 0.0) or 0.0),
        "reasoning": str(getattr(signal, "reasoning", "") or ""),
        "timestamp": float(time.time()),
    }
    if fundamental_context is not None:
        payload.update(
            {
                "fundamental_rating": int(getattr(fundamental_context, "fundamental_rating", 0) or 0),
                "fundamental_conviction": str(
                    getattr(fundamental_context, "fundamental_conviction", "weak") or "weak"
                ),
                "fundamental_note": str(getattr(fundamental_context, "fundamental_note", "") or ""),
            }
        )
    return payload
