"""
modules/decision_trace_store.py
─────────────────────────────────────────────────────────────────────────────
Append-only JSONL decision trace logger for replay/audit.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict


class DecisionTraceStore:
    def __init__(self, path: str = "logs/decision_trace.jsonl") -> None:
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def append(self, payload: Dict[str, Any]) -> None:
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

