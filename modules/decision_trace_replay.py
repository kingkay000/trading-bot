"""
modules/decision_trace_replay.py
─────────────────────────────────────────────────────────────────────────────
Simple replay/forensics utility for decision trace JSONL files.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Dict, Any


def summarize_trace(path: str) -> Dict[str, Any]:
    stages = Counter()
    symbols = Counter()
    total = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            total += 1
            stages[row.get("stage", "unknown")] += 1
            symbols[row.get("symbol", "unknown")] += 1
    return {
        "total": total,
        "stages": dict(stages),
        "symbols": dict(symbols),
    }

