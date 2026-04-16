import json
import os
import sys
import tempfile

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.append(root)

from modules.calibration import CalibrationResolver
from modules.decision_trace_store import DecisionTraceStore


def test_calibration_resolver_symbol_override():
    cfg = {
        "risk": {"min_rr_ratio": 1.5},
        "signals": {"executable_tiers": ["TIER_1", "TIER_2"]},
        "calibration": {
            "symbols": {
                "XAUUSD": {"min_rr_ratio": 1.7, "executable_tiers": ["TIER_1"]},
            }
        },
    }
    r = CalibrationResolver(cfg)
    xau = r.for_symbol("XAUUSD")
    eur = r.for_symbol("EURUSD")
    assert xau.min_rr_ratio == 1.7
    assert xau.executable_tiers == ["TIER_1"]
    assert eur.min_rr_ratio == 1.5


def test_decision_trace_store_appends_jsonl():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "trace.jsonl")
        s = DecisionTraceStore(path)
        s.append({"symbol": "EURUSD", "stage": "test_stage"})
        with open(path, "r", encoding="utf-8") as f:
            line = f.readline().strip()
        row = json.loads(line)
        assert row["symbol"] == "EURUSD"
        assert row["stage"] == "test_stage"
        assert "ts" in row
