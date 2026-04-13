import os
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.append(root)

from modules.bridge_payload import build_signal_payload, normalize_direction


class _Sig:
    symbol = "EURUSD"
    signal = "long"
    entry_price = 1.101
    stop_loss = 1.098
    take_profit_1 = 1.106
    confidence = 88
    reasoning = "test"


def test_normalize_direction():
    assert normalize_direction("LONG") == "BUY"
    assert normalize_direction("BUY") == "BUY"
    assert normalize_direction("short") == "SELL"
    assert normalize_direction("SELL") == "SELL"
    assert normalize_direction("x") == "HOLD"


def test_build_signal_payload_keys():
    payload = build_signal_payload(_Sig())
    assert payload["symbol"] == "EURUSD"
    assert payload["direction"] == "BUY"
    assert "entry_price" in payload
    assert "stop_loss" in payload
    assert "take_profit" in payload
    assert "timestamp" in payload
