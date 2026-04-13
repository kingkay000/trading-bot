import os
import sys
import time

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.append(root)

from fastapi.testclient import TestClient

from modules.execution_server import app, pending_signals, pending_position_events


def test_poll_contract_for_fxguru():
    pending_signals.clear()
    client = TestClient(app)
    payload = {
        "contract_version": "fxguru-v1",
        "signal_uuid": "sig-1",
        "symbol": "EURUSD",
        "direction": "BUY",
        "entry_price": 1.101,
        "stop_loss": 1.099,
        "take_profit": 1.105,
        "confidence": 91,
        "reasoning": "contract test",
        "timestamp": time.time(),
    }
    r = client.post(
        "/signals",
        json=payload,
        headers={"X-API-KEY": os.getenv("EXECUTION_BRIDGE_KEY", "default_secret_key")},
    )
    assert r.status_code == 200

    polled = client.get("/poll/EURUSD")
    assert polled.status_code == 200
    data = polled.json()
    assert data["status"] == "new_trade"
    assert data["direction"] in ("BUY", "SELL", "LONG", "SHORT")
    assert "contract_version" in data
    assert data["signal_uuid"] == "sig-1"
    assert "entry" in data and "sl" in data and "tp" in data

    # idempotency: duplicate uuid should not queue a new trade
    r2 = client.post(
        "/signals",
        json=payload,
        headers={"X-API-KEY": os.getenv("EXECUTION_BRIDGE_KEY", "default_secret_key")},
    )
    assert r2.status_code == 200
    assert r2.json().get("status") == "duplicate"


def test_position_event_poll_contract():
    pending_position_events.clear()
    client = TestClient(app)
    evt = {
        "symbol": "EURUSD",
        "event_type": "POSITION_CLOSED",
        "reason": "test",
        "exit_price": 1.1,
        "pnl": 10.5,
        "timestamp": time.time(),
    }
    r = client.post(
        "/position-events",
        json=evt,
        headers={"X-API-KEY": os.getenv("EXECUTION_BRIDGE_KEY", "default_secret_key")},
    )
    assert r.status_code == 200
    polled = client.get("/poll/position-events/EURUSD")
    assert polled.status_code == 200
    data = polled.json()
    assert data["status"] == "event"
    assert data["event_type"] == "POSITION_CLOSED"
