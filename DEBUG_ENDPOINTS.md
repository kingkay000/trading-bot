# Debug Endpoint Usage Guide (Execution Bridge)

This guide explains how to safely use the **debug-only execution bridge endpoints** for EA/MT5 contract testing and demo workflows.

---

## 1) Purpose

The debug endpoints are designed to:

- Trigger test signals on-demand (without waiting for full strategy conditions).
- Inject simulated position lifecycle events for reconciliation tests.
- Inspect/reset in-memory bridge state between test runs.
- Validate expected payload contracts returned to the EA.

These endpoints are intended for **development/staging** and controlled demo sessions.

---

## 2) Safety Model

Debug endpoints are protected by **two controls**:

1. API key authentication (`X-API-KEY` header).
2. Explicit runtime opt-in with:

```bash
DEBUG_ENDPOINTS_ENABLED=true
```

If debug endpoints are disabled, the API returns HTTP `403`.

> Keep `DEBUG_ENDPOINTS_ENABLED=false` in production unless you are actively running supervised tests.

---

## 3) Endpoint Inventory

All routes below are served by the execution bridge service.

### 3.1 Inject a manual test signal

`POST /debug/signal/inject`

Use this to queue a direct `BUY`/`SELL`/`HOLD` signal for EA polling.

Example:

```bash
curl -X POST "$BASE_URL/debug/signal/inject" \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: $EXECUTION_BRIDGE_KEY" \
  -d '{
    "symbol": "EURUSD",
    "direction": "BUY",
    "entry": 1.1000,
    "sl": 1.0985,
    "tp": 1.1030,
    "confidence": 55.0,
    "reasoning": "debug injection"
  }'
```

---

### 3.2 Auto-generate immediate direction from recent candles

`POST /debug/signal/auto-direction/{symbol}?timeframe=1h&lookback=12`

This computes a quick direction from recent close movement and queues a test signal.

Example:

```bash
curl -X POST "$BASE_URL/debug/signal/auto-direction/EURUSD?timeframe=1h&lookback=12" \
  -H "X-API-KEY: $EXECUTION_BRIDGE_KEY"
```

Notes:

- Requires fresh market data in the store for the chosen symbol/timeframe.
- Returns HTTP `400` if there is insufficient fresh data.

---

### 3.3 Inject a position lifecycle event

`POST /debug/position-event/inject`

Use for EA reconciliation tests (e.g., closed/manual-close simulation).

Example:

```bash
curl -X POST "$BASE_URL/debug/position-event/inject" \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: $EXECUTION_BRIDGE_KEY" \
  -d '{
    "symbol": "EURUSD",
    "event_type": "POSITION_CLOSED",
    "reason": "manual close (debug)",
    "exit_price": 1.1012,
    "pnl": 12.5,
    "timestamp": 1713024000
  }'
```

---

### 3.4 Inspect bridge state

`GET /debug/bridge/state`

Returns pending symbols/events, analysis keys, id cache size, and data freshness.

Example:

```bash
curl "$BASE_URL/debug/bridge/state" \
  -H "X-API-KEY: $EXECUTION_BRIDGE_KEY"
```

---

### 3.5 Reset bridge state (test cleanup)

`POST /debug/bridge/reset`

Clears pending queues and in-memory idempotency state for a clean retest.

Example:

```bash
curl -X POST "$BASE_URL/debug/bridge/reset" \
  -H "X-API-KEY: $EXECUTION_BRIDGE_KEY"
```

---

### 3.6 Fetch contract self-test examples

`GET /debug/contract/self-test`

Returns reference payload shapes for:

- `GET /poll/{symbol}` response contract
- `GET /poll/position-events/{symbol}` response contract

Example:

```bash
curl "$BASE_URL/debug/contract/self-test" \
  -H "X-API-KEY: $EXECUTION_BRIDGE_KEY"
```

---

## 4) Recommended End-to-End Test Flow

1. **Reset** state: `POST /debug/bridge/reset`
2. **Inject** test signal (manual or auto-direction).
3. EA calls `GET /poll/{symbol}` and executes when `status=new_trade`.
4. **Inject** a close event via `POST /debug/position-event/inject`.
5. EA calls `GET /poll/position-events/{symbol}` and reconciles local state.
6. Optionally verify queue depletion with `GET /debug/bridge/state`.

---

## 5) Operational Practices

- Use a dedicated API key for test sessions where possible.
- Never expose debug endpoints without authentication.
- Keep a written test script/checklist so demo operators follow repeatable steps.
- Always reset bridge state before and after scripted integration tests.

---

## 6) Troubleshooting

- **403 (disabled):** `DEBUG_ENDPOINTS_ENABLED` is not set to true.
- **403 (auth):** missing/wrong `X-API-KEY`.
- **400 on auto-direction:** stale or insufficient market data for selected timeframe.
- **EA gets `no_signal`:** signal likely already consumed by prior poll or not queued.
- **EA gets `expired`:** polled signal exceeded allowed age window.

