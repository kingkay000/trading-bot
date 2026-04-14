# FxGuru2Link.mq5 Compatibility Notes (Phase 4)

This note confirms compatibility between Python bridge payloads and
`experts/FxGuru2Link.mq5` polling logic.

## What the EA expects

From `FxGuru2Link.mq5`:
- Polls `GET /poll/{symbol}` and triggers execution only when JSON contains
  `"status":"new_trade"` and a non-empty `"direction"` field.
- Reads direction with `GetJsonValue(result, "direction")`.
- Also reads event endpoint `GET /poll/position-events/{symbol}`.

## What Python serves

`modules/execution_server.py` currently responds with:
- `status`, `direction`, `entry`, `sl`, `tp` from pending signals.

`main.py` now uses `modules/bridge_payload.py` to normalize direction to
`BUY|SELL|HOLD` and preserve required fields:
- `symbol`, `direction`, `entry_price`, `stop_loss`, `take_profit`, ...
- Adds non-breaking `contract_version: "fxguru-v1"` for forward compatibility.
- Adds non-breaking `signal_uuid` for idempotent bridge handling.

## Compatibility guarantee

The direction field remains present and normalized to EA-friendly values.
No endpoint path changes were introduced.
Position-event endpoint behavior remains unchanged.
Additional fields are additive/non-breaking and ignored by legacy EA parser.

Therefore, FxGuru2Link's polling and execution trigger contract remains intact.
