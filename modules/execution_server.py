from fastapi import FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from typing import Dict, List, Optional
import uvicorn
import time
import os

app = FastAPI(title="Trading Bot Execution Bridge")

# Simple API Key security (could be expanded)
API_KEY = os.getenv("EXECUTION_BRIDGE_KEY", "default_secret_key")
api_key_header = APIKeyHeader(name="X-API-KEY")


async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(status_code=403, detail="Could not validate credentials")


class TradeSignal(BaseModel):
    symbol: str
    direction: str  # "long" | "short"
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    timestamp: float = time.time()


# Store pending signals in memory (simplification for now)
# In production, use Redis or a database
pending_signals: Dict[str, TradeSignal] = {}


@app.post("/signals")
async def post_signal(signal: TradeSignal, api_key: str = Security(get_api_key)):
    """Python bot posts a signal here."""
    pending_signals[signal.symbol] = signal
    return {"status": "queued", "symbol": signal.symbol}


@app.get("/poll/{symbol}")
async def poll_signal(symbol: str):
    """MQL5 EA polls this endpoint for a specific symbol."""
    signal = pending_signals.pop(symbol, None)
    if signal:
        # Check if signal is too old (e.g. > 5 minutes)
        if time.time() - signal.timestamp > 300:
            return {"status": "expired"}
        return {
            "status": "new_trade",
            "direction": signal.direction.upper(),
            "entry": signal.entry_price,
            "sl": signal.stop_loss,
            "tp": signal.take_profit,
        }
    return {"status": "no_signal"}


@app.get("/health")
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
