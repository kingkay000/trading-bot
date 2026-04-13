import os
import sys

import pandas as pd

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.append(root)

from modules.market_structure_engine import MarketStructureEngine
from modules.liquidity_map_engine import LiquidityMapEngine
from modules.trade_blueprint_engine import TradeBlueprintEngine


def _sample_df() -> pd.DataFrame:
    rows = []
    price = 1.1000
    for i in range(60):
        drift = 0.001 if i % 5 else -0.0005
        o = price
        c = price + drift
        h = max(o, c) + 0.0008
        l = min(o, c) - 0.0008
        v = 100 + i
        rows.append({"open": o, "high": h, "low": l, "close": c, "volume": v})
        price = c
    return pd.DataFrame(rows)


def test_market_structure_engine_returns_fields():
    df = _sample_df()
    eng = MarketStructureEngine()
    out = eng.analyze(df)
    assert out.macro_trend in ("BULLISH", "BEARISH", "RANGING")
    assert out.micro_trend in ("BULLISH", "BEARISH", "RANGING")
    assert isinstance(out.last_swing_highs, list)
    assert isinstance(out.last_swing_lows, list)


def test_liquidity_and_blueprint_engines():
    df = _sample_df()
    ms = MarketStructureEngine().analyze(df)
    from modules.market_structure_engine import SwingPoint

    highs = [SwingPoint(index=x["index"], price=x["price"], kind="SH") for x in ms.last_swing_highs]
    lows = [SwingPoint(index=x["index"], price=x["price"], kind="SL") for x in ms.last_swing_lows]
    lm = LiquidityMapEngine().analyze(df, highs, lows)
    assert lm.sweep_side in ("BUY_SIDE", "SELL_SIDE", "NONE")

    bp = TradeBlueprintEngine(min_rr=1.5).build(
        direction="LONG",
        close=float(df["close"].iloc[-1]),
        sr_levels=[float(df["high"].max()) + 0.001, float(df["high"].max()) + 0.002],
        market_structure=ms.to_dict(),
    )
    assert bp.direction in ("LONG", "SHORT", "NONE")
