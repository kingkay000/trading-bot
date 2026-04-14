import os
import sys

import pandas as pd

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.append(root)

from modules.ob_fvg_engine import OBFVGEngine


def test_detects_bullish_fvg():
    df = pd.DataFrame(
        [
            {"open": 1.00, "high": 1.02, "low": 0.99, "close": 1.01},
            {"open": 1.01, "high": 1.04, "low": 1.00, "close": 1.03},
            {"open": 1.05, "high": 1.08, "low": 1.05, "close": 1.07},  # gap above c0 high
            {"open": 1.07, "high": 1.10, "low": 1.06, "close": 1.09},
        ]
    )
    eng = OBFVGEngine()
    fvgs = eng.detect_fvgs(df)
    assert any(g.direction == "bullish" and not g.mitigated for g in fvgs)


def test_detects_order_block():
    df = pd.DataFrame(
        [
            {"open": 1.10, "high": 1.11, "low": 1.09, "close": 1.095, "ATR_14": 0.01},
            {"open": 1.095, "high": 1.10, "low": 1.08, "close": 1.085, "ATR_14": 0.01},
            {"open": 1.09, "high": 1.095, "low": 1.08, "close": 1.082, "ATR_14": 0.01},  # bearish candle
            {"open": 1.083, "high": 1.12, "low": 1.082, "close": 1.115, "ATR_14": 0.01},
            {"open": 1.115, "high": 1.14, "low": 1.11, "close": 1.135, "ATR_14": 0.01},
            {"open": 1.135, "high": 1.16, "low": 1.13, "close": 1.155, "ATR_14": 0.01},
            {"open": 1.155, "high": 1.17, "low": 1.14, "close": 1.145, "ATR_14": 0.01},
        ]
    )
    eng = OBFVGEngine()
    obs = eng.detect_order_blocks(df)
    assert any(b.direction == "bullish" for b in obs)
