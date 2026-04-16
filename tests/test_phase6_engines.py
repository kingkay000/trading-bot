import os
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.append(root)

from modules.portfolio_risk_allocator import PortfolioRiskAllocator
from modules.probability_engine import ProbabilityEngine
from modules.signal_contract_guard import SignalContractGuard


class _Sig:
    signal = "BUY"
    entry_price = 1.1000
    stop_loss = 1.0990
    take_profit_1 = 1.1020
    risk_reward_ratio = 2.0


class _Bp:
    direction = "LONG"


def test_probability_engine_output_ranges():
    p = ProbabilityEngine().assess(score=80, rr=2.0, regime="TRENDING")
    assert 0.0 < p.p_win < 1.0
    assert p.expected_value > -1.0


def test_portfolio_allocator_blocks_quote_overexposure():
    alloc = PortfolioRiskAllocator(max_same_quote_positions=1)
    open_pos = {"EURUSD": object()}
    res = alloc.approve("GBPUSD", open_pos)
    assert not res.approved


def test_signal_contract_guard_valid_buy():
    g = SignalContractGuard().validate(_Sig(), _Bp(), expected_direction="BUY", min_rr=1.5)
    assert g.approved
