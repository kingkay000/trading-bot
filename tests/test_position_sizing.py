import os
import sys

# Add project root to sys.path for importing project modules
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.append(root)

from utils.helpers import calc_position_size, contract_size_for_symbol


def test_calc_position_size_returns_base_units_for_fx():
    units = calc_position_size(
        symbol="EURUSD",
        account_balance=10_000,
        risk_pct=0.015,
        entry_price=1.1729,
        stop_loss_price=1.1691,
    )
    # 150 / 0.0038 = 39,473.6842 units
    assert abs(units - 39473.684211) < 0.01


def test_contract_size_for_fx_and_metal():
    assert contract_size_for_symbol("EURUSD") == 100000
    assert contract_size_for_symbol("XAUUSD") == 100
