"""
modules/portfolio_risk_allocator.py
─────────────────────────────────────────────────────────────────────────────
Simple portfolio-level risk allocator to cap correlated exposure.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class AllocationDecision:
    approved: bool
    reason: str = ""


class PortfolioRiskAllocator:
    def __init__(self, max_same_quote_positions: int = 2) -> None:
        self.max_same_quote_positions = max_same_quote_positions

    def approve(self, symbol: str, open_positions: Dict[str, Any]) -> AllocationDecision:
        sym = symbol.replace("/", "").upper()
        quote = sym[3:] if len(sym) >= 6 else sym[-3:]
        same_quote = 0
        for s in open_positions.keys():
            k = str(s).replace("/", "").upper()
            q = k[3:] if len(k) >= 6 else k[-3:]
            if q == quote:
                same_quote += 1
        if same_quote >= self.max_same_quote_positions:
            return AllocationDecision(
                approved=False,
                reason=f"Portfolio cap hit for quote {quote} ({same_quote} open)",
            )
        return AllocationDecision(approved=True)

