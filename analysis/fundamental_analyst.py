"""
analysis/fundamental_analyst.py
────────────────────────────────
Fundamental analyst context + rating helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple


def _clamp_score(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


def _format_age(seconds: float) -> str:
    total = max(0, int(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    if hours > 0:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def compute_fundamental_rating(
    composite_score: float,
    signal_direction: Literal["BUY", "SELL"],
    sources_used: List[str],
    data_age_seconds: Dict[str, float],
    config: dict,
    components: Optional[Dict[str, float]] = None,
    caution_flag: bool = False,
) -> Tuple[Literal[-1, 0, 1], Literal["weak", "moderate", "strong"], str, float]:
    """
    Returns (fundamental_rating, fundamental_conviction, fundamental_note,
             oldest_source_age_seconds).
    """
    direction = signal_direction.upper()
    if direction not in {"BUY", "SELL"}:
        direction = "BUY"

    aligned_score = float(composite_score) if direction == "BUY" else float(composite_score) * -1.0

    rating_cfg = (config or {}).get("rating", {})
    approval_threshold = float(rating_cfg.get("approval_threshold", 0.30))
    strong_threshold = float(rating_cfg.get("strong_threshold", 0.60))
    moderate_threshold = float(rating_cfg.get("moderate_threshold", 0.30))

    if aligned_score >= approval_threshold:
        rating: Literal[-1, 0, 1] = 1
    elif aligned_score <= -approval_threshold:
        rating = -1
    else:
        rating = 0

    abs_score = abs(aligned_score)
    sources_count = len(sources_used)
    if abs_score >= strong_threshold and sources_count >= 2:
        conviction: Literal["weak", "moderate", "strong"] = "strong"
    elif abs_score >= moderate_threshold or sources_count >= 2:
        conviction = "moderate"
    else:
        conviction = "weak"

    oldest_source_age_seconds = (
        float(max(data_age_seconds.values())) if data_age_seconds else 0.0
    )

    if not sources_used:
        note = "Fundamental data unavailable. Technical signal unaffected."
        return rating, conviction, note, oldest_source_age_seconds

    if rating == 1 and conviction == "strong":
        opening = f"Fundamentals strongly support this {direction}."
    elif rating == 1:
        opening = f"Fundamentals moderately support this {direction}."
    elif rating == -1 and conviction == "strong":
        opening = f"Fundamentals strongly oppose this {direction}."
    elif rating == -1:
        opening = f"Fundamentals lean against this {direction}."
    else:
        opening = "Fundamentals are neutral on this setup."

    detail = ""
    comps = components or {}
    top_component = max(comps.items(), key=lambda x: abs(x[1]), default=None)
    if top_component and abs(float(top_component[1])) >= 0.2:
        component_labels = {
            "cb_bias": "CB policy bias",
            "calendar": "Economic calendar",
            "dxy": "DXY momentum",
            "sentiment": "Market sentiment",
        }
        label = component_labels.get(top_component[0], top_component[0])
        detail = f" Lead factor: {label}."

    caution = " High-impact event near." if caution_flag else ""
    age_text = _format_age(oldest_source_age_seconds)
    note = f"{opening}{detail}{caution} Data: {age_text}."
    if len(note) > 200:
        note = note[:197] + "..."
    return rating, conviction, note, oldest_source_age_seconds


@dataclass
class FundamentalContext:
    # Existing fields
    symbol: str
    composite_score: float
    direction_bias: Literal["bullish", "bearish", "neutral"]
    caution_flag: bool
    components: Dict[str, float]
    narrative: str
    data_age_seconds: Dict[str, float]
    sources_used: List[str]
    sources_failed: List[str]

    # New fields
    fundamental_rating: Literal[-1, 0, 1]
    fundamental_conviction: Literal["weak", "moderate", "strong"]
    fundamental_note: str
    oldest_source_age_seconds: float


class FundamentalAnalyst:
    """
    Lightweight fundamental context builder.

    Produces a composite score in [-1, +1] and its discretized alignment rating
    versus the current technical signal direction.
    """

    def __init__(self, config: dict):
        self.config = config or {}
        self.fa_cfg = self.config.get("fundamental_analysis", {})
        self.enabled = bool(self.fa_cfg.get("enabled", False))

    def analyse(
        self,
        symbol: str,
        signal_direction: str = "BUY",
        components: Optional[Dict[str, float]] = None,
        caution_flag: Optional[bool] = None,
        data_age_seconds: Optional[Dict[str, float]] = None,
        sources_used: Optional[List[str]] = None,
        sources_failed: Optional[List[str]] = None,
        narrative: str = "",
    ) -> FundamentalContext:
        components = components or {}
        sources_used = list(sources_used or [])
        sources_failed = list(sources_failed or [])
        data_age_seconds = data_age_seconds or {}
        caution = bool(self.fa_cfg.get("caution_flag_default", False)) if caution_flag is None else bool(caution_flag)

        composite_score = _clamp_score(sum(float(v) for v in components.values()))
        if composite_score > 0.05:
            direction_bias: Literal["bullish", "bearish", "neutral"] = "bullish"
        elif composite_score < -0.05:
            direction_bias = "bearish"
        else:
            direction_bias = "neutral"

        rating, conviction, note, oldest_age = compute_fundamental_rating(
            composite_score=composite_score,
            signal_direction=signal_direction.upper() if signal_direction else "BUY",
            sources_used=sources_used,
            data_age_seconds=data_age_seconds,
            config=self.fa_cfg,
            components=components,
            caution_flag=caution,
        )

        if not narrative:
            narrative = (
                "No fundamental sources configured."
                if not sources_used
                else f"Composite score {composite_score:+.2f} ({direction_bias})."
            )

        return FundamentalContext(
            symbol=symbol,
            composite_score=composite_score,
            direction_bias=direction_bias,
            caution_flag=caution,
            components=components,
            narrative=narrative,
            data_age_seconds=data_age_seconds,
            sources_used=sources_used,
            sources_failed=sources_failed,
            fundamental_rating=rating,
            fundamental_conviction=conviction,
            fundamental_note=note,
            oldest_source_age_seconds=oldest_age,
        )

