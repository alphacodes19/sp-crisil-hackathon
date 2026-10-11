"""
Explainable prototype impact engine.

Computes the 1-10 "potential market-impact severity" score using the
master context's initial weighted formula (Section 10):

    event severity           0.30
    market/entity relevance  0.25
    novelty                  0.15
    velocity                 0.15
    corroboration            0.15

Every factor is normalized to [0, 1] first; the weighted sum (itself
bounded to [0, 1], since the weights sum to 1.0 and every factor is
bounded) is then linearly mapped onto the project's canonical 1-10
scale:

    score = 1 + weighted_sum * 9

This is a documented, simple min-max-style mapping - not a learned or
calibrated transform, and NOT a probability of anything.

`ImpactResult.is_calibrated` is always False and `method` always
identifies this as `"heuristic_v1"`, so nothing downstream can mistake
it for more than a documented prototype heuristic. This applies to
every factor, not just the overall score: in particular,
`market_relevance` is a proxy for entity-linking confidence (how sure
the entity resolver is about WHICH company is involved), not a
validated measure of that company's economic or systemic materiality -
see `_market_relevance()` below for the full caveat.

Deliberately excluded from this score (per the master context):
- Portfolio exposure - belongs to Module B, not this portfolio-agnostic
  core risk engine.
- Sentiment - sentiment is direction/polarity, impact is severity;
  multiplying them together is explicitly disallowed. This module has
  no dependency on src/nlp/sentiment.py at all.
"""
from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel

from src.entities.resolver import EntityMatch
from src.events.classifier import EVENT_TYPES

# ASSUMPTION / NOT MEASURED: initial, documented severities per event
# type on a [0, 1] scale - a qualitative starting judgment of typical
# market-impact severity for each category, NOT calibrated against
# historical price reactions. Revisit with real evaluation data before
# claiming anything stronger (per the master context's evaluation
# discipline).
EVENT_SEVERITY: Dict[str, float] = {
    "GEOPOLITICAL": 0.85,
    "REGULATORY_LEGAL": 0.80,
    "MACRO_CENTRAL_BANK": 0.80,
    "CREDIT": 0.75,
    "CYBER_OPERATIONAL": 0.75,
    "RATING_ACTION": 0.70,
    "M&A": 0.60,
    "SUPPLY_CHAIN": 0.55,
    "COMMODITY": 0.50,
    "MANAGEMENT_GOVERNANCE": 0.45,
    "EARNINGS": 0.40,
    "PRODUCT_TECH": 0.30,
}
assert set(EVENT_SEVERITY) == set(EVENT_TYPES), "severity lookup must cover exactly the 12-class taxonomy"

# Deterministic fallback for UNCLASSIFIED (and, defensively, any other
# unrecognized label): default LOW rather than high, so an event we
# failed to classify never gets to inflate the score. An explicit,
# conservative design choice - not a measured value.
UNCLASSIFIED_SEVERITY_FALLBACK = 0.20

# IMPORTANT: "market/entity relevance" here is a PROXY FOR ENTITY-
# LINKING STRENGTH ONLY - i.e. how confidently the entity resolver
# matched a known ticker/company (ticker > exact_name > alias >
# fuzzy). It is NOT a validated measure of economic or systemic
# materiality (e.g. it says nothing about the entity's market cap,
# index weight, interconnectedness, or systemic importance). Two
# events with identical entity-match confidence can have wildly
# different real-world materiality; this factor cannot and does not
# distinguish them. Documented assumption, not a measured value.
_MATCH_TYPE_RELEVANCE_WEIGHT: Dict[str, float] = {
    "ticker": 1.0,
    "exact_name": 0.9,
    "alias": 0.75,
    "fuzzy": 0.5,
}
# Small bounded bonus when more than one entity is involved, reflecting
# broader potential relevance without fabricating a precise
# cross-entity weighting scheme.
_MULTI_ENTITY_RELEVANCE_BONUS = 0.10

FACTOR_WEIGHTS: Dict[str, float] = {
    "event_severity": 0.30,
    "market_relevance": 0.25,
    "novelty": 0.15,
    "velocity": 0.15,
    "corroboration": 0.15,
}
assert abs(sum(FACTOR_WEIGHTS.values()) - 1.0) < 1e-9

IMPACT_METHOD = "heuristic_v1"


class ImpactFactorBreakdown(BaseModel):
    event_severity: float
    market_relevance: float
    novelty: float
    velocity: float
    corroboration: float


class ImpactResult(BaseModel):
    score: float  # 1-10, potential market-impact SEVERITY, not a probability
    breakdown: ImpactFactorBreakdown
    method: str = IMPACT_METHOD
    is_calibrated: bool = False


def _event_severity(event_type: str) -> float:
    return EVENT_SEVERITY.get(event_type, UNCLASSIFIED_SEVERITY_FALLBACK)


def _market_relevance(entities: List[EntityMatch]) -> float:
    """
    See the _MATCH_TYPE_RELEVANCE_WEIGHT comment above: this returns an
    entity-linking-confidence proxy, not an economic/systemic
    materiality score. A "ticker" match on a small, thinly-traded
    issuer scores identically to a "ticker" match on a systemically
    important one - this function has no data to distinguish them.
    """
    if not entities:
        return 0.0  # can't claim market relevance to our tracked universe with no resolved entity
    best = max(_MATCH_TYPE_RELEVANCE_WEIGHT.get(e.match_type, 0.5) for e in entities)
    bonus = _MULTI_ENTITY_RELEVANCE_BONUS if len(entities) > 1 else 0.0
    return min(1.0, best + bonus)


def compute_impact(
    event_type: str,
    entities: List[EntityMatch],
    novelty: float,
    velocity: float,
    corroboration: float,
) -> ImpactResult:
    """
    Combine event_type + resolved entities (both already computed
    elsewhere - no duplicated logic) with the three event-resolution
    factors (also already computed elsewhere, by
    src/events/resolution.py) into one explainable 1-10 score.
    """
    for name, value in (("novelty", novelty), ("velocity", velocity), ("corroboration", corroboration)):
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"{name} must be in [0, 1], got {value}")

    severity = _event_severity(event_type)
    relevance = _market_relevance(entities)

    weighted_sum = (
        FACTOR_WEIGHTS["event_severity"] * severity
        + FACTOR_WEIGHTS["market_relevance"] * relevance
        + FACTOR_WEIGHTS["novelty"] * novelty
        + FACTOR_WEIGHTS["velocity"] * velocity
        + FACTOR_WEIGHTS["corroboration"] * corroboration
    )
    weighted_sum = max(0.0, min(1.0, weighted_sum))  # defensive clip - should already hold by construction

    score = 1.0 + weighted_sum * 9.0

    return ImpactResult(
        score=round(score, 3),
        breakdown=ImpactFactorBreakdown(
            event_severity=severity,
            market_relevance=relevance,
            novelty=novelty,
            velocity=velocity,
            corroboration=corroboration,
        ),
    )
