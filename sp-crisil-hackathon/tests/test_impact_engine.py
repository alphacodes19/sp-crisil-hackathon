import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from src.entities.resolver import EntityMatch
from src.events.classifier import EVENT_TYPES, UNCLASSIFIED
from src.impact.impact_engine import (
    EVENT_SEVERITY,
    FACTOR_WEIGHTS,
    UNCLASSIFIED_SEVERITY_FALLBACK,
    compute_impact,
)


def _entity(match_type="ticker", ticker="TSLA"):
    return EntityMatch(canonical_name=f"{ticker} Inc.", ticker=ticker, matched_alias=ticker, match_type=match_type)


# --- severity lookup covers the whole taxonomy ------------------------------


def test_severity_lookup_covers_all_12_classes():
    assert set(EVENT_SEVERITY.keys()) == set(EVENT_TYPES)
    for severity in EVENT_SEVERITY.values():
        assert 0.0 <= severity <= 1.0


def test_factor_weights_match_master_context_and_sum_to_one():
    assert FACTOR_WEIGHTS == {
        "event_severity": 0.30,
        "market_relevance": 0.25,
        "novelty": 0.15,
        "velocity": 0.15,
        "corroboration": 0.15,
    }
    assert abs(sum(FACTOR_WEIGHTS.values()) - 1.0) < 1e-9


# --- bounded score / breakdown -----------------------------------------------


def test_score_is_within_1_to_10_for_extreme_inputs():
    high = compute_impact("GEOPOLITICAL", [_entity()], novelty=1.0, velocity=1.0, corroboration=1.0)
    low = compute_impact("PRODUCT_TECH", [], novelty=0.0, velocity=0.0, corroboration=0.0)
    assert 1.0 <= low.score <= 10.0
    assert 1.0 <= high.score <= 10.0
    assert high.score > low.score


def test_breakdown_fields_match_inputs_and_lookup():
    result = compute_impact("CREDIT", [_entity()], novelty=0.5, velocity=0.5, corroboration=0.5)
    assert result.breakdown.event_severity == EVENT_SEVERITY["CREDIT"]
    assert result.breakdown.novelty == 0.5
    assert result.breakdown.velocity == 0.5
    assert result.breakdown.corroboration == 0.5
    assert 0.0 <= result.breakdown.market_relevance <= 1.0


def test_method_and_calibration_metadata_are_explicit():
    result = compute_impact("CREDIT", [_entity()], novelty=0.5, velocity=0.5, corroboration=0.5)
    assert result.method == "heuristic_v1"
    assert result.is_calibrated is False


# --- UNCLASSIFIED fallback ----------------------------------------------------


def test_unclassified_event_uses_conservative_severity_fallback():
    result = compute_impact(UNCLASSIFIED, [_entity()], novelty=0.5, velocity=0.5, corroboration=0.5)
    assert result.breakdown.event_severity == UNCLASSIFIED_SEVERITY_FALLBACK
    assert result.breakdown.event_severity < min(EVENT_SEVERITY.values())  # deliberately low, not average


def test_unrecognized_event_type_also_falls_back_safely():
    # Defensive: even a totally unexpected string must not crash or
    # silently pick an arbitrary high severity.
    result = compute_impact("SOME_FUTURE_CLASS_NOT_YET_ADDED", [_entity()], novelty=0.5, velocity=0.5, corroboration=0.5)
    assert result.breakdown.event_severity == UNCLASSIFIED_SEVERITY_FALLBACK


# --- market relevance from entities ------------------------------------------


def test_no_entities_means_zero_market_relevance():
    result = compute_impact("CREDIT", [], novelty=0.5, velocity=0.5, corroboration=0.5)
    assert result.breakdown.market_relevance == 0.0


def test_ticker_match_has_higher_relevance_than_fuzzy_match():
    ticker_result = compute_impact("CREDIT", [_entity(match_type="ticker")], novelty=0.5, velocity=0.5, corroboration=0.5)
    fuzzy_result = compute_impact("CREDIT", [_entity(match_type="fuzzy")], novelty=0.5, velocity=0.5, corroboration=0.5)
    assert ticker_result.breakdown.market_relevance > fuzzy_result.breakdown.market_relevance


def test_multiple_entities_get_a_small_bounded_bonus_but_never_exceed_one():
    single = compute_impact("CREDIT", [_entity(ticker="TSLA")], novelty=0.5, velocity=0.5, corroboration=0.5)
    multi = compute_impact(
        "CREDIT", [_entity(ticker="TSLA"), _entity(ticker="AAPL")], novelty=0.5, velocity=0.5, corroboration=0.5
    )
    assert multi.breakdown.market_relevance >= single.breakdown.market_relevance
    assert multi.breakdown.market_relevance <= 1.0


# --- sentiment is never an input ---------------------------------------------


def test_compute_impact_has_no_sentiment_parameter():
    import inspect

    sig = inspect.signature(compute_impact)
    assert "sentiment" not in sig.parameters


# --- input validation ---------------------------------------------------------


@pytest.mark.parametrize("bad_value", [-0.01, 1.01, 2.0, -5.0])
def test_out_of_range_novelty_raises(bad_value):
    with pytest.raises(ValueError):
        compute_impact("CREDIT", [_entity()], novelty=bad_value, velocity=0.5, corroboration=0.5)


@pytest.mark.parametrize("bad_value", [-0.01, 1.01])
def test_out_of_range_velocity_or_corroboration_raises(bad_value):
    with pytest.raises(ValueError):
        compute_impact("CREDIT", [_entity()], novelty=0.5, velocity=bad_value, corroboration=0.5)
    with pytest.raises(ValueError):
        compute_impact("CREDIT", [_entity()], novelty=0.5, velocity=0.5, corroboration=bad_value)


def test_boundary_values_zero_and_one_are_accepted():
    result = compute_impact("CREDIT", [_entity()], novelty=0.0, velocity=1.0, corroboration=0.0)
    assert 1.0 <= result.score <= 10.0


# --- documentation claims (Milestone 7 hardening, part C) -------------------


def test_market_relevance_is_documented_as_an_entity_linking_proxy_not_materiality():
    import src.impact.impact_engine as impact_engine_module

    docstring = impact_engine_module._market_relevance.__doc__ or ""
    assert "materiality" in docstring.lower()
    assert "entity-linking" in docstring.lower() or "entity linking" in docstring.lower()


def test_impact_result_is_explicitly_uncalibrated_heuristic():
    result = compute_impact("CREDIT", [_entity()], novelty=0.5, velocity=0.5, corroboration=0.5)
    assert result.is_calibrated is False
    assert result.method == "heuristic_v1"
    # Same two entities, same confidence - the score cannot and does not
    # distinguish "systemically important" from "thinly traded"; this
    # factor is identical either way, by design (see module docs).
    small_issuer = compute_impact("CREDIT", [_entity(ticker="SMALLCO")], novelty=0.5, velocity=0.5, corroboration=0.5)
    big_issuer = compute_impact("CREDIT", [_entity(ticker="BIGCO")], novelty=0.5, velocity=0.5, corroboration=0.5)
    assert small_issuer.breakdown.market_relevance == big_issuer.breakdown.market_relevance
