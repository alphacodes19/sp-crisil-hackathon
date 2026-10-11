"""
End-to-end test for the synthetic high-impact demo fixture
(data/replay/synthetic/high_impact_demo.jsonl). See
src/ingestion/synthetic_demo.py for what this fixture is and why it
exists; it is entirely fictional and is never part of the default
GDELT/SEC replay pipeline.

This test runs the REAL EntityResolver, EventResolver, and
compute_impact() - the only injected piece is sentiment (fully
irrelevant to the impact score, which has no sentiment dependency at
all; see test_impact_engine.py::test_compute_impact_has_no_sentiment_parameter).
The event classifier doesn't need a zero-shot stub either: the
fixture's text contains unambiguous REGULATORY_LEGAL keyword rule
phrases, so classification resolves deterministically via rules alone.
The resulting impact score is asserted to exceed 7 as an OUTCOME of
real factor values, not forced/hardcoded anywhere.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from src.entities.resolver import EntityResolver
from src.events.classifier import EventClassifier
from src.events.resolution import EventResolver
from src.impact.signal_builder import process_documents
from src.ingestion.synthetic_demo import load_replay as load_synthetic_replay
from src.nlp.sentiment import SentimentAnalyzer


def _fake_sentiment_infer(text):
    return [{"label": "negative", "score": 0.7}, {"label": "neutral", "score": 0.2}, {"label": "positive", "score": 0.1}]


def test_fixture_is_clearly_labeled_synthetic_not_real():
    docs = load_synthetic_replay()
    assert len(docs) >= 4  # "multiple related reports"
    for doc in docs:
        assert doc.source_name == "SYNTHETIC_DEMO_FIXTURE"
        assert ".test" in (doc.url or "")  # IANA-reserved, never a real domain
        assert "SYNTHETIC" in doc.text.upper()
        assert "FICTIONAL" in doc.text.upper()


def test_fixture_documents_come_from_distinct_simulated_publishers():
    docs = load_synthetic_replay()
    publishers = {d.publisher for d in docs}
    assert len(publishers) == len(docs)  # every document has its own distinct publisher


def test_synthetic_fixture_reaches_high_impact_through_the_real_heuristic():
    docs = load_synthetic_replay()

    signals, failures = process_documents(
        docs,
        resolver=EntityResolver(),
        sentiment_analyzer=SentimentAnalyzer(inference_fn=_fake_sentiment_infer),
        event_classifier=EventClassifier(enable_zero_shot=False),  # rules alone resolve it; no model needed
        event_resolver=EventResolver(),
    )

    assert failures == []
    assert len(signals) == len(docs)

    # All related reports must have been grouped by the EXISTING resolver
    # (no special-casing for this fixture) into exactly one resolved event.
    event_ids = {s.event_id for s in signals}
    assert len(event_ids) == 1, f"expected one resolved event, got {event_ids}"

    representative = signals[0]
    resolution = representative.event_resolution
    assert resolution is not None

    # Meaningful velocity/corroboration, as required - not just nonzero,
    # but driven by genuinely distinct publishers/documents.
    assert resolution.document_count == len(docs)
    assert resolution.distinct_publisher_count == len(docs)  # no duplicate URLs/publishers in this fixture
    assert resolution.corroboration > 0.5
    assert resolution.velocity > 0.5

    assert representative.event.event_type == "REGULATORY_LEGAL"
    assert representative.event.method == "rule"  # unambiguous keyword match, no model involved

    assert representative.impact_score is not None
    if representative.impact_score <= 7.0:
        pytest.fail(
            "Synthetic fixture did not exceed impact 7 as expected; "
            f"factor breakdown was: {representative.impact_breakdown}. "
            "This should be investigated by inspecting the factors above "
            "(event_severity, market_relevance, novelty, velocity, "
            "corroboration) rather than adjusting the score directly."
        )
    assert representative.impact_score > 7.0
    assert representative.impact_method == "heuristic_v1"
    assert representative.impact_breakdown.event_severity > 0.0
    assert representative.impact_breakdown.market_relevance == 1.0  # exact ticker match, single entity


def test_ordinary_gdelt_and_sec_replay_data_are_unaffected_by_the_fixture():
    """
    The synthetic fixture must never leak into the ordinary replay
    pipeline's default document set.
    """
    from src.ingestion.gdelt import load_replay as load_gdelt_replay
    from src.ingestion.sec_edgar import load_replay as load_sec_replay

    gdelt_docs = load_gdelt_replay()
    sec_docs = load_sec_replay()
    synthetic_ids = {d.document_id for d in load_synthetic_replay()}

    assert not synthetic_ids & {d.document_id for d in gdelt_docs}
    assert not synthetic_ids & {d.document_id for d in sec_docs}
    # Real replay counts (Milestones 1-2) are exactly as before: 3 + 3.
    assert len(gdelt_docs) == 3
    assert len(sec_docs) == 3
