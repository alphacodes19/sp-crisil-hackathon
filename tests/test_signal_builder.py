import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from src.entities.resolver import EntityResolver
from src.events.classifier import EventClassifier
from src.events.resolution import EventResolver
from src.impact.signal_builder import (
    Signal,
    SignalBuildFailure,
    build_signal,
    process_documents,
    run_replay_pipeline,
)
from src.ingestion.gdelt import load_replay as load_gdelt_replay
from src.ingestion.sec_edgar import load_replay as load_sec_replay
from src.nlp.sentiment import SentimentAnalyzer
from src.utils.io import read_jsonl, write_jsonl
from src.utils.schema import UnifiedDocument


def _doc(document_id: str, title: str, text: str, url: str = "https://example.com/a") -> UnifiedDocument:
    return UnifiedDocument(
        document_id=document_id,
        source_type="news",
        source_name="GDELT",
        publisher="example.com",
        timestamp="2026-10-08T10:00:00Z",
        title=title,
        text=text,
        url=url,
    )


def _fixed_sentiment_analyzer(label="positive", pos=0.8, neu=0.15, neg=0.05):
    scores = {"positive": pos, "neutral": neu, "negative": neg}

    def _infer(text):
        return [{"label": k, "score": v} for k, v in scores.items()]

    return SentimentAnalyzer(inference_fn=_infer)


def _fixed_event_classifier(event_type="CREDIT", score=0.9):
    def _infer(text, candidate_labels):
        return {"labels": [event_type], "scores": [score]}

    # No keyword rules will fire on generic test text, so this always
    # falls through to the injected zero-shot stub.
    return EventClassifier(zero_shot_inference_fn=_infer)


def _resolver():
    return EntityResolver()


# --- signal construction & field semantics ----------------------------------


def test_build_signal_basic_fields():
    doc = _doc("doc-1", "Tesla update", "Tesla Inc. reported new developments today.")
    signal = build_signal(doc, _resolver(), _fixed_sentiment_analyzer(), _fixed_event_classifier())

    assert signal.signal_id == "signal-doc-1"
    assert signal.event_id == "provisional-event-doc-1"
    assert signal.document_id == "doc-1"
    assert signal.source.document_id == "doc-1"
    assert signal.source.source_name == "GDELT"
    assert signal.source.source_type == "news"
    assert signal.source.publisher == "example.com"
    assert signal.source.url == "https://example.com/a"
    assert signal.sentiment.label == "positive"
    assert signal.sentiment.score == pytest.approx(0.8 - 0.05)
    assert signal.sentiment.model_score == pytest.approx(0.8)
    assert signal.event.event_type == "CREDIT"
    assert signal.event.method == "zero_shot"
    assert signal.impact_score is None
    assert signal.evidence is None


def test_build_signal_multiple_entities():
    doc = _doc("doc-2", "Partnership", "Microsoft and NVIDIA announced a new AI partnership.")
    signal = build_signal(doc, _resolver(), _fixed_sentiment_analyzer(), _fixed_event_classifier())
    tickers = {e.ticker for e in signal.entities}
    assert tickers == {"MSFT", "NVDA"}
    assert signal.scope == "multi_entity"


def test_build_signal_single_entity_scope():
    doc = _doc("doc-3", "Tesla", "Tesla Inc. shares moved today.")
    signal = build_signal(doc, _resolver(), _fixed_sentiment_analyzer(), _fixed_event_classifier())
    assert len(signal.entities) == 1
    assert signal.entities[0].ticker == "TSLA"
    assert signal.scope == "single_entity"


def test_build_signal_no_entities_scope_unmapped():
    doc = _doc("doc-4", "Unrelated", "A local bakery won an award for its sourdough bread.")
    signal = build_signal(doc, _resolver(), _fixed_sentiment_analyzer(), _fixed_event_classifier())
    assert signal.entities == []
    assert signal.scope == "unmapped"


def test_signal_ids_are_distinguishable_from_document_id_and_each_other():
    doc = _doc("doc-5", "Tesla", "Tesla Inc. shares moved today.")
    signal = build_signal(doc, _resolver(), _fixed_sentiment_analyzer(), _fixed_event_classifier())
    assert signal.signal_id != signal.document_id
    assert signal.event_id != signal.document_id
    assert signal.signal_id != signal.event_id
    assert signal.document_id == "doc-5"


def test_sentiment_and_event_fields_match_component_outputs():
    doc = _doc("doc-6", "Tesla", "Tesla entered into a new revolving credit facility with its lenders.")
    # Rule-based path this time (no zero-shot needed - the keyword
    # "revolving credit" is a CREDIT rule).
    signal = build_signal(
        doc,
        _resolver(),
        _fixed_sentiment_analyzer(label="negative", pos=0.1, neu=0.1, neg=0.8),
        EventClassifier(enable_zero_shot=False),
    )
    assert signal.event.event_type == "CREDIT"
    assert signal.event.method == "rule"
    assert signal.event.classifier_score is None
    assert len(signal.event.matched_rules) >= 1
    assert signal.sentiment.label == "negative"
    assert signal.sentiment.score == pytest.approx(0.1 - 0.8)


# --- deterministic IDs / reproducibility ------------------------------------


def test_build_signal_is_deterministic_across_calls():
    doc = _doc("doc-7", "Tesla", "Tesla Inc. shares moved today.")
    s1 = build_signal(doc, _resolver(), _fixed_sentiment_analyzer(), _fixed_event_classifier())
    s2 = build_signal(doc, _resolver(), _fixed_sentiment_analyzer(), _fixed_event_classifier())
    assert s1.signal_id == s2.signal_id
    assert s1.event_id == s2.event_id


# --- batch orchestration / failure isolation --------------------------------


def test_process_documents_isolates_a_single_failure():
    good_doc = _doc("good-1", "Tesla", "Tesla Inc. shares moved today.")
    bad_doc = _doc("bad-1", "Bad", "This one will fail sentiment analysis.")

    def _raising_infer(text):
        if "fail" in text:
            raise RuntimeError("simulated model failure")
        return [{"label": "neutral", "score": 1.0}, {"label": "positive", "score": 0.0}, {"label": "negative", "score": 0.0}]

    flaky_analyzer = SentimentAnalyzer(inference_fn=_raising_infer)

    signals, failures = process_documents(
        [good_doc, bad_doc],
        resolver=_resolver(),
        sentiment_analyzer=flaky_analyzer,
        event_classifier=_fixed_event_classifier(),
    )

    assert len(signals) == 1
    assert signals[0].document_id == "good-1"
    assert len(failures) == 1
    assert isinstance(failures[0], SignalBuildFailure)
    assert failures[0].document_id == "bad-1"
    assert "simulated model failure" in failures[0].error


def test_process_documents_all_succeed_returns_no_failures():
    docs = [_doc("a", "Tesla", "Tesla Inc. shares moved today."), _doc("b", "Apple", "Apple Inc. reported results.")]
    signals, failures = process_documents(
        docs, resolver=_resolver(), sentiment_analyzer=_fixed_sentiment_analyzer(), event_classifier=_fixed_event_classifier()
    )
    assert len(signals) == 2
    assert failures == []


# --- JSONL persistence and reload -------------------------------------------


def test_signal_persists_and_reloads_via_existing_io_helpers(tmp_path):
    doc = _doc("doc-8", "Tesla", "Tesla Inc. shares moved today.")
    signal = build_signal(doc, _resolver(), _fixed_sentiment_analyzer(), _fixed_event_classifier())

    out_path = tmp_path / "signals.jsonl"
    write_jsonl(out_path, [signal])
    reloaded = list(read_jsonl(out_path, Signal))

    assert len(reloaded) == 1
    assert reloaded[0].signal_id == signal.signal_id
    assert reloaded[0].event_id == signal.event_id
    assert reloaded[0].entities[0].ticker == "TSLA"
    assert reloaded[0].impact_score is None


# --- end-to-end replay pipeline ----------------------------------------------


def test_run_replay_pipeline_writes_signals_for_all_real_replay_documents(tmp_path):
    out_path = tmp_path / "signals.jsonl"
    signals, failures = run_replay_pipeline(
        output_path=out_path,
        resolver=_resolver(),
        sentiment_analyzer=_fixed_sentiment_analyzer(),
        event_classifier=_fixed_event_classifier(),
    )

    expected_doc_count = len(list(load_gdelt_replay())) + len(list(load_sec_replay()))
    assert expected_doc_count >= 3  # sanity: replay data still present
    assert len(signals) == expected_doc_count
    assert failures == []

    reloaded = list(read_jsonl(out_path, Signal))
    assert len(reloaded) == expected_doc_count
    assert {s.document_id for s in reloaded} == {
        d.document_id for d in list(load_gdelt_replay()) + list(load_sec_replay())
    }


def test_run_replay_pipeline_is_reproducible_across_two_runs(tmp_path):
    out_path_1 = tmp_path / "run1.jsonl"
    out_path_2 = tmp_path / "run2.jsonl"

    signals_1, _ = run_replay_pipeline(
        output_path=out_path_1,
        resolver=_resolver(),
        sentiment_analyzer=_fixed_sentiment_analyzer(),
        event_classifier=_fixed_event_classifier(),
    )
    signals_2, _ = run_replay_pipeline(
        output_path=out_path_2,
        resolver=_resolver(),
        sentiment_analyzer=_fixed_sentiment_analyzer(),
        event_classifier=_fixed_event_classifier(),
    )

    ids_1 = sorted((s.signal_id, s.event_id) for s in signals_1)
    ids_2 = sorted((s.signal_id, s.event_id) for s in signals_2)
    assert ids_1 == ids_2


# =============================================================================
# Milestone 7: batch path is resolution- and impact-aware
# (build_signal() itself is untouched and unchanged - see tests above)
# =============================================================================


def test_batch_path_produces_real_resolved_event_id_not_provisional():
    doc = _doc("doc-9", "Tesla", "Tesla Inc. entered into a new credit facility with lenders.")
    signals, failures = process_documents(
        [doc],
        resolver=_resolver(),
        sentiment_analyzer=_fixed_sentiment_analyzer(),
        event_classifier=_fixed_event_classifier(),
        event_resolver=EventResolver(),
    )
    assert failures == []
    signal = signals[0]
    assert signal.event_id.startswith("resolved-event-")
    assert not signal.event_id.startswith("provisional-event-")


def test_batch_path_populates_impact_score_and_breakdown():
    doc = _doc("doc-10", "Tesla", "Tesla Inc. entered into a new credit facility with lenders.")
    signals, _ = process_documents(
        [doc],
        resolver=_resolver(),
        sentiment_analyzer=_fixed_sentiment_analyzer(),
        event_classifier=_fixed_event_classifier(),
        event_resolver=EventResolver(),
    )
    signal = signals[0]
    assert signal.impact_score is not None
    assert 1.0 <= signal.impact_score <= 10.0
    assert signal.impact_breakdown is not None
    assert signal.impact_method == "heuristic_v1"
    assert signal.evidence is not None
    assert signal.event_resolution is not None
    assert signal.event_resolution.resolved_event_id == signal.event_id


def test_batch_path_single_document_event_is_not_self_corroborated():
    doc = _doc("doc-11", "Tesla", "Tesla Inc. entered into a new credit facility with lenders.")
    signals, _ = process_documents(
        [doc],
        resolver=_resolver(),
        sentiment_analyzer=_fixed_sentiment_analyzer(),
        event_classifier=_fixed_event_classifier(),
        event_resolver=EventResolver(),
    )
    signal = signals[0]
    assert signal.event_resolution.document_count == 1
    assert signal.event_resolution.corroboration == 0.0
    assert "not independently corroborated" in signal.evidence


def test_batch_path_merges_two_related_documents_into_one_resolved_event():
    doc_a = _doc(
        "doc-12",
        "Tesla enters new credit facility",
        "Tesla Inc. entered into a new credit facility with lenders.",
    )
    doc_b = UnifiedDocument(
        document_id="doc-13",
        source_type="news",
        source_name="GDELT",
        publisher="bloomberg.com",
        timestamp="2026-10-08T14:00:00Z",  # a few hours after doc_a (same day as _doc's fixed timestamp)
        title="Tesla credit facility deal confirmed",
        text="Tesla has confirmed its new credit facility agreement with lenders.",
        url="https://bloomberg.com/doc-13",
    )

    signals, failures = process_documents(
        [doc_a, doc_b],
        resolver=_resolver(),
        sentiment_analyzer=_fixed_sentiment_analyzer(),
        event_classifier=_fixed_event_classifier(),
        event_resolver=EventResolver(),
    )
    assert failures == []
    by_id = {s.document_id: s for s in signals}
    assert by_id["doc-12"].event_id == by_id["doc-13"].event_id
    assert by_id["doc-12"].event_resolution.document_count == 2
    assert by_id["doc-12"].event_resolution.distinct_publisher_count == 2
    # Shared event-level factors must be identical across member signals.
    assert by_id["doc-12"].event_resolution.corroboration == by_id["doc-13"].event_resolution.corroboration
    # Original per-document provenance must still be preserved independently.
    assert by_id["doc-12"].source.url != by_id["doc-13"].source.url
    assert by_id["doc-12"].document_id != by_id["doc-13"].document_id


def test_batch_path_signal_with_enriched_fields_persists_and_reloads(tmp_path):
    doc = _doc("doc-14", "Tesla", "Tesla Inc. entered into a new credit facility with lenders.")
    signals, _ = process_documents(
        [doc],
        resolver=_resolver(),
        sentiment_analyzer=_fixed_sentiment_analyzer(),
        event_classifier=_fixed_event_classifier(),
        event_resolver=EventResolver(),
    )
    out_path = tmp_path / "signals.jsonl"
    write_jsonl(out_path, signals)
    reloaded = list(read_jsonl(out_path, Signal))

    assert len(reloaded) == 1
    assert reloaded[0].impact_score == signals[0].impact_score
    assert reloaded[0].event_resolution.resolved_event_id == signals[0].event_resolution.resolved_event_id
    assert reloaded[0].event_id.startswith("resolved-event-")


def test_run_replay_pipeline_default_event_resolver_produces_resolved_ids(tmp_path):
    out_path = tmp_path / "signals.jsonl"
    signals, failures = run_replay_pipeline(
        output_path=out_path,
        resolver=_resolver(),
        sentiment_analyzer=_fixed_sentiment_analyzer(),
        event_classifier=_fixed_event_classifier(),
        # event_resolver intentionally omitted - exercises the default EventResolver()
    )
    assert failures == []
    assert all(s.event_id.startswith("resolved-event-") for s in signals)
    assert all(s.impact_score is not None for s in signals)


def test_build_signal_path_remains_unaffected_by_milestone_7_changes():
    # build_signal() (single-document, no batch/resolution context) must
    # be byte-for-byte unchanged in behavior from Milestone 6.
    doc = _doc("doc-15", "Tesla", "Tesla Inc. shares moved today.")
    signal = build_signal(doc, _resolver(), _fixed_sentiment_analyzer(), _fixed_event_classifier())
    assert signal.event_id == "provisional-event-doc-15"
    assert signal.impact_score is None
    assert signal.impact_breakdown is None
    assert signal.impact_method is None
    assert signal.evidence is None
    assert signal.event_resolution is None
