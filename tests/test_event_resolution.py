import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from src.entities.resolver import EntityMatch
from src.events.classifier import EventClassification
from src.events.resolution import (
    DocumentFeatures,
    EventResolver,
    NOVELTY_FLOOR,
    _EventHistorySignature,
    _compute_corroboration,
    _compute_novelty,
    _compute_velocity,
    _text_similarity,
)
from src.utils.schema import UnifiedDocument


def _doc(document_id, title, text, timestamp, url=None, publisher="pub-a"):
    return UnifiedDocument(
        document_id=document_id,
        source_type="news",
        source_name="GDELT",
        publisher=publisher,
        timestamp=timestamp,
        title=title,
        text=text,
        url=url or f"https://example.com/{document_id}",
    )


def _entity(ticker, canonical_name=None):
    return EntityMatch(
        canonical_name=canonical_name or f"{ticker} Inc.",
        ticker=ticker,
        matched_alias=ticker,
        match_type="ticker",
    )


def _event(event_type="CREDIT", score=None, method="rule"):
    return EventClassification(event_type=event_type, classifier_score=score, method=method, matched_rules=[])


def _features(document_id, title, text, timestamp, tickers, event_type="CREDIT", url=None, publisher="pub-a"):
    return DocumentFeatures(
        document=_doc(document_id, title, text, timestamp, url=url, publisher=publisher),
        entities=[_entity(t) for t in tickers],
        event_classification=_event(event_type=event_type),
    )


# --- merging: same event ----------------------------------------------------


def test_two_articles_describing_the_same_event_are_merged():
    f1 = _features(
        "d1",
        "Tesla enters new credit facility deal",
        "Tesla Inc. announced a new credit facility with lenders.",
        "2026-10-01T10:00:00Z",
        tickers=["TSLA"],
        event_type="CREDIT",
        publisher="reuters.com",
    )
    f2 = _features(
        "d2",
        "Tesla credit facility deal announced",
        "Tesla has entered into a new credit facility agreement.",
        "2026-10-01T14:00:00Z",
        tickers=["TSLA"],
        event_type="CREDIT",
        publisher="bloomberg.com",
    )
    resolver = EventResolver()
    resolved = resolver.resolve([f1, f2])

    assert resolved["d1"].resolved_event_id == resolved["d2"].resolved_event_id
    assert resolved["d1"].document_count == 2
    assert resolved["d1"].distinct_publisher_count == 2
    assert resolved["d1"].corroboration > 0.0


def test_two_different_events_same_company_are_not_merged():
    f1 = _features(
        "d1",
        "Tesla enters new credit facility deal",
        "Tesla Inc. announced a new credit facility with lenders.",
        "2026-01-01T10:00:00Z",
        tickers=["TSLA"],
        event_type="CREDIT",
    )
    f2 = _features(
        "d2",
        "Tesla faces safety probe over suspension",
        "Regulators opened a safety investigation into Tesla vehicles.",
        "2026-08-01T10:00:00Z",  # months later - well outside the time-gap window
        tickers=["TSLA"],
        event_type="REGULATORY_LEGAL",
    )
    resolver = EventResolver()
    resolved = resolver.resolve([f1, f2])

    assert resolved["d1"].resolved_event_id != resolved["d2"].resolved_event_id
    assert resolved["d1"].document_count == 1
    assert resolved["d2"].document_count == 1


def test_incompatible_event_types_are_not_merged_even_if_close_in_time():
    f1 = _features(
        "d1",
        "Tesla credit facility news",
        "Tesla Inc. entered into a new credit facility.",
        "2026-10-01T10:00:00Z",
        tickers=["TSLA"],
        event_type="CREDIT",
    )
    f2 = _features(
        "d2",
        "Tesla safety probe opened",
        "Regulators opened a safety investigation into Tesla vehicles.",
        "2026-10-01T11:00:00Z",  # 1 hour later, same entity - but different event type
        tickers=["TSLA"],
        event_type="REGULATORY_LEGAL",
    )
    resolver = EventResolver()
    resolved = resolver.resolve([f1, f2])
    assert resolved["d1"].resolved_event_id != resolved["d2"].resolved_event_id


def test_no_shared_entity_means_no_merge_even_with_similar_titles():
    f1 = _features(
        "d1", "Company announces credit facility", "Some company entered a credit facility.",
        "2026-10-01T10:00:00Z", tickers=["TSLA"], event_type="CREDIT",
    )
    f2 = _features(
        "d2", "Company announces credit facility", "Some other company entered a credit facility.",
        "2026-10-01T11:00:00Z", tickers=["AAPL"], event_type="CREDIT",
    )
    resolver = EventResolver()
    resolved = resolver.resolve([f1, f2])
    assert resolved["d1"].resolved_event_id != resolved["d2"].resolved_event_id


def test_unclassified_events_never_merge_with_anything():
    f1 = _features(
        "d1", "Tesla news item one", "Tesla did something notable today.",
        "2026-10-01T10:00:00Z", tickers=["TSLA"], event_type="UNCLASSIFIED",
    )
    f2 = _features(
        "d2", "Tesla news item one", "Tesla did something notable today.",
        "2026-10-01T10:30:00Z", tickers=["TSLA"], event_type="UNCLASSIFIED",
    )
    resolver = EventResolver()
    resolved = resolver.resolve([f1, f2])
    assert resolved["d1"].resolved_event_id != resolved["d2"].resolved_event_id
    assert resolved["d1"].novelty == 0.5  # neutral default, documented
    assert resolved["d2"].novelty == 0.5


# --- corroboration: duplicates must not inflate it --------------------------


def test_duplicate_url_articles_do_not_inflate_corroboration():
    same_url = "https://example.com/same-article"
    f1 = _features(
        "d1", "Tesla credit facility deal", "Tesla entered a new credit facility.",
        "2026-10-01T10:00:00Z", tickers=["TSLA"], event_type="CREDIT", url=same_url, publisher="reuters.com",
    )
    f2 = _features(
        "d2", "Tesla credit facility deal re-crawled", "Tesla entered a new credit facility.",
        "2026-10-01T10:05:00Z", tickers=["TSLA"], event_type="CREDIT", url=same_url, publisher="reuters.com",
    )
    resolver = EventResolver()
    resolved = resolver.resolve([f1, f2])

    # Merged (same event), but the duplicate URL must not count as a
    # second independent source.
    assert resolved["d1"].resolved_event_id == resolved["d2"].resolved_event_id
    assert resolved["d1"].document_count == 2
    assert resolved["d1"].distinct_source_count == 1
    assert resolved["d1"].distinct_publisher_count == 1
    assert resolved["d1"].corroboration == 0.0


def test_single_document_event_is_not_self_corroborated():
    f1 = _features(
        "d1", "Tesla credit facility deal", "Tesla entered a new credit facility.",
        "2026-10-01T10:00:00Z", tickers=["TSLA"], event_type="CREDIT",
    )
    resolver = EventResolver()
    resolved = resolver.resolve([f1])
    assert resolved["d1"].document_count == 1
    assert resolved["d1"].corroboration == 0.0
    assert resolved["d1"].velocity == 0.0


# --- novelty -----------------------------------------------------------------


def _history_entry(event_type, tickers, title, timestamp):
    return _EventHistorySignature(
        event_type=event_type, tickers=frozenset(tickers), representative_title=title, first_seen=timestamp
    )


from datetime import datetime, timedelta, timezone

_T0 = datetime(2026, 10, 1, 10, 0, 0, tzinfo=timezone.utc)
_LOOKBACK_30D = timedelta(hours=24 * 30)


def test_novelty_with_empty_history_is_high():
    novelty = _compute_novelty("CREDIT", frozenset({"TSLA"}), "Tesla credit facility deal", _T0, [], _LOOKBACK_30D)
    assert novelty == 1.0


def test_novelty_near_duplicate_title_gets_lower_novelty_than_different_title():
    history = [_history_entry("CREDIT", {"TSLA"}, "Tesla enters new credit facility deal", _T0)]

    near_duplicate = _compute_novelty(
        "CREDIT", frozenset({"TSLA"}), "Tesla enters new credit facility agreement", _T0 + timedelta(hours=1), history, _LOOKBACK_30D
    )
    genuinely_different = _compute_novelty(
        "CREDIT", frozenset({"TSLA"}), "Tesla defaults on loan covenant after missed payment", _T0 + timedelta(hours=1), history, _LOOKBACK_30D
    )

    assert near_duplicate < genuinely_different
    assert NOVELTY_FLOOR <= near_duplicate <= 1.0
    assert NOVELTY_FLOOR <= genuinely_different <= 1.0


def test_novelty_unaffected_by_unrelated_history():
    history = [_history_entry("EARNINGS", {"AAPL"}, "Apple reports quarterly earnings", _T0)]
    novelty = _compute_novelty(
        "CREDIT", frozenset({"TSLA"}), "Tesla credit facility deal", _T0 + timedelta(hours=1), history, _LOOKBACK_30D
    )
    assert novelty == 1.0  # no overlap in event_type or entity - not "previously observed" / relevant


def test_novelty_neutral_default_for_unclassified_or_no_entities_or_missing_title():
    assert _compute_novelty("UNCLASSIFIED", frozenset({"TSLA"}), "Tesla news", _T0, [], _LOOKBACK_30D) == 0.5
    assert _compute_novelty("CREDIT", frozenset(), "Tesla news", _T0, [], _LOOKBACK_30D) == 0.5
    assert _compute_novelty("CREDIT", frozenset({"TSLA"}), "   ", _T0, [], _LOOKBACK_30D) == 0.5
    assert _compute_novelty("CREDIT", frozenset({"TSLA"}), "", _T0, [], _LOOKBACK_30D) == 0.5


def test_novelty_respects_lookback_window_excludes_stale_prior_events():
    stale_history = [
        _history_entry("CREDIT", {"TSLA"}, "Tesla enters new credit facility deal", _T0 - timedelta(hours=24 * 60))
    ]  # 60 days before - outside the default 30-day lookback
    novelty = _compute_novelty(
        "CREDIT", frozenset({"TSLA"}), "Tesla enters new credit facility deal", _T0, stale_history, _LOOKBACK_30D
    )
    assert novelty == 1.0  # the only relevant-looking prior event is too old to count


def test_novelty_within_lookback_window_is_considered():
    recent_history = [
        _history_entry("CREDIT", {"TSLA"}, "Tesla enters new credit facility deal", _T0 - timedelta(hours=24 * 5))
    ]  # 5 days before - inside the default 30-day lookback
    novelty = _compute_novelty(
        "CREDIT", frozenset({"TSLA"}), "Tesla enters new credit facility deal", _T0, recent_history, _LOOKBACK_30D
    )
    assert novelty < 1.0  # near-identical title, within window -> reduced novelty


def test_novelty_ignores_prior_entries_with_missing_title_without_crashing():
    history = [
        _history_entry("CREDIT", {"TSLA"}, "", _T0),  # missing title - must be skipped, not crash
        _history_entry("CREDIT", {"TSLA"}, "Tesla enters new credit facility deal", _T0),
    ]
    novelty = _compute_novelty(
        "CREDIT", frozenset({"TSLA"}), "Tesla enters new credit facility agreement", _T0 + timedelta(hours=1), history, _LOOKBACK_30D
    )
    assert 0.0 <= novelty <= 1.0
    assert novelty < 1.0  # the real (non-empty-title) relevant prior entry still reduces it


def test_novelty_bounded_in_unit_interval_for_identical_titles():
    history = [_history_entry("CREDIT", {"TSLA"}, "Tesla enters new credit facility deal", _T0)]
    novelty = _compute_novelty(
        "CREDIT", frozenset({"TSLA"}), "Tesla enters new credit facility deal", _T0 + timedelta(hours=1), history, _LOOKBACK_30D
    )
    assert novelty == NOVELTY_FLOOR  # identical title -> max similarity 1.0 -> floor, never 0


def test_novelty_processed_chronologically_never_uses_future_events():
    """
    Integration-level (via EventResolver, not the pure function):
    build two documents about the same company/event_type with
    near-duplicate titles, far enough apart in time that they do NOT
    merge into one cluster (so each gets its own novelty computation),
    but still within the novelty lookback window. The EARLIER
    document's novelty must be 1.0 (nothing preceded it), regardless of
    the LATER document's existence - i.e. novelty never "sees into the
    future", even though both documents are passed into resolve() at
    once and even if fed in reverse chronological order.
    """
    early = DocumentFeatures(
        document=_doc(
            "early-1",
            "Tesla enters new credit facility deal",
            "Tesla Inc. entered into a new credit facility with lenders.",
            "2026-10-01T10:00:00Z",
        ),
        entities=[_entity("TSLA")],
        event_classification=_event(event_type="CREDIT"),
    )
    later = DocumentFeatures(
        document=_doc(
            "later-1",
            "Tesla enters new credit facility deal confirmed",
            "Tesla Inc. confirmed its new credit facility with lenders.",
            "2026-10-05T10:00:00Z",  # 96 hours later: beyond the 72h merge gap, so NOT merged
        ),
        entities=[_entity("TSLA")],
        event_classification=_event(event_type="CREDIT"),
    )

    resolver = EventResolver()
    resolved_forward = resolver.resolve([early, later])
    resolved_reversed = resolver.resolve([later, early])  # input order must not matter

    for resolved in (resolved_forward, resolved_reversed):
        assert resolved["early-1"].resolved_event_id != resolved["later-1"].resolved_event_id
        assert resolved["early-1"].novelty == 1.0  # nothing preceded it
        assert resolved["later-1"].novelty < 1.0  # the near-duplicate earlier event reduced it


# --- velocity ------------------------------------------------------------


def test_velocity_zero_for_single_document():
    assert _compute_velocity(1, "2026-10-01T10:00:00+00:00", "2026-10-01T10:00:00+00:00", 2.0, 1.0) == 0.0


def test_velocity_handles_identical_timestamps_without_division_by_zero():
    from datetime import datetime, timezone

    ts = datetime(2026, 10, 1, 10, 0, 0, tzinfo=timezone.utc)
    # 3 documents, all at the exact same timestamp - must not raise, must be bounded.
    velocity = _compute_velocity(3, ts, ts, reference_rate_per_hour=2.0, min_time_span_hours=1.0)
    assert 0.0 <= velocity <= 1.0


def test_velocity_increases_with_more_mentions_in_same_span():
    from datetime import datetime, timezone

    first = datetime(2026, 10, 1, 10, 0, 0, tzinfo=timezone.utc)
    last = datetime(2026, 10, 1, 12, 0, 0, tzinfo=timezone.utc)
    low = _compute_velocity(2, first, last, reference_rate_per_hour=2.0, min_time_span_hours=1.0)
    high = _compute_velocity(5, first, last, reference_rate_per_hour=2.0, min_time_span_hours=1.0)
    assert high > low
    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0


# --- corroboration (pure function) -----------------------------------------


def test_corroboration_pure_function_bounds():
    assert _compute_corroboration(1, 3) == 0.0
    assert _compute_corroboration(0, 3) == 0.0
    assert 0.0 < _compute_corroboration(2, 3) < 1.0
    assert _compute_corroboration(4, 3) == 1.0
    assert _compute_corroboration(100, 3) == 1.0  # saturates, never exceeds 1.0


# --- text similarity (pure function) ----------------------------------------


def test_text_similarity_identical_titles_is_one():
    assert _text_similarity("Tesla enters credit facility", "Tesla enters credit facility") == 1.0


def test_text_similarity_empty_title_is_zero():
    assert _text_similarity("", "Tesla enters credit facility") == 0.0


def test_text_similarity_unrelated_titles_is_low():
    assert _text_similarity("Tesla enters credit facility", "Local bakery wins award") < 0.2


# --- determinism / replay reproducibility -----------------------------------


def test_resolution_is_deterministic_regardless_of_input_order():
    f1 = _features(
        "d1", "Tesla credit facility deal", "Tesla entered a new credit facility.",
        "2026-10-01T10:00:00Z", tickers=["TSLA"], event_type="CREDIT", publisher="reuters.com",
    )
    f2 = _features(
        "d2", "Tesla credit facility deal confirmed", "Tesla confirmed the new credit facility.",
        "2026-10-01T12:00:00Z", tickers=["TSLA"], event_type="CREDIT", publisher="bloomberg.com",
    )
    f3 = _features(
        "d3", "Apple quarterly earnings", "Apple reported its quarterly results.",
        "2026-10-02T09:00:00Z", tickers=["AAPL"], event_type="EARNINGS",
    )

    resolver = EventResolver()
    resolved_order_a = resolver.resolve([f1, f2, f3])
    resolved_order_b = resolver.resolve([f3, f2, f1])  # shuffled input order

    for doc_id in ("d1", "d2", "d3"):
        assert resolved_order_a[doc_id].resolved_event_id == resolved_order_b[doc_id].resolved_event_id


# --- every input document gets a resolved event -----------------------------


def test_every_document_receives_a_resolved_event_even_as_a_singleton():
    f1 = _features(
        "d1", "Unique Tesla event", "Something happened at Tesla.",
        "2026-10-01T10:00:00Z", tickers=["TSLA"], event_type="PRODUCT_TECH",
    )
    resolver = EventResolver()
    resolved = resolver.resolve([f1])
    assert "d1" in resolved
    assert resolved["d1"].resolved_event_id.startswith("resolved-event-")
    assert resolved["d1"].member_document_ids == ["d1"]
