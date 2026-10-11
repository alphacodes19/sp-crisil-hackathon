import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from fastapi.testclient import TestClient

import src.api.main as api_main
from src.impact.signal_builder import (
    Signal,
    SignalEntity,
    SignalEventClassification,
    SignalProvenance,
    SignalSentiment,
)
from src.utils.io import write_jsonl


def _signal(document_id: str, ticker: str, event_id: str = None) -> Signal:
    return Signal(
        signal_id=f"signal-{document_id}",
        event_id=event_id or f"provisional-event-{document_id}",
        document_id=document_id,
        timestamp="2026-10-08T10:00:00Z",
        source=SignalProvenance(
            document_id=document_id,
            source_name="GDELT",
            source_type="news",
            publisher="example.com",
            url=f"https://example.com/{document_id}",
        ),
        entities=[
            SignalEntity(canonical_name="Tesla Inc.", ticker=ticker, matched_alias=ticker, match_type="ticker")
        ]
        if ticker
        else [],
        sentiment=SignalSentiment(label="positive", score=0.5, model_score=0.7),
        event=SignalEventClassification(event_type="CREDIT", classifier_score=None, method="rule", matched_rules=["x"]),
        scope="single_entity" if ticker else "unmapped",
        impact_score=None,
        evidence=None,
    )


@pytest.fixture
def client():
    return TestClient(api_main.app)


@pytest.fixture(autouse=True)
def _reset_signals_path(monkeypatch, tmp_path):
    # Every test gets an isolated, nonexistent-by-default signals path
    # so tests never touch the real data/signals.jsonl and never leak
    # state between each other.
    default_path = tmp_path / "signals.jsonl"
    monkeypatch.setattr(api_main, "SIGNALS_PATH", default_path)
    return default_path


# --- /health ------------------------------------------------------------


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# --- missing / empty signals file ---------------------------------------


def test_get_signals_missing_file_returns_empty_list(client):
    resp = client.get("/signals")
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_signals_empty_file_returns_empty_list(client, _reset_signals_path):
    _reset_signals_path.write_text("")  # 0-byte file, exists but empty
    resp = client.get("/signals")
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_signals_by_ticker_missing_file_returns_empty_list(client):
    resp = client.get("/signals/TSLA")
    assert resp.status_code == 200
    assert resp.json() == []


# --- /signals -------------------------------------------------------------


def test_get_signals_returns_persisted_signals(client, _reset_signals_path):
    signals = [_signal("d1", "TSLA"), _signal("d2", "AAPL")]
    write_jsonl(_reset_signals_path, signals)

    resp = client.get("/signals")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 2
    assert {s["document_id"] for s in body} == {"d1", "d2"}


# --- /signals/{ticker} ------------------------------------------------------


def test_get_signals_by_ticker_filters_correctly(client, _reset_signals_path):
    signals = [_signal("d1", "TSLA"), _signal("d2", "AAPL"), _signal("d3", "TSLA")]
    write_jsonl(_reset_signals_path, signals)

    resp = client.get("/signals/TSLA")
    assert resp.status_code == 200
    body = resp.json()
    assert {s["document_id"] for s in body} == {"d1", "d3"}


def test_get_signals_by_ticker_is_case_insensitive(client, _reset_signals_path):
    write_jsonl(_reset_signals_path, [_signal("d1", "TSLA")])
    resp = client.get("/signals/tsla")
    assert resp.status_code == 200
    assert len(resp.json()) == 1


def test_get_signals_by_ticker_no_match_returns_empty_list_not_error(client, _reset_signals_path):
    write_jsonl(_reset_signals_path, [_signal("d1", "TSLA")])
    resp = client.get("/signals/ZZZZ")
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_signals_by_ticker_includes_unmapped_documents_correctly(client, _reset_signals_path):
    # A document with no resolved entities must never spuriously match a ticker filter.
    write_jsonl(_reset_signals_path, [_signal("d1", None)])
    resp = client.get("/signals/TSLA")
    assert resp.status_code == 200
    assert resp.json() == []


# --- /events/{event_id} ------------------------------------------------------


def test_get_event_found(client, _reset_signals_path):
    write_jsonl(_reset_signals_path, [_signal("d1", "TSLA")])
    resp = client.get("/events/provisional-event-d1")
    assert resp.status_code == 200
    assert resp.json()["document_id"] == "d1"


def test_get_event_not_found_returns_404(client, _reset_signals_path):
    write_jsonl(_reset_signals_path, [_signal("d1", "TSLA")])
    resp = client.get("/events/provisional-event-does-not-exist")
    assert resp.status_code == 404


def test_get_event_missing_file_returns_404(client):
    resp = client.get("/events/provisional-event-d1")
    assert resp.status_code == 404


# --- Milestone 7: API compatibility with the enriched (resolved-event /
# impact-aware) signal contract ------------------------------------------


def test_get_signals_serves_resolved_event_signals_with_enriched_fields(client, _reset_signals_path):
    """
    A signal produced by the real batch pipeline (resolved event_id,
    real impact score/breakdown, event_resolution metadata) must still
    round-trip through the API exactly like a Milestone-6-shaped
    signal - the API itself needed no changes because `Signal` only
    grew new *optional* fields.
    """
    import sys as _sys
    from pathlib import Path as _Path

    _root = _Path(__file__).resolve().parents[1]
    if str(_root) not in _sys.path:
        _sys.path.insert(0, str(_root))

    from src.entities.resolver import EntityResolver
    from src.events.classifier import EventClassifier
    from src.events.resolution import EventResolver
    from src.impact.signal_builder import process_documents
    from src.nlp.sentiment import SentimentAnalyzer
    from src.utils.schema import UnifiedDocument

    def _fake_sent(text):
        return [{"label": "neutral", "score": 0.6}, {"label": "positive", "score": 0.25}, {"label": "negative", "score": 0.15}]

    doc = UnifiedDocument(
        document_id="rd-1",
        source_type="news",
        source_name="GDELT",
        publisher="example.com",
        timestamp="2026-10-08T10:00:00Z",
        title="Tesla credit facility",
        text="Tesla Inc. entered into a new credit facility with its lenders.",
        url="https://example.com/rd-1",
    )
    signals, failures = process_documents(
        [doc],
        resolver=EntityResolver(),
        sentiment_analyzer=SentimentAnalyzer(inference_fn=_fake_sent),
        event_classifier=EventClassifier(enable_zero_shot=False),
        event_resolver=EventResolver(),
    )
    assert failures == []
    write_jsonl(_reset_signals_path, signals)

    resp = client.get("/signals")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["event_id"].startswith("resolved-event-")
    assert body[0]["impact_score"] is not None
    assert body[0]["event_resolution"]["document_count"] == 1

    # /signals/{ticker} still filters correctly on enriched signals.
    resp_ticker = client.get("/signals/TSLA")
    assert resp_ticker.status_code == 200
    assert len(resp_ticker.json()) == 1

    # /events/{event_id} resolves the real resolved event id.
    resp_event = client.get(f"/events/{body[0]['event_id']}")
    assert resp_event.status_code == 200
    assert resp_event.json()["document_id"] == "rd-1"
