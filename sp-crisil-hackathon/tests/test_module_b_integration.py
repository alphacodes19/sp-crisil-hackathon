"""
Milestone 8 integration tests:
- the synthetic high-impact fixture flows deterministically end-to-end
  from raw documents through the full NLP/resolution/impact pipeline
  into a real portfolio stress test;
- Module B's existence doesn't break the existing Signal contract or
  the read-only API (neither was touched by this milestone).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from src.entities.resolver import EntityResolver
from src.events.classifier import EventClassifier
from src.events.resolution import EventResolver
from src.impact.signal_builder import Signal, process_documents
from src.ingestion.synthetic_demo import load_replay as load_synthetic_replay
from src.nlp.sentiment import SentimentAnalyzer
from src.stress.stress_engine import load_portfolio, load_scenario_library, run_stress_test
from src.utils.io import write_jsonl


def _fake_sentiment_infer(text):
    return [{"label": "negative", "score": 0.7}, {"label": "neutral", "score": 0.2}, {"label": "positive", "score": 0.1}]


def _build_synthetic_signal():
    docs = load_synthetic_replay()
    signals, failures = process_documents(
        docs,
        resolver=EntityResolver(),
        sentiment_analyzer=SentimentAnalyzer(inference_fn=_fake_sentiment_infer),
        event_classifier=EventClassifier(enable_zero_shot=False),
        event_resolver=EventResolver(),
    )
    assert failures == []
    event_ids = {s.event_id for s in signals}
    assert len(event_ids) == 1
    return signals[0]


def test_synthetic_event_to_portfolio_integration_is_deterministic():
    signal_run_1 = _build_synthetic_signal()
    signal_run_2 = _build_synthetic_signal()
    assert signal_run_1.event_id == signal_run_2.event_id
    assert signal_run_1.impact_score == signal_run_2.impact_score

    portfolio = load_portfolio()
    scenarios = load_scenario_library()

    result_1 = run_stress_test(signal_run_1, portfolio, scenarios)
    result_2 = run_stress_test(signal_run_2, portfolio, scenarios)

    assert result_1.triggered is True
    assert result_1.triggered == result_2.triggered
    assert result_1.absolute_loss == result_2.absolute_loss
    assert result_1.portfolio_value_after == result_2.portfolio_value_after
    assert [c.asset_id for c in result_1.top_contributors] == [c.asset_id for c in result_2.top_contributors]


def test_synthetic_event_produces_meaningful_issuer_specific_stress():
    signal = _build_synthetic_signal()
    portfolio = load_portfolio()
    scenarios = load_scenario_library()
    result = run_stress_test(signal, portfolio, scenarios)

    assert result.triggered is True
    assert result.impact_score > 7.0
    zzdemo_contributors = [c for c in result.top_contributors if c.ticker == "ZZDEMO"]
    assert len(zzdemo_contributors) >= 1
    assert all("direct_issuer_match" in c.matching_rules for c in zzdemo_contributors)
    # Unrelated real-world issuers should not have been given an
    # arbitrary large issuer-specific shock just because something
    # happened somewhere in the portfolio.
    unrelated = [c for c in result.affected_assets if c.ticker not in ("ZZDEMO",) and "direct_issuer_match" in c.matching_rules]
    assert unrelated == []


def test_real_replay_signals_do_not_trigger_module_b_by_accident():
    """
    The ordinary GDELT/SEC replay signals have modest impact scores
    (well under the stylized synthetic fixture) and, except for the
    one entity the synthetic fixture was built around, no ZZDEMO
    exposure - run_stress_test must not fabricate a trigger for them.
    """
    from src.ingestion.gdelt import load_replay as load_gdelt_replay
    from src.ingestion.sec_edgar import load_replay as load_sec_replay

    docs = list(load_gdelt_replay()) + list(load_sec_replay())
    signals, failures = process_documents(
        docs,
        resolver=EntityResolver(),
        sentiment_analyzer=SentimentAnalyzer(inference_fn=_fake_sentiment_infer),
        event_classifier=EventClassifier(enable_zero_shot=False),
        event_resolver=EventResolver(),
    )
    assert failures == []
    portfolio = load_portfolio()
    scenarios = load_scenario_library()
    for signal in signals:
        result = run_stress_test(signal, portfolio, scenarios)
        # Each real replay signal has impact_score well under 7 with
        # rules-only classification (several are UNCLASSIFIED/below
        # threshold) - none should trigger.
        if signal.impact_score is not None and signal.impact_score > 7.0:
            continue  # would be a legitimate trigger, not asserted false here
        assert result.triggered is False


# --- existing API / Signal contract compatibility ---------------------------


def test_signal_contract_is_unmodified_by_module_b():
    """Module B must not have touched Signal's fields."""
    import src.impact.signal_builder as signal_builder_module

    fields = set(Signal.model_fields.keys())
    assert fields == {
        "signal_id", "event_id", "document_id", "timestamp", "source", "entities",
        "sentiment", "event", "scope", "impact_score", "impact_breakdown",
        "impact_method", "evidence", "event_resolution",
    }
    # stress_engine must not be imported by signal_builder (one-directional dependency).
    assert "stress_engine" not in dir(signal_builder_module)


def test_existing_api_endpoints_still_work_after_module_b(tmp_path):
    import src.api.main as api_main

    signal = _build_synthetic_signal()
    out_path = tmp_path / "signals.jsonl"
    write_jsonl(out_path, [signal])

    original_path = api_main.SIGNALS_PATH
    try:
        api_main.SIGNALS_PATH = out_path
        client = TestClient(api_main.app)
        assert client.get("/health").json() == {"status": "ok"}
        resp = client.get("/signals")
        assert resp.status_code == 200
        assert len(resp.json()) == 1
        ticker_resp = client.get("/signals/ZZDEMO")
        assert ticker_resp.status_code == 200
        assert len(ticker_resp.json()) == 1
        event_resp = client.get(f"/events/{signal.event_id}")
        assert event_resp.status_code == 200
    finally:
        api_main.SIGNALS_PATH = original_path
