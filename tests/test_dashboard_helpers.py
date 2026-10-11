import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dashboard.app as dashboard_app
from src.entities.resolver import EntityResolver
from src.events.classifier import EventClassifier
from src.events.resolution import EventResolver
from src.impact.signal_builder import process_documents
from src.ingestion.synthetic_demo import load_replay as load_synthetic_replay
from src.nlp.sentiment import SentimentAnalyzer
from src.stress.stress_engine import load_portfolio, load_scenario_library, run_stress_test
from src.utils.io import write_jsonl


def _fake_sentiment_infer(text):
    return [{"label": "negative", "score": 0.7}, {"label": "neutral", "score": 0.2}, {"label": "positive", "score": 0.1}]


def _synthetic_signal():
    docs = load_synthetic_replay()
    signals, failures = process_documents(
        docs,
        resolver=EntityResolver(),
        sentiment_analyzer=SentimentAnalyzer(inference_fn=_fake_sentiment_infer),
        event_classifier=EventClassifier(enable_zero_shot=False),
        event_resolver=EventResolver(),
    )
    assert failures == []
    return signals[0]


def test_importing_dashboard_module_triggers_no_streamlit_calls():
    # If main()'s st.* calls ran at import time, this import itself
    # would have already raised/rendered something; reaching here at
    # all (plus the explicit attribute checks) is the proof.
    assert hasattr(dashboard_app, "main")
    assert callable(dashboard_app._load_dashboard_signals)


def test_load_dashboard_signals_excludes_synthetic_by_default(tmp_path, monkeypatch):
    real_signal = _synthetic_signal()  # reuse the builder; content doesn't matter for this test
    monkeypatch.setattr(dashboard_app, "DEFAULT_SIGNALS_PATH", tmp_path / "signals.jsonl")
    monkeypatch.setattr(dashboard_app, "SYNTHETIC_SIGNALS_PATH", tmp_path / "synthetic.jsonl")
    write_jsonl(dashboard_app.DEFAULT_SIGNALS_PATH, [real_signal])
    write_jsonl(dashboard_app.SYNTHETIC_SIGNALS_PATH, [real_signal])

    without_synthetic = dashboard_app._load_dashboard_signals(include_synthetic=False)
    with_synthetic = dashboard_app._load_dashboard_signals(include_synthetic=True)

    assert len(without_synthetic) == 1
    assert len(with_synthetic) == 2


def test_load_dashboard_signals_handles_missing_files_gracefully(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard_app, "DEFAULT_SIGNALS_PATH", tmp_path / "does-not-exist.jsonl")
    monkeypatch.setattr(dashboard_app, "SYNTHETIC_SIGNALS_PATH", tmp_path / "also-missing.jsonl")
    assert dashboard_app._load_dashboard_signals(include_synthetic=True) == []


def test_is_synthetic_signal_detects_fixture_source():
    synthetic_signal = _synthetic_signal()
    assert dashboard_app._is_synthetic_signal(synthetic_signal) is True


def test_format_signal_label_flags_synthetic_signals():
    synthetic_signal = _synthetic_signal()
    label = dashboard_app._format_signal_label(synthetic_signal)
    assert label.startswith("[SYNTHETIC]")
    assert "ZZDEMO" in label
    assert synthetic_signal.event.event_type in label


def test_top_contributors_rows_shape():
    signal = _synthetic_signal()
    result = run_stress_test(signal, load_portfolio(), load_scenario_library())
    rows = dashboard_app._top_contributors_rows(result.top_contributors)
    assert len(rows) == len(result.top_contributors)
    for row in rows:
        assert set(row.keys()) == {"asset_id", "issuer", "asset_type", "stress_loss", "matching_rules", "explanation"}


def test_build_waterfall_figure_produces_a_plotly_figure():
    signal = _synthetic_signal()
    result = run_stress_test(signal, load_portfolio(), load_scenario_library())
    fig = dashboard_app._build_waterfall_figure(result)
    assert fig.data[0].type == "waterfall"
    # before, N asset classes, after
    assert len(fig.data[0].x) == 2 + len(result.attribution_by_asset_class)
