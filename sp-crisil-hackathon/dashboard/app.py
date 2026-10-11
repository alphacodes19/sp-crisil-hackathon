"""
Module B dashboard: event -> impact -> exposure -> scenario -> asset
shock -> portfolio loss, as a minimum usable Streamlit/Plotly app.

Run (from the repo root), after generating signals:

    python scripts/generate_replay_signals.py
    streamlit run dashboard/app.py

This dashboard NEVER downloads or runs an ML model itself - it only
reads already-persisted signals (data/signals.jsonl, and optionally
data/signals_synthetic_demo.jsonl - see below), data/portfolio/portfolio.csv,
and data/scenarios/scenario_library.json, then runs the pure-Python
`run_stress_test()` from src/stress/stress_engine.py. If no inference
was actually run to produce those signals yet, this dashboard has
nothing to show and says so - it never fabricates/stubs a result to
fill the screen.

The synthetic high-impact demo fixture (src/ingestion/synthetic_demo.py)
is loaded ONLY when the operator explicitly ticks the sidebar checkbox
below; data/signals.jsonl (the ordinary real-source GDELT/SEC replay)
is never silently mixed with the fictional data, on disk or in this
UI - every synthetic entry is also labeled "[SYNTHETIC]" in its own
selector entry as a second, redundant layer of transparency.

Pure helper functions below (`_load_dashboard_signals`,
`_is_synthetic_signal`, `_format_signal_label`,
`_build_waterfall_figure`, `_top_contributors_rows`) contain no
Streamlit calls and are unit-tested directly
(tests/test_dashboard_helpers.py) without needing a running Streamlit
server. All `st.*`/rendering calls are confined to `main()`, which only
runs under `streamlit run` (i.e. when this file is executed as
`__main__`) - importing this module for its helpers never triggers any
UI rendering.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.impact.signal_builder import DEFAULT_SIGNALS_PATH, Signal
from src.stress.stress_engine import (
    AssetStressResult,
    StressTestResult,
    load_portfolio,
    load_scenario_library,
    run_stress_test,
)
from src.utils.io import read_jsonl

SYNTHETIC_SIGNALS_PATH = ROOT / "data" / "signals_synthetic_demo.jsonl"
SYNTHETIC_SOURCE_NAME = "SYNTHETIC_DEMO_FIXTURE"


def _read_signals_file(path: Path) -> List[Signal]:
    if not path.exists():
        return []
    return list(read_jsonl(path, Signal))


def _is_synthetic_signal(signal: Signal) -> bool:
    return signal.source.source_name == SYNTHETIC_SOURCE_NAME


def _load_dashboard_signals(include_synthetic: bool) -> List[Signal]:
    """Always loads the ordinary real-source replay signals; ONLY also
    loads the synthetic fixture's signals when `include_synthetic` is
    True (set by the sidebar checkbox - never on by default)."""
    signals = _read_signals_file(DEFAULT_SIGNALS_PATH)
    if include_synthetic:
        signals = signals + _read_signals_file(SYNTHETIC_SIGNALS_PATH)
    return signals


def _format_signal_label(signal: Signal) -> str:
    prefix = "[SYNTHETIC] " if _is_synthetic_signal(signal) else ""
    impact = f"{signal.impact_score:.2f}" if signal.impact_score is not None else "n/a"
    entities = ", ".join(e.ticker for e in signal.entities) or "no entity resolved"
    return f"{prefix}{signal.event.event_type} | {entities} | impact {impact} | {signal.document_id}"


def _top_contributors_rows(top_contributors: List[AssetStressResult]) -> List[dict]:
    return [
        {
            "asset_id": c.asset_id,
            "issuer": c.issuer,
            "asset_type": c.asset_type,
            "stress_loss": round(c.stress_loss, 2),
            "matching_rules": ", ".join(c.matching_rules),
            "explanation": c.explanation,
        }
        for c in top_contributors
    ]


def _build_waterfall_figure(result: StressTestResult):
    """Builds a Plotly waterfall: portfolio value before -> one bar per
    asset-class loss/gain -> portfolio value after. Import of `plotly`
    is local to this function so importing this module for its other
    (tested) helpers never requires plotly to be installed."""
    import plotly.graph_objects as go

    classes = sorted(result.attribution_by_asset_class.keys())
    measures = ["absolute"] + ["relative"] * len(classes) + ["total"]
    labels = ["Before stress"] + classes + ["After stress"]
    values = [result.portfolio_value_before] + [
        -result.attribution_by_asset_class[c] for c in classes
    ] + [result.portfolio_value_after]

    fig = go.Figure(
        go.Waterfall(
            name="Portfolio stress",
            orientation="v",
            measure=measures,
            x=labels,
            y=values,
            connector={"line": {"color": "rgb(120,120,120)"}},
        )
    )
    fig.update_layout(title="Portfolio value: before stress -> loss by asset class -> after stress")
    return fig


def main() -> None:  # pragma: no cover - exercised only under `streamlit run`, not pytest
    import streamlit as st

    st.set_page_config(page_title="Risk Signal -> Portfolio Stress", layout="wide")
    st.title("Event-Driven Portfolio Stress Test (Module B)")
    st.caption(
        "Reads persisted signals + the synthetic portfolio/scenario library only - "
        "never downloads or runs an ML model from this dashboard."
    )

    with st.sidebar:
        st.header("Replay controls")
        include_synthetic = st.checkbox(
            "Include synthetic high-impact demo fixture (clearly fictional - see docs/README)",
            value=False,
        )
        threshold = st.number_input("Impact-score trigger threshold", value=7.0, step=0.5)

    signals = _load_dashboard_signals(include_synthetic)
    if not signals:
        st.warning(
            "No persisted signals found. Run `python scripts/generate_replay_signals.py` "
            "(and, if you want the synthetic fixture available, also "
            "`python scripts/generate_synthetic_demo_signals.py`) before starting this dashboard."
        )
        return

    labels = [_format_signal_label(s) for s in signals]
    selected_index = st.selectbox("Select a signal", options=range(len(signals)), format_func=lambda i: labels[i])
    signal = signals[selected_index]

    if _is_synthetic_signal(signal):
        st.info("This signal comes from the SYNTHETIC, entirely fictional demo fixture - not real news.")

    st.subheader("Event")
    col1, col2, col3 = st.columns(3)
    col1.metric("Event type", signal.event.event_type)
    col2.metric("Classification method", signal.event.method)
    col3.metric("Impact score (1-10, uncalibrated heuristic)", f"{signal.impact_score:.2f}" if signal.impact_score is not None else "n/a")
    st.write(f"**Entities:** {', '.join(e.ticker for e in signal.entities) or 'none resolved'}")
    st.write(f"**Source:** {signal.source.source_name} ({signal.source.publisher}) - [{signal.source.url}]({signal.source.url})" if signal.source.url else f"**Source:** {signal.source.source_name} ({signal.source.publisher})")
    if signal.event_resolution:
        st.write(
            f"**Resolved event evidence:** {signal.evidence} "
            f"(first seen {signal.event_resolution.first_seen}, latest seen {signal.event_resolution.latest_seen})"
        )

    st.subheader("Stress test")
    portfolio = load_portfolio()
    scenario_library = load_scenario_library()
    result = run_stress_test(signal, portfolio, scenario_library, threshold=threshold)

    st.write(f"**Trigger reason:** {result.trigger_reason}")
    if not result.triggered:
        st.info("Stress test did not trigger for this signal - nothing further to show.")
        return

    st.write(f"**Scenario:** {result.scenario_name}  |  **Severity multiplier:** {result.severity_multiplier}")
    with st.expander("Scenario assumptions used (stylized, not calibrated)"):
        st.json(result.scenario_assumptions)

    colA, colB, colC = st.columns(3)
    colA.metric("Portfolio value before", f"${result.portfolio_value_before:,.0f}")
    colB.metric("Portfolio value after", f"${result.portfolio_value_after:,.0f}")
    colC.metric("Loss", f"${result.absolute_loss:,.0f}  ({result.pct_loss * 100:.2f}%)")

    st.plotly_chart(_build_waterfall_figure(result), width="stretch")

    st.subheader("Top loss contributors")
    st.dataframe(_top_contributors_rows(result.top_contributors))

    if result.skipped_assets:
        with st.expander(f"{len(result.skipped_assets)} asset(s) skipped (missing/invalid data)"):
            st.dataframe(_top_contributors_rows(result.skipped_assets))


if __name__ == "__main__":  # pragma: no cover
    main()
