"""
CLI entry point: generate signals for the SYNTHETIC high-impact demo
fixture only (src/ingestion/synthetic_demo.py), writing them to
data/signals_synthetic_demo.jsonl - a SEPARATE file from the ordinary
real-source replay's data/signals.jsonl.

Run (from the repo root), after scripts/generate_replay_signals.py:

    python scripts/generate_synthetic_demo_signals.py

Like scripts/generate_replay_signals.py, this builds REAL
SentimentAnalyzer/EventClassifier instances, so it requires
`transformers`/`torch` installed and network access to Hugging Face
the first time. NOT runnable inside the dev sandbox used to build this
project - run on a normal machine with internet before the demo.

The dashboard (dashboard/app.py) reads data/signals.jsonl by default
and only ALSO reads data/signals_synthetic_demo.jsonl when the
operator explicitly ticks "Include synthetic high-impact demo fixture"
in its sidebar - the fictional fixture is never silently mixed into
the ordinary real-source replay data on disk or in the UI.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.impact.signal_builder import process_documents
from src.ingestion.synthetic_demo import load_replay
from src.utils.io import write_jsonl

DEFAULT_OUTPUT_PATH = ROOT / "data" / "signals_synthetic_demo.jsonl"


def main() -> None:
    documents = load_replay()
    signals, failures = process_documents(documents)  # real default components
    write_jsonl(DEFAULT_OUTPUT_PATH, signals)
    print(f"Wrote {len(signals)} synthetic demo signal(s) to {DEFAULT_OUTPUT_PATH}")
    if failures:
        print(f"{len(failures)} document(s) failed and were skipped:")
        for failure in failures:
            print(f"  - {failure.document_id}: {failure.error}")


if __name__ == "__main__":
    main()
