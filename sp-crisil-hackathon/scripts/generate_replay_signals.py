"""
CLI entry point: generate data/signals.jsonl from the GDELT + SEC
replay data, running every document through the real (non-mocked)
EntityResolver -> SentimentAnalyzer -> EventClassifier -> signal_builder
pipeline.

Run (from the repo root):

    python scripts/generate_replay_signals.py

Then start the API (also from the repo root):

    uvicorn src.api.main:app --reload

This script builds REAL SentimentAnalyzer/EventClassifier instances (no
injected stubs), so it requires `transformers` + a backend (e.g.
`torch`) installed and, the first time, network access to download
FinBERT's and the zero-shot model's weights:

    pip install -r requirements.txt

NOT runnable inside the dev sandbox used to build this project (egress
there is restricted to a short allow-list that does not include
huggingface.co) - run this on a normal machine with internet before the
demo.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.impact.signal_builder import run_replay_pipeline


def main() -> None:
    signals, failures = run_replay_pipeline()
    print(f"Wrote {len(signals)} signal(s) to data/signals.jsonl")
    if failures:
        print(f"{len(failures)} document(s) failed and were skipped:")
        for failure in failures:
            print(f"  - {failure.document_id}: {failure.error}")


if __name__ == "__main__":
    main()
