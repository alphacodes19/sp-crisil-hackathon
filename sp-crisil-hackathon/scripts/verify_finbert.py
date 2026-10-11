"""
Optional manual verification script for the real FinBERT model.

This is NOT part of the pytest suite and is not run automatically. Use
it on a normal machine with internet access (to download
ProsusAI/finbert's weights the first time) and `transformers` + a
backend (e.g. `torch`) installed:

    pip install transformers torch
    python scripts/verify_finbert.py

It exercises the real SentimentAnalyzer (no injected inference_fn, so
it builds the actual Hugging Face pipeline) against a few hand-picked
example sentences and prints label / score / model_score for each, so
you can eyeball that real FinBERT output still looks sane end-to-end.

This script was NOT runnable inside the dev sandbox used to build this
project (egress there is restricted to a short allow-list that does
not include huggingface.co), so it is unverified until you run it.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.nlp.sentiment import SentimentAnalyzer

EXAMPLE_SENTENCES = [
    ("Positive (expected)", "The company reported record quarterly profits and raised its full-year guidance."),
    ("Negative (expected)", "The company announced significant layoffs amid a sharp decline in revenue."),
    ("Neutral (expected)", "The company filed its quarterly report with the Securities and Exchange Commission."),
]


def main() -> None:
    print(f"Loading {SentimentAnalyzer().__class__.__name__} (ProsusAI/finbert)...")
    analyzer = SentimentAnalyzer()  # real model load - no inference_fn injected

    for expectation, text in EXAMPLE_SENTENCES:
        result = analyzer.analyze(text)
        print(f"\n[{expectation}]")
        print(f"  text:        {text}")
        print(f"  label:       {result.label}")
        print(f"  score:       {result.score:.3f}  (P(positive) - P(negative))")
        print(f"  model_score: {result.model_score:.3f}  (probability of predicted label)")


if __name__ == "__main__":
    main()
