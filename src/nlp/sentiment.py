"""
FinBERT-based sentiment component.

Wraps ProsusAI/finbert through Hugging Face Transformers. Per Section 8
of the master context:

    label        : "positive" | "neutral" | "negative"
    score        : P(positive) - P(negative), continuous in [-1, 1]
    model_score  : the model's own probability for the predicted label

IMPORTANT - sentiment is direction/polarity only. It is NOT the impact
score (severity) and must never be used as one; this module has no
dependency on, and is never imported by, src/impact/.

Design notes:
- SentimentAnalyzer loads the model/tokenizer once per instance (in
  __init__), not once per document. Reuse one instance across many
  analyze() calls.
- The actual Hugging Face `transformers` import is lazy (inside
  `_build_default_pipeline`), so simply importing this module - or
  constructing a SentimentAnalyzer with an injected `pipeline_fn` for
  tests - never requires `transformers`/`torch` to be installed, let
  alone network access to download FinBERT's weights. That's what lets
  the unit tests run fully offline (see tests/test_sentiment.py).
- Score/label computation is a separate pure function
  (`_build_sentiment_result`) that only consumes already-produced
  per-class scores, so it can be unit-tested directly without any
  model at all.

Long-document handling (SEC 8-Ks can far exceed FinBERT's ~512-token
context window):
- Short text (<= `max_words_per_chunk` words) is scored in a single
  pass, exactly as before - no behavior change for the common case.
- Long text is split into word-count-based chunks (`_chunk_text`;
  simple whitespace splitting, not real tokenization - a deliberately
  lightweight, conservative approximation of the model's token limit,
  not an exact one) and each chunk is scored independently.
- Chunk-level scores are then combined into ONE document-level
  result via a length-weighted average of each class's probability
  across chunks (`_aggregate_chunk_scores`), longer chunks counting
  more. The averaged per-class scores are then run through the exact
  same `_build_sentiment_result` used for single-pass inference, so
  `score` and `model_score` keep their original meaning
  (`score = P(positive) - P(negative)`, `model_score` = the predicted
  label's probability) whether the input was chunked or not. This is a
  simple average, not a second model pass or hierarchical model.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Literal, Optional

from pydantic import BaseModel

MODEL_NAME = "ProsusAI/finbert"

SentimentLabel = Literal["positive", "neutral", "negative"]
RECOGNIZED_LABELS = {"positive", "negative", "neutral"}

# Conservative word-count approximation of FinBERT's ~512-token context
# window (roughly 0.75 words/token in English financial text, with
# margin). This is intentionally approximate - it avoids depending on
# the real tokenizer (which would require `transformers` to even be
# importable) while still keeping chunks comfortably under the limit.
DEFAULT_MAX_WORDS_PER_CHUNK = 350

# A raw model output is a list of per-class {"label": ..., "score": ...}
# entries, e.g. what Hugging Face's text-classification pipeline returns
# with top_k=None (all classes, not just the top one).
RawClassScores = List[Dict[str, object]]
InferenceFn = Callable[[str], RawClassScores]


class SentimentResult(BaseModel):
    label: SentimentLabel
    score: float  # P(positive) - P(negative), in [-1, 1]
    model_score: float  # probability of the predicted label


def _build_sentiment_result(raw_scores: RawClassScores) -> SentimentResult:
    """
    Pure function: raw per-class model scores -> SentimentResult.
    No model, no I/O - fully unit-testable on its own.
    """
    scores: Dict[str, float] = {}
    for entry in raw_scores:
        label = str(entry["label"]).strip().lower()
        if label not in RECOGNIZED_LABELS:
            raise ValueError(
                f"Unrecognized sentiment label '{entry['label']}' in model output; "
                f"expected one of {sorted(RECOGNIZED_LABELS)}"
            )
        scores[label] = float(entry["score"])  # type: ignore[arg-type]

    missing = RECOGNIZED_LABELS - scores.keys()
    if missing:
        raise ValueError(
            f"Missing expected sentiment class(es) in model output: {sorted(missing)}"
        )

    predicted_label = max(scores, key=scores.get)  # type: ignore[arg-type]
    continuous_score = scores["positive"] - scores["negative"]
    # Defensive clip in case of floating-point drift outside [-1, 1].
    continuous_score = max(-1.0, min(1.0, continuous_score))

    return SentimentResult(
        label=predicted_label,  # type: ignore[arg-type]
        score=continuous_score,
        model_score=scores[predicted_label],
    )


def _chunk_text(text: str, max_words: int) -> List[str]:
    """
    Split text into whitespace-delimited word chunks of at most
    `max_words` words each. Pure, no model/tokenizer involved. Returns
    a single-element list for text already within the limit.
    """
    words = text.split()
    if not words:
        return []
    return [
        " ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)
    ]


def _aggregate_chunk_scores(
    chunk_raw_scores: List[RawClassScores], chunk_weights: List[float]
) -> RawClassScores:
    """
    Combine several chunks' per-class scores into one set via a
    length-weighted average per class (longer chunks count more).
    Pure function; does not itself validate labels - the result is
    fed back through `_build_sentiment_result`, which already does
    that validation, so it isn't duplicated here.
    """
    if not chunk_raw_scores:
        raise ValueError("no chunk scores to aggregate")
    if len(chunk_raw_scores) != len(chunk_weights):
        raise ValueError("chunk_raw_scores and chunk_weights must be the same length")

    # Guard against degenerate all-zero weights (e.g. a chunk of only
    # whitespace) so no chunk silently contributes nothing.
    safe_weights = [w if w > 0 else 1.0 for w in chunk_weights]
    total_weight = sum(safe_weights)

    weighted_sums: Dict[str, float] = {}
    for raw_scores, weight in zip(chunk_raw_scores, safe_weights):
        for entry in raw_scores:
            label = str(entry["label"]).strip().lower()
            weighted_sums[label] = weighted_sums.get(label, 0.0) + float(entry["score"]) * weight

    return [
        {"label": label, "score": total / total_weight}
        for label, total in weighted_sums.items()
    ]


class SentimentAnalyzer:
    """
    Loads FinBERT once (unless an `inference_fn` is injected, e.g. for
    tests) and reuses it across calls to `analyze()`.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        inference_fn: Optional[InferenceFn] = None,
        max_words_per_chunk: int = DEFAULT_MAX_WORDS_PER_CHUNK,
    ) -> None:
        self._model_name = model_name
        self._infer: InferenceFn = inference_fn or self._build_default_pipeline(model_name)
        self._max_words_per_chunk = max_words_per_chunk

    @staticmethod
    def _build_default_pipeline(model_name: str) -> InferenceFn:
        """
        Build the real Hugging Face pipeline. Imports `transformers`
        lazily so this function - and therefore real model loading -
        only runs when nothing has been injected for testing.
        Requires network access the first time (to download weights)
        and the `transformers` (+ a backend like `torch`) package.
        """
        from transformers import pipeline  # local import - see docstring above

        hf_pipeline = pipeline(
            "text-classification",
            model=model_name,
            tokenizer=model_name,
            top_k=None,  # return scores for all classes, not just the top one
        )

        def _infer(text: str) -> RawClassScores:
            raw = hf_pipeline(text)
            # Some transformers versions wrap a single input's result in
            # an extra list (List[List[Dict]]); normalize to List[Dict].
            if raw and isinstance(raw[0], list):
                raw = raw[0]
            return raw

        return _infer

    def analyze(self, text: str) -> SentimentResult:
        if not text or not text.strip():
            raise ValueError("text must not be empty")

        chunks = _chunk_text(text, self._max_words_per_chunk)

        if len(chunks) <= 1:
            # Short text: single-pass inference, unchanged from before.
            raw_scores = self._infer(text)
            return _build_sentiment_result(raw_scores)

        # Long text: score each chunk independently, then aggregate
        # into one document-level result (see module docstring).
        chunk_raw_scores = [self._infer(chunk) for chunk in chunks]
        chunk_weights = [float(len(chunk.split())) for chunk in chunks]
        aggregated = _aggregate_chunk_scores(chunk_raw_scores, chunk_weights)
        return _build_sentiment_result(aggregated)
