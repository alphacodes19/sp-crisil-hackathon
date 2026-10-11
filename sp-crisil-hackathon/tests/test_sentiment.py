import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from src.nlp.sentiment import (
    SentimentAnalyzer,
    _aggregate_chunk_scores,
    _build_sentiment_result,
    _chunk_text,
)


def _fake_inference(scores):
    """Build a fake inference_fn that ignores its input and always
    returns the given per-class scores (mocks the HF pipeline output)."""

    def _infer(text: str):
        return [{"label": label, "score": score} for label, score in scores.items()]

    return _infer


def test_positive_sentiment():
    analyzer = SentimentAnalyzer(
        inference_fn=_fake_inference({"positive": 0.90, "neutral": 0.07, "negative": 0.03})
    )
    result = analyzer.analyze("Company reports record profits and raised guidance.")
    assert result.label == "positive"
    assert result.score == pytest.approx(0.90 - 0.03)
    assert result.model_score == pytest.approx(0.90)


def test_negative_sentiment():
    analyzer = SentimentAnalyzer(
        inference_fn=_fake_inference({"positive": 0.05, "neutral": 0.10, "negative": 0.85})
    )
    result = analyzer.analyze("Company announces layoffs amid declining revenue.")
    assert result.label == "negative"
    assert result.score == pytest.approx(0.05 - 0.85)
    assert result.model_score == pytest.approx(0.85)


def test_neutral_sentiment():
    analyzer = SentimentAnalyzer(
        inference_fn=_fake_inference({"positive": 0.30, "neutral": 0.45, "negative": 0.25})
    )
    result = analyzer.analyze("The company filed its quarterly report with the SEC.")
    assert result.label == "neutral"
    assert result.score == pytest.approx(0.30 - 0.25)
    assert result.model_score == pytest.approx(0.45)


def test_score_is_bounded_in_minus_one_to_one():
    # Pathological input (scores don't sum to 1) should still be clipped
    # defensively rather than ever leaving [-1, 1].
    analyzer = SentimentAnalyzer(
        inference_fn=_fake_inference({"positive": 1.5, "neutral": 0.0, "negative": -0.5})
    )
    result = analyzer.analyze("Some text.")
    assert -1.0 <= result.score <= 1.0


def test_empty_text_raises_without_calling_model():
    def _should_not_be_called(text: str):
        raise AssertionError("inference_fn must not be called for empty text")

    analyzer = SentimentAnalyzer(inference_fn=_should_not_be_called)
    with pytest.raises(ValueError):
        analyzer.analyze("")
    with pytest.raises(ValueError):
        analyzer.analyze("    ")


def test_unexpected_label_raises_value_error():
    analyzer = SentimentAnalyzer(
        inference_fn=_fake_inference({"positive": 0.1, "surprise": 0.8, "negative": 0.1})
    )
    with pytest.raises(ValueError, match="Unrecognized sentiment label"):
        analyzer.analyze("Some unexpected model output.")


def test_missing_expected_class_raises_value_error():
    # Only two of the three expected classes present.
    analyzer = SentimentAnalyzer(
        inference_fn=_fake_inference({"positive": 0.6, "negative": 0.4})
    )
    with pytest.raises(ValueError, match="Missing expected sentiment class"):
        analyzer.analyze("Incomplete model output.")


def test_chunk_text_single_chunk_for_short_text():
    text = "The company filed its quarterly report with the SEC."
    chunks = _chunk_text(text, max_words=50)
    assert chunks == [text]


def test_chunk_text_splits_long_text_into_multiple_chunks():
    words = [f"word{i}" for i in range(25)]
    text = " ".join(words)
    chunks = _chunk_text(text, max_words=10)
    assert len(chunks) == 3
    assert chunks[0] == " ".join(words[0:10])
    assert chunks[1] == " ".join(words[10:20])
    assert chunks[2] == " ".join(words[20:25])


def test_chunk_text_empty_string_returns_no_chunks():
    assert _chunk_text("", max_words=50) == []
    assert _chunk_text("   ", max_words=50) == []


def test_aggregate_chunk_scores_is_length_weighted_average():
    # Chunk A (10 words): strongly positive. Chunk B (2 words): strongly negative.
    chunk_a = [{"label": "positive", "score": 0.9}, {"label": "neutral", "score": 0.08}, {"label": "negative", "score": 0.02}]
    chunk_b = [{"label": "positive", "score": 0.05}, {"label": "neutral", "score": 0.05}, {"label": "negative", "score": 0.90}]
    aggregated = _aggregate_chunk_scores([chunk_a, chunk_b], [10.0, 2.0])
    scores = {e["label"]: e["score"] for e in aggregated}
    # Weighted average: (0.9*10 + 0.05*2) / 12 for positive, etc.
    assert scores["positive"] == pytest.approx((0.9 * 10 + 0.05 * 2) / 12)
    assert scores["negative"] == pytest.approx((0.02 * 10 + 0.90 * 2) / 12)
    # The longer, positive chunk should dominate the final label.
    result = _build_sentiment_result(aggregated)
    assert result.label == "positive"


def test_aggregate_chunk_scores_equal_weights_is_plain_average():
    chunk_a = [{"label": "positive", "score": 0.8}, {"label": "neutral", "score": 0.1}, {"label": "negative", "score": 0.1}]
    chunk_b = [{"label": "positive", "score": 0.2}, {"label": "neutral", "score": 0.3}, {"label": "negative", "score": 0.5}]
    aggregated = _aggregate_chunk_scores([chunk_a, chunk_b], [1.0, 1.0])
    scores = {e["label"]: e["score"] for e in aggregated}
    assert scores["positive"] == pytest.approx(0.5)
    assert scores["negative"] == pytest.approx(0.3)


def test_aggregate_chunk_scores_rejects_mismatched_lengths():
    chunk_a = [{"label": "positive", "score": 0.8}]
    with pytest.raises(ValueError):
        _aggregate_chunk_scores([chunk_a], [1.0, 2.0])


def test_aggregate_chunk_scores_rejects_empty_input():
    with pytest.raises(ValueError):
        _aggregate_chunk_scores([], [])


def test_analyze_short_text_uses_single_pass_inference():
    calls = []

    def _counting_infer(text: str):
        calls.append(text)
        return [{"label": "positive", "score": 0.7}, {"label": "neutral", "score": 0.2}, {"label": "negative", "score": 0.1}]

    analyzer = SentimentAnalyzer(inference_fn=_counting_infer, max_words_per_chunk=50)
    short_text = "Short filing summary well under the chunk threshold."
    result = analyzer.analyze(short_text)

    assert calls == [short_text]  # inference called once, on the whole text
    assert result.label == "positive"
    assert result.score == pytest.approx(0.6)
    assert result.model_score == pytest.approx(0.7)


def test_analyze_long_text_is_chunked_and_aggregated():
    # Force chunking with a tiny max_words_per_chunk.
    call_texts = []

    def _infer_by_chunk_content(text: str):
        call_texts.append(text)
        # First chunk -> strongly positive, second chunk -> strongly negative,
        # mimicking a long SEC filing with mixed sections.
        if "good" in text:
            return [{"label": "positive", "score": 0.95}, {"label": "neutral", "score": 0.03}, {"label": "negative", "score": 0.02}]
        return [{"label": "positive", "score": 0.02}, {"label": "neutral", "score": 0.03}, {"label": "negative", "score": 0.95}]

    long_text = "good " * 10 + "bad " * 10  # 20 words total, two 10-word chunks
    analyzer = SentimentAnalyzer(inference_fn=_infer_by_chunk_content, max_words_per_chunk=10)
    result = analyzer.analyze(long_text.strip())

    assert len(call_texts) == 2  # inference called once per chunk, not once per document
    # Equal-length chunks with opposite extreme scores should roughly cancel out.
    assert result.score == pytest.approx(0.0, abs=0.01)


def test_analyze_empty_text_still_raises_without_chunking_or_model_call():
    def _should_not_be_called(text: str):
        raise AssertionError("inference_fn must not be called for empty text")

    analyzer = SentimentAnalyzer(inference_fn=_should_not_be_called, max_words_per_chunk=5)
    with pytest.raises(ValueError):
        analyzer.analyze("")


def test_build_sentiment_result_is_a_pure_function_without_a_model():
    # Exercises the scoring logic directly, bypassing SentimentAnalyzer
    # entirely - confirms it needs no model/analyzer instance at all.
    result = _build_sentiment_result(
        [
            {"label": "POSITIVE", "score": 0.7},  # case-insensitive label handling
            {"label": "neutral", "score": 0.2},
            {"label": "negative", "score": 0.1},
        ]
    )
    assert result.label == "positive"
    assert result.score == pytest.approx(0.6)
    assert result.model_score == pytest.approx(0.7)
