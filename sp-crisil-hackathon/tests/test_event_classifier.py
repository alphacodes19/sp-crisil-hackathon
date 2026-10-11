import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from src.events.classifier import (
    EVENT_TYPES,
    UNCLASSIFIED,
    EventClassifier,
    _build_zero_shot_input,
    _match_keyword_rules,
    _normalize_for_rules,
    _parse_zero_shot_output,
)
from src.utils.schema import UnifiedDocument


def _zero_shot_stub(labels_in_score_order):
    """Build a zero_shot_inference_fn stub returning a fixed, pre-sorted
    (label, score) list regardless of input, mimicking HF's pipeline
    output shape: {"labels": [...], "scores": [...]}."""

    def _infer(text, candidate_labels):
        labels = [l for l, _ in labels_in_score_order]
        scores = [s for _, s in labels_in_score_order]
        return {"sequence": text, "labels": labels, "scores": scores}

    return _infer


def _failing_zero_shot(*_args, **_kwargs):
    raise AssertionError("zero-shot inference must not be called")


# --- rule-based classification: clear examples across major categories -----


def test_rule_based_credit():
    clf = EventClassifier(zero_shot_inference_fn=_failing_zero_shot)
    result = clf.classify_text(
        "Tesla enters new credit agreement",
        "Tesla, Inc. entered into a $20 billion delayed draw term loan credit facility.",
    )
    assert result.event_type == "CREDIT"
    assert result.method == "rule"
    assert result.classifier_score is None
    assert "credit facility" in result.matched_rules or "credit agreement" in result.matched_rules


def test_rule_based_mna():
    clf = EventClassifier(zero_shot_inference_fn=_failing_zero_shot)
    result = clf.classify_text(
        "Big deal announced",
        "The companies entered into a definitive merger agreement valued at $5 billion.",
    )
    assert result.event_type == "M&A"
    assert result.method == "rule"


def test_rule_based_earnings():
    clf = EventClassifier(zero_shot_inference_fn=_failing_zero_shot)
    result = clf.classify_text(
        "Q3 update",
        "The company reported its financial results for the third quarter, citing higher revenue guidance.",
    )
    assert result.event_type == "EARNINGS"
    assert result.method == "rule"


def test_rule_based_cyber_operational():
    clf = EventClassifier(zero_shot_inference_fn=_failing_zero_shot)
    result = clf.classify_text(
        "Security notice",
        "The company disclosed a data breach affecting customer records after a ransomware attack.",
    )
    assert result.event_type == "CYBER_OPERATIONAL"
    assert result.method == "rule"


def test_rule_based_regulatory_legal():
    clf = EventClassifier(zero_shot_inference_fn=_failing_zero_shot)
    result = clf.classify_text(
        "Regulators act",
        "The firm is now subject to a regulatory investigation and received a subpoena from authorities.",
    )
    assert result.event_type == "REGULATORY_LEGAL"
    assert result.method == "rule"


# --- rule override: zero-shot is never even called when a rule resolves ----


def test_rule_match_overrides_and_skips_zero_shot_entirely():
    # The stub would raise if called - proves the rule path short-circuits
    # before any zero-shot inference happens.
    clf = EventClassifier(zero_shot_inference_fn=_failing_zero_shot)
    result = clf.classify_text(
        "Filing",
        "The company entered into a new revolving credit facility with its lenders.",
    )
    assert result.event_type == "CREDIT"
    assert result.method == "rule"


# --- zero-shot fallback: no rule match at all -------------------------------


def test_zero_shot_used_when_no_keyword_rules_match():
    clf = EventClassifier(
        zero_shot_inference_fn=_zero_shot_stub(
            [("GEOPOLITICAL", 0.81), ("MACRO_CENTRAL_BANK", 0.10)] + [(t, 0.01) for t in EVENT_TYPES if t not in ("GEOPOLITICAL", "MACRO_CENTRAL_BANK")]
        )
    )
    result = clf.classify_text(
        "World news",
        "Escalating tensions between the two nations raised concern among global investors.",
    )
    assert result.event_type == "GEOPOLITICAL"
    assert result.method == "zero_shot"
    assert result.classifier_score == pytest.approx(0.81)
    assert result.matched_rules == []


# --- ambiguous / multiple keyword matches -----------------------------------


def test_ambiguous_keyword_matches_falls_through_to_zero_shot():
    # Contains both a CREDIT phrase and a CYBER_OPERATIONAL phrase.
    text = (
        "The company entered into a new credit facility this quarter, and separately "
        "disclosed a data breach affecting some customer accounts."
    )
    clf = EventClassifier(
        zero_shot_inference_fn=_zero_shot_stub(
            [("CYBER_OPERATIONAL", 0.55)] + [(t, 0.04) for t in EVENT_TYPES if t != "CYBER_OPERATIONAL"]
        )
    )
    result = clf.classify_text("Mixed update", text)
    assert result.method == "zero_shot"
    assert result.event_type == "CYBER_OPERATIONAL"
    # Both ambiguous rule hits are surfaced for transparency.
    assert any(r.startswith("CREDIT:") for r in result.matched_rules)
    assert any(r.startswith("CYBER_OPERATIONAL:") for r in result.matched_rules)


# --- unclassified fallback ---------------------------------------------------


def test_unclassified_for_low_confidence_zero_shot_result():
    clf = EventClassifier(
        min_zero_shot_confidence=0.33,
        zero_shot_inference_fn=_zero_shot_stub(
            [("PRODUCT_TECH", 0.12)] + [(t, 0.08) for t in EVENT_TYPES if t != "PRODUCT_TECH"]
        ),
    )
    result = clf.classify_text("Irrelevant", "A local bakery won an award for its sourdough bread.")
    assert result.event_type == UNCLASSIFIED
    assert result.method == "zero_shot"
    assert result.classifier_score == pytest.approx(0.12)


def test_unclassified_when_zero_shot_disabled_and_no_rule_match():
    clf = EventClassifier(enable_zero_shot=False)
    result = clf.classify_text("Irrelevant", "A local bakery won an award for its sourdough bread.")
    assert result.event_type == UNCLASSIFIED
    assert result.method == "unclassified"
    assert result.classifier_score is None


# --- empty input -------------------------------------------------------------


def test_empty_title_and_text_raises():
    clf = EventClassifier(zero_shot_inference_fn=_failing_zero_shot)
    with pytest.raises(ValueError):
        clf.classify_text("", "")
    with pytest.raises(ValueError):
        clf.classify_text("   ", "   ")


# --- malformed / missing model scores ---------------------------------------


def test_malformed_zero_shot_output_missing_scores_key_raises():
    def _bad_infer(text, candidate_labels):
        return {"labels": list(candidate_labels)}  # no "scores"

    clf = EventClassifier(zero_shot_inference_fn=_bad_infer)
    with pytest.raises(ValueError, match="Malformed zero-shot output"):
        clf.classify_text("Irrelevant", "A local bakery won an award.")


def test_zero_shot_output_with_mismatched_lengths_raises():
    def _bad_infer(text, candidate_labels):
        return {"labels": list(candidate_labels), "scores": [0.5]}

    clf = EventClassifier(zero_shot_inference_fn=_bad_infer)
    with pytest.raises(ValueError, match="equal-length"):
        clf.classify_text("Irrelevant", "A local bakery won an award.")


def test_zero_shot_output_with_unrecognized_label_raises():
    def _bad_infer(text, candidate_labels):
        return {"labels": ["NOT_A_REAL_CLASS"], "scores": [0.9]}

    clf = EventClassifier(zero_shot_inference_fn=_bad_infer)
    with pytest.raises(ValueError, match="Unrecognized event label"):
        clf.classify_text("Irrelevant", "A local bakery won an award.")


def test_zero_shot_output_with_non_numeric_score_raises():
    def _bad_infer(text, candidate_labels):
        return {"labels": [candidate_labels[0]], "scores": ["not-a-number"]}

    clf = EventClassifier(zero_shot_inference_fn=_bad_infer)
    with pytest.raises(ValueError, match="Non-numeric score"):
        clf.classify_text("Irrelevant", "A local bakery won an award.")


def test_parse_zero_shot_output_is_a_pure_function():
    parsed = _parse_zero_shot_output({"labels": ["CREDIT", "EARNINGS"], "scores": [0.7, 0.3]})
    assert parsed == {"CREDIT": 0.7, "EARNINGS": 0.3}


# --- long text handling: rules scan full text; zero-shot is bounded --------


def test_rules_scan_full_text_even_beyond_zero_shot_bound():
    filler = ("quarterly performance discussion and general commentary " * 60).strip()
    text = filler + " The lenders and borrower finalized a new credit facility today."
    assert len(text.split()) > 300  # exceeds DEFAULT_MAX_WORDS_FOR_ZERO_SHOT

    clf = EventClassifier(
        zero_shot_inference_fn=_failing_zero_shot,  # must not be needed - rule should resolve it
        max_words_for_zero_shot=50,
    )
    result = clf.classify_text("Long filing", text)
    assert result.event_type == "CREDIT"
    assert result.method == "rule"


def test_zero_shot_input_is_bounded_and_truncation_flag_set_for_long_text():
    filler_words = [f"word{i}" for i in range(500)]
    long_text = " ".join(filler_words)  # no keyword rules will match this

    captured = {}

    def _capturing_infer(text, candidate_labels):
        captured["text"] = text
        return {"labels": ["EARNINGS"], "scores": [0.5]}

    clf = EventClassifier(zero_shot_inference_fn=_capturing_infer, max_words_for_zero_shot=50)
    result = clf.classify_text("Title here", long_text)

    assert result.zero_shot_text_truncated is True
    # bounded input should contain far fewer words than the original 500
    assert len(captured["text"].split()) < 100


def test_build_zero_shot_input_not_truncated_for_short_text():
    bounded, truncated = _build_zero_shot_input("Title", "short body text here", max_words=50)
    assert truncated is False
    assert "Title" in bounded
    assert "short body text here" in bounded


# --- classify() convenience wrapper over UnifiedDocument --------------------


def test_classify_accepts_unified_document():
    doc = UnifiedDocument(
        document_id="doc-1",
        source_type="filing",
        source_name="SEC_EDGAR",
        publisher="Tesla, Inc.",
        timestamp="2026-09-29T00:00:00Z",
        title="Credit agreement",
        text="Tesla entered into a new revolving credit facility with its lenders.",
        url=None,
    )
    clf = EventClassifier(zero_shot_inference_fn=_failing_zero_shot)
    result = clf.classify(doc)
    assert result.event_type == "CREDIT"
    assert result.method == "rule"


# --- helper function unit tests ---------------------------------------------


def test_normalize_for_rules_lowercases_and_collapses_whitespace():
    assert _normalize_for_rules("  Credit   FACILITY\nhere  ") == "credit facility here"


def test_match_keyword_rules_finds_multiple_categories():
    hits = _match_keyword_rules(_normalize_for_rules("credit facility and a data breach occurred"))
    assert set(hits.keys()) == {"CREDIT", "CYBER_OPERATIONAL"}


def test_event_type_taxonomy_matches_master_context_exactly():
    assert EVENT_TYPES == (
        "CREDIT",
        "RATING_ACTION",
        "REGULATORY_LEGAL",
        "MACRO_CENTRAL_BANK",
        "GEOPOLITICAL",
        "M&A",
        "CYBER_OPERATIONAL",
        "SUPPLY_CHAIN",
        "EARNINGS",
        "PRODUCT_TECH",
        "COMMODITY",
        "MANAGEMENT_GOVERNANCE",
    )
