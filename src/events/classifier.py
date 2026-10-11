"""
Event classification.

Classifies a document (title + body) into the 12-class event taxonomy
from the master context, Section 9 ("EVENT CLASSIFICATION"):

    CREDIT, RATING_ACTION, REGULATORY_LEGAL, MACRO_CENTRAL_BANK,
    GEOPOLITICAL, M&A, CYBER_OPERATIONAL, SUPPLY_CHAIN, EARNINGS,
    PRODUCT_TECH, COMMODITY, MANAGEMENT_GOVERNANCE

plus an UNCLASSIFIED fallback for input that does not support a
defensible assignment to any of the 12 (it is a fallback, not a 13th
core category).

Hybrid approach (per the master context: "rules may override very
strong keyword patterns"):
1. Deterministic, high-precision keyword rules are checked FIRST,
   scanning the full title+body text (not bounded/truncated - this is
   cheap regex/substring work, unlike model inference).
   - Exactly one category matched  -> that category wins outright,
     method="rule". No model call is made at all in this case.
   - Zero or 2+ categories matched -> rules alone can't resolve it
     (nothing matched, or it's genuinely ambiguous), so control falls
     through to zero-shot classification. Any rule hits found in this
     ambiguous case are still returned in `matched_rules` for
     transparency, tagged with which category each phrase belongs to.
2. Zero-shot semantic classification (a Hugging Face NLI
   zero-shot-classification pipeline) scores all 12 classes as
   candidate labels and the top-scoring one is used, subject to a
   minimum-confidence threshold below which the result is UNCLASSIFIED
   instead.

Model choice is configurable (`zero_shot_model_name`) so it can be
swapped without rewriting the classifier. The default,
"valhalla/distilbart-mnli-12-3", is a distilled MNLI model: much
smaller and faster on CPU than "facebook/bart-large-mnli" (the common
zero-shot default), at a modest accuracy cost that's acceptable for a
hackathon-scale demo under CPU/latency/download-size constraints. Swap
in `facebook/bart-large-mnli` (or anything else exposing the standard
zero-shot-classification pipeline interface) via the constructor if
your environment can afford it.

Like sentiment.py, the real `transformers` import is lazy (only inside
`_build_default_pipeline`), and the zero-shot step itself can be
injected (`zero_shot_inference_fn`) or disabled entirely
(`enable_zero_shot=False`), so the unit tests never need
`transformers`/`torch` installed or network access - see
tests/test_event_classifier.py.

Long-document handling (requirement: never silently truncate a long
SEC filing and then claim to have classified the whole thing):
- Deterministic rules scan the COMPLETE title+body text, always.
- Zero-shot classification runs on a bounded prefix only
  (title + first `max_words_for_zero_shot` words of body) for latency
  and model-context reasons - this is a deliberate, documented,
  lightweight bounding strategy, not silent truncation. When bounding
  actually drops text, the result's `zero_shot_text_truncated` flag is
  set to True so callers can tell.

Separation of concerns: this module has no dependency on impact
scoring, event resolution, novelty/velocity, the FastAPI layer, or
Module B, and is never imported by them.
"""
from __future__ import annotations

import re
from typing import Callable, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, field_validator

from src.utils.schema import UnifiedDocument

EVENT_TYPES: Tuple[str, ...] = (
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
UNCLASSIFIED = "UNCLASSIFIED"
_VALID_EVENT_TYPE_VALUES = set(EVENT_TYPES) | {UNCLASSIFIED}

ClassificationMethod = Literal["rule", "zero_shot", "unclassified"]

DEFAULT_ZERO_SHOT_MODEL = "valhalla/distilbart-mnli-12-3"
DEFAULT_MIN_ZERO_SHOT_CONFIDENCE = 0.33
# Conservative word-count bound for the zero-shot prefix - same
# rationale as sentiment.py's chunking constant: approximate, not a
# real tokenizer-based limit, chosen to keep inference latency low.
DEFAULT_MAX_WORDS_FOR_ZERO_SHOT = 300

# A raw zero-shot output mirrors Hugging Face's zero-shot-classification
# pipeline: {"sequence": ..., "labels": [...], "scores": [...]}.
ZeroShotRawOutput = Dict[str, object]
ZeroShotInferenceFn = Callable[[str, List[str]], ZeroShotRawOutput]

# Deterministic, high-precision keyword rules. Deliberately small and
# conservative (favor precision over recall) - these are meant to be
# unambiguous "override" signals, not an exhaustive taxonomy. Extend
# cautiously; a phrase that's common outside its intended category
# causes false "rule" overrides.
_KEYWORD_RULES: Dict[str, List[str]] = {
    "CREDIT": [
        "credit facility",
        "credit agreement",
        "term loan",
        "revolving credit",
        "delayed draw term loan",
        "loan covenant",
        "covenant breach",
        "debt restructuring",
    ],
    "RATING_ACTION": [
        "credit rating",
        "ratings action",
        "downgraded",
        "upgraded by",
        "placed on credit watch",
        "moody's",
        "s&p global ratings",
        "fitch ratings",
    ],
    "REGULATORY_LEGAL": [
        "securities and exchange commission",
        "class action lawsuit",
        "regulatory investigation",
        "consent decree",
        "subpoena",
        "antitrust probe",
        "settlement agreement",
        "safety probe",
        "ftc investigation",
    ],
    "MACRO_CENTRAL_BANK": [
        "federal reserve",
        "fomc",
        "interest rate decision",
        "rate hike",
        "rate cut",
        "central bank",
        "monetary policy",
    ],
    "GEOPOLITICAL": [
        "economic sanctions",
        "geopolitical tension",
        "trade war",
        "tariffs imposed",
        "armed conflict",
        "military conflict",
        "export ban",
    ],
    "M&A": [
        "definitive merger agreement",
        "agreed to acquire",
        "tender offer",
        "all-stock deal",
        "merger agreement",
        "acquisition of",
    ],
    "CYBER_OPERATIONAL": [
        "data breach",
        "cyberattack",
        "ransomware attack",
        "security incident",
        "service outage",
        "network outage",
    ],
    "SUPPLY_CHAIN": [
        "supply chain disruption",
        "chip shortage",
        "production cut",
        "component shortage",
        "shipping delay",
        "logistics disruption",
    ],
    "EARNINGS": [
        "results of operations",
        "quarterly earnings",
        "earnings per share",
        "revenue guidance",
        "beat analyst estimates",
        "missed analyst estimates",
        "financial results for the",
    ],
    "PRODUCT_TECH": [
        "product launch",
        "unveiled its new",
        "next-generation chip",
        "software update",
        "patent granted",
    ],
    "COMMODITY": [
        "crude oil prices",
        "commodity prices",
        "natural gas prices",
        "copper prices",
        "gold prices",
    ],
    "MANAGEMENT_GOVERNANCE": [
        "chief executive officer resigned",
        "ceo resignation",
        "board of directors appointed",
        "executive departure",
        "named new chief executive",
    ],
}


class EventClassification(BaseModel):
    """Structured event classification result."""

    event_type: str
    # Nullable (rule-based matches have no model probability at all).
    # When present (zero-shot path), this is the top candidate label's
    # score from the zero-shot pipeline's softmax over the given
    # candidate labels - NOT a calibrated probability.
    classifier_score: Optional[float] = None
    method: ClassificationMethod
    matched_rules: List[str] = []
    # True only when the zero-shot step ran on a bounded prefix that
    # actually dropped text (see module docstring). Always False for
    # the "rule" and "unclassified" (zero-shot-disabled) methods, since
    # rules always scan the full text and "unclassified" via a disabled
    # zero-shot step never truncates anything because it never runs.
    zero_shot_text_truncated: bool = False

    @field_validator("event_type")
    @classmethod
    def event_type_must_be_valid(cls, v: str) -> str:
        if v not in _VALID_EVENT_TYPE_VALUES:
            raise ValueError(
                f"event_type '{v}' is not one of the 12 taxonomy classes or {UNCLASSIFIED}"
            )
        return v


def _normalize_for_rules(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _match_keyword_rules(normalized_text: str) -> Dict[str, List[str]]:
    """Return {event_type: [matched phrase, ...]} for every category with >=1 hit."""
    hits: Dict[str, List[str]] = {}
    for event_type, phrases in _KEYWORD_RULES.items():
        matched = [p for p in phrases if p in normalized_text]
        if matched:
            hits[event_type] = matched
    return hits


def _build_zero_shot_input(title: str, text: str, max_words: int) -> Tuple[str, bool]:
    """
    Build the bounded (title + first `max_words` words of body) input
    used for zero-shot classification, and whether that bound actually
    dropped any text.
    """
    words = text.split()
    truncated = len(words) > max_words
    bounded_body = " ".join(words[:max_words])
    if title:
        combined = f"{title}. {bounded_body}".strip()
    else:
        combined = bounded_body
    return combined, truncated


def _parse_zero_shot_output(raw: ZeroShotRawOutput) -> Dict[str, float]:
    """
    Pure function: validate and convert a raw zero-shot pipeline output
    into a {label: score} dict. Raises ValueError explicitly on any
    malformed or missing data, rather than silently degrading.
    """
    if not isinstance(raw, dict) or "labels" not in raw or "scores" not in raw:
        raise ValueError(
            "Malformed zero-shot output: expected a dict with 'labels' and 'scores' keys"
        )
    labels = raw["labels"]
    scores = raw["scores"]
    if not isinstance(labels, list) or not isinstance(scores, list):
        raise ValueError("Malformed zero-shot output: 'labels' and 'scores' must be lists")
    if len(labels) == 0 or len(labels) != len(scores):
        raise ValueError(
            "Malformed zero-shot output: 'labels' and 'scores' must be equal-length and non-empty"
        )

    result: Dict[str, float] = {}
    for label, score in zip(labels, scores):
        if label not in EVENT_TYPES:
            raise ValueError(
                f"Unrecognized event label '{label}' in zero-shot output; "
                f"expected one of {EVENT_TYPES}"
            )
        try:
            result[label] = float(score)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Non-numeric score for label '{label}' in zero-shot output"
            ) from exc
    return result


class EventClassifier:
    """
    Hybrid rule + zero-shot event classifier. Loads the zero-shot model
    once per instance (unless `zero_shot_inference_fn` is injected, or
    `enable_zero_shot=False`), and reuses it across calls.
    """

    def __init__(
        self,
        zero_shot_model_name: str = DEFAULT_ZERO_SHOT_MODEL,
        zero_shot_inference_fn: Optional[ZeroShotInferenceFn] = None,
        enable_zero_shot: bool = True,
        min_zero_shot_confidence: float = DEFAULT_MIN_ZERO_SHOT_CONFIDENCE,
        max_words_for_zero_shot: int = DEFAULT_MAX_WORDS_FOR_ZERO_SHOT,
    ) -> None:
        self.model_name = zero_shot_model_name
        self._enable_zero_shot = enable_zero_shot
        self._min_confidence = min_zero_shot_confidence
        self._max_words = max_words_for_zero_shot
        self._infer: Optional[ZeroShotInferenceFn] = None
        if enable_zero_shot:
            self._infer = zero_shot_inference_fn or self._build_default_pipeline(
                zero_shot_model_name
            )

    @staticmethod
    def _build_default_pipeline(model_name: str) -> ZeroShotInferenceFn:
        """
        Build the real Hugging Face zero-shot pipeline. Imports
        `transformers` lazily so this only runs when nothing has been
        injected for testing and zero-shot hasn't been disabled.
        Requires network access the first time (to download weights)
        and the `transformers` (+ a backend like `torch`) package.
        """
        from transformers import pipeline  # local import - see module docstring

        hf_pipeline = pipeline("zero-shot-classification", model=model_name)

        def _infer(text: str, candidate_labels: List[str]) -> ZeroShotRawOutput:
            return hf_pipeline(text, candidate_labels=candidate_labels, multi_label=False)

        return _infer

    def classify_text(self, title: str, text: str) -> EventClassification:
        combined_raw = f"{title or ''} {text or ''}".strip()
        if not combined_raw:
            raise ValueError("title and text must not both be empty")

        normalized = _normalize_for_rules(combined_raw)
        rule_hits = _match_keyword_rules(normalized)

        if len(rule_hits) == 1:
            event_type, matched = next(iter(rule_hits.items()))
            return EventClassification(
                event_type=event_type,
                classifier_score=None,
                method="rule",
                matched_rules=matched,
            )

        # Zero, or 2+ (ambiguous), rule-category hits. Surface any hits
        # found for transparency even though they didn't resolve things.
        ambiguous_matched_rules = sorted(
            f"{event_type}:{phrase}"
            for event_type, phrases in rule_hits.items()
            for phrase in phrases
        )

        if not self._enable_zero_shot or self._infer is None:
            return EventClassification(
                event_type=UNCLASSIFIED,
                classifier_score=None,
                method="unclassified",
                matched_rules=ambiguous_matched_rules,
            )

        zero_shot_text, truncated = _build_zero_shot_input(title or "", text or "", self._max_words)
        raw_output = self._infer(zero_shot_text, list(EVENT_TYPES))
        label_scores = _parse_zero_shot_output(raw_output)

        top_label = max(label_scores, key=label_scores.get)  # type: ignore[arg-type]
        top_score = label_scores[top_label]

        event_type = top_label if top_score >= self._min_confidence else UNCLASSIFIED

        return EventClassification(
            event_type=event_type,
            classifier_score=top_score,
            method="zero_shot",
            matched_rules=ambiguous_matched_rules,
            zero_shot_text_truncated=truncated,
        )

    def classify(self, document: UnifiedDocument) -> EventClassification:
        """Convenience wrapper over classify_text() for a UnifiedDocument."""
        return self.classify_text(document.title, document.text)
