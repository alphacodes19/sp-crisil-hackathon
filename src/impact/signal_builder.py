"""
Signal builder.

Combines the outputs of the existing, independently-built components:

    UnifiedDocument          (src/utils/schema.py)
    EntityResolver           (src/entities/resolver.py)
    SentimentAnalyzer        (src/nlp/sentiment.py)
    EventClassifier          (src/events/classifier.py)
    EventResolver            (src/events/resolution.py)         [Milestone 7]
    compute_impact           (src/impact/impact_engine.py)      [Milestone 7]

into one structured `Signal` per document, matching the core signal
contract in Section 13 of the master context. This module does not
duplicate any of those components' logic - it only orchestrates calls
to their existing public interfaces.

Two Signal-construction paths, intentionally kept distinct (Milestone 7
added the second one without changing the first):

1. `build_signal()` - builds a Signal from ONE document in isolation,
   with no peer-document context at all. It cannot genuinely resolve
   this document's event against others, so (unchanged since
   Milestone 6): `event_id` stays a PROVISIONAL, document-scoped ID
   (`"provisional-event-<document_id>"`), and `impact_score`/`evidence`
   stay `None`. This path still exists for ad-hoc/single-document use
   and its existing tests/behavior are preserved exactly.

2. `process_documents()` / `run_replay_pipeline()` - the BATCH path.
   Here there IS peer-document context, so documents are run through
   real event resolution (`EventResolver`) and the real impact engine
   (`compute_impact`): `event_id` becomes a genuine resolved event ID
   (never provisional), and `impact_score`, `impact_breakdown`,
   `impact_method`, `evidence`, and the full `event_resolution` block
   are populated for real. Per the master context: "Remove the
   provisional event-ID behavior only where the resolver has produced
   a valid resolved ID" - which in this design is every document that
   goes through the batch path (even a singleton "event" of one
   document still gets a real resolved ID; it just also gets
   corroboration=0.0 and an evidence note making clear it is NOT
   independently corroborated by itself).

IDs are fully deterministic: `signal_id`/provisional `event_id` are
derived only from `document_id`; resolved `event_id`s are derived from
the sorted set of member document_ids (see resolution.py). Running the
replay pipeline twice therefore produces byte-for-byte identical
signal_id/event_id values - required for reproducible replay.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple

from pydantic import BaseModel

from src.entities.resolver import EntityResolver
from src.events.classifier import EventClassifier
from src.events.resolution import DocumentFeatures, EventResolver, ResolvedEvent
from src.impact.impact_engine import ImpactFactorBreakdown, ImpactResult, compute_impact
from src.ingestion.gdelt import load_replay as load_gdelt_replay
from src.ingestion.sec_edgar import load_replay as load_sec_replay
from src.nlp.sentiment import SentimentAnalyzer, SentimentResult
from src.utils.io import write_jsonl
from src.utils.schema import UnifiedDocument

DEFAULT_SIGNALS_PATH = Path(__file__).resolve().parents[2] / "data" / "signals.jsonl"

EventScope = Literal["single_entity", "multi_entity", "unmapped"]


class SignalEntity(BaseModel):
    """One resolved entity mention, carried over from EntityMatch."""

    canonical_name: str
    ticker: str
    matched_alias: str
    match_type: str


class SignalSentiment(BaseModel):
    """Carried over from SentimentResult - direction/polarity only."""

    label: str
    score: float
    model_score: float


class SignalEventClassification(BaseModel):
    """Carried over from EventClassification."""

    event_type: str
    classifier_score: Optional[float] = None
    method: str
    matched_rules: List[str] = []


class SignalProvenance(BaseModel):
    """Enough to trace a signal back to its original source document."""

    document_id: str
    source_name: str  # e.g. "GDELT" or "SEC_EDGAR"
    source_type: str  # e.g. "news" or "filing"
    publisher: str
    url: Optional[str] = None


class SignalEventResolution(BaseModel):
    """
    Resolved-event-level metadata (Milestone 7), projected from
    `ResolvedEvent` (src/events/resolution.py). Only present
    (non-`None`) on signals built via the batch path - see module
    docstring.
    """

    resolved_event_id: str
    first_seen: datetime
    latest_seen: datetime
    document_count: int  # raw count, including exact-URL duplicates
    distinct_source_count: int  # after de-duplicating exact-URL duplicates
    distinct_publisher_count: int
    novelty: float
    velocity: float
    corroboration: float


class Signal(BaseModel):
    """The structured, persisted output of the risk-signal pipeline."""

    signal_id: str
    event_id: str  # provisional (build_signal) or resolved (batch path) - see module docstring
    document_id: str
    timestamp: datetime
    source: SignalProvenance
    entities: List[SignalEntity] = []
    sentiment: SignalSentiment
    event: SignalEventClassification
    scope: EventScope
    # impact_score/impact_breakdown/impact_method: None on the single-
    # document path (build_signal - no real impact engine inputs
    # available without peer-document context); populated for real on
    # the batch path. evidence/event_resolution follow the same split.
    impact_score: Optional[float] = None
    impact_breakdown: Optional[ImpactFactorBreakdown] = None
    impact_method: Optional[str] = None
    evidence: Optional[str] = None
    event_resolution: Optional[SignalEventResolution] = None


def _derive_scope(entities: List[SignalEntity]) -> EventScope:
    """
    The only scope distinction we can defensibly make right now, using
    only what's already resolved for this document - NOT sector/
    market/macro scope, which would need data we don't have yet
    (the alias table has no sector/country columns - see Milestone 3).
    """
    if len(entities) == 0:
        return "unmapped"
    if len(entities) == 1:
        return "single_entity"
    return "multi_entity"


def build_signal(
    document: UnifiedDocument,
    resolver: EntityResolver,
    sentiment_analyzer: SentimentAnalyzer,
    event_classifier: EventClassifier,
) -> Signal:
    """
    Build one Signal from one document, using the given (already
    constructed) component instances. Raises whatever the underlying
    components raise (e.g. ValueError on empty text) - callers that
    need batch resilience should use `process_documents()` instead,
    which isolates per-document failures.
    """
    entity_matches = resolver.resolve(document)
    sentiment_result = sentiment_analyzer.analyze(document.text)
    event_result = event_classifier.classify(document)

    entities = [
        SignalEntity(
            canonical_name=m.canonical_name,
            ticker=m.ticker,
            matched_alias=m.matched_alias,
            match_type=m.match_type,
        )
        for m in entity_matches
    ]

    return Signal(
        signal_id=f"signal-{document.document_id}",
        event_id=f"provisional-event-{document.document_id}",
        document_id=document.document_id,
        timestamp=document.timestamp,
        source=SignalProvenance(
            document_id=document.document_id,
            source_name=document.source_name,
            source_type=document.source_type,
            publisher=document.publisher,
            url=document.url,
        ),
        entities=entities,
        sentiment=SignalSentiment(
            label=sentiment_result.label,
            score=sentiment_result.score,
            model_score=sentiment_result.model_score,
        ),
        event=SignalEventClassification(
            event_type=event_result.event_type,
            classifier_score=event_result.classifier_score,
            method=event_result.method,
            matched_rules=event_result.matched_rules,
        ),
        scope=_derive_scope(entities),
        impact_score=None,
        evidence=None,
    )


@dataclass(frozen=True)
class SignalBuildFailure:
    """One document that failed to produce a signal, and why."""

    document_id: str
    error: str


def _build_resolved_signal(
    feat: DocumentFeatures,
    sentiment_result: SentimentResult,
    resolved_event: ResolvedEvent,
    impact_result: ImpactResult,
) -> Signal:
    """
    The batch-path counterpart to `build_signal()`: same nested-model
    shape, but `event_id` is the real resolved event ID, and
    impact_score/impact_breakdown/impact_method/evidence/
    event_resolution are populated for real instead of staying `None`.
    """
    document = feat.document
    entities = [
        SignalEntity(
            canonical_name=m.canonical_name,
            ticker=m.ticker,
            matched_alias=m.matched_alias,
            match_type=m.match_type,
        )
        for m in feat.entities
    ]

    evidence = (
        f"{resolved_event.distinct_publisher_count} distinct publisher(s) / "
        f"{resolved_event.distinct_source_count} distinct source(s) / "
        f"{resolved_event.document_count} total document(s)"
    )
    if resolved_event.distinct_publisher_count <= 1:
        evidence += " - single-source, not independently corroborated"

    return Signal(
        signal_id=f"signal-{document.document_id}",
        event_id=resolved_event.resolved_event_id,
        document_id=document.document_id,
        timestamp=document.timestamp,
        source=SignalProvenance(
            document_id=document.document_id,
            source_name=document.source_name,
            source_type=document.source_type,
            publisher=document.publisher,
            url=document.url,
        ),
        entities=entities,
        sentiment=SignalSentiment(
            label=sentiment_result.label,
            score=sentiment_result.score,
            model_score=sentiment_result.model_score,
        ),
        event=SignalEventClassification(
            event_type=feat.event_classification.event_type,
            classifier_score=feat.event_classification.classifier_score,
            method=feat.event_classification.method,
            matched_rules=feat.event_classification.matched_rules,
        ),
        scope=_derive_scope(entities),
        impact_score=impact_result.score,
        impact_breakdown=impact_result.breakdown,
        impact_method=impact_result.method,
        evidence=evidence,
        event_resolution=SignalEventResolution(
            resolved_event_id=resolved_event.resolved_event_id,
            first_seen=resolved_event.first_seen,
            latest_seen=resolved_event.latest_seen,
            document_count=resolved_event.document_count,
            distinct_source_count=resolved_event.distinct_source_count,
            distinct_publisher_count=resolved_event.distinct_publisher_count,
            novelty=resolved_event.novelty,
            velocity=resolved_event.velocity,
            corroboration=resolved_event.corroboration,
        ),
    )


def process_documents(
    documents: Iterable[UnifiedDocument],
    resolver: Optional[EntityResolver] = None,
    sentiment_analyzer: Optional[SentimentAnalyzer] = None,
    event_classifier: Optional[EventClassifier] = None,
    event_resolver: Optional[EventResolver] = None,
) -> Tuple[List[Signal], List[SignalBuildFailure]]:
    """
    Build a Signal for each document, now resolution- and impact-aware
    (Milestone 7): entity resolution + sentiment + event classification
    still run per document with failure isolation exactly as before
    (one bad document never takes down the batch), but the documents
    that DO succeed are then run through real cross-document event
    resolution (`EventResolver`) and the real impact engine
    (`compute_impact`) before their Signals are built - both of which
    need the whole successful batch at once, not just one document.
    Never fabricates a sentiment, event, resolution, or impact result
    for a document that failed - it's simply absent from `signals`.

    If `resolver`/`sentiment_analyzer`/`event_classifier`/
    `event_resolver` aren't given, real default instances are
    constructed (which, for the sentiment analyzer and event
    classifier, means loading real Hugging Face models). Tests should
    always inject stub instances instead.
    """
    resolver = resolver or EntityResolver()
    sentiment_analyzer = sentiment_analyzer or SentimentAnalyzer()
    event_classifier = event_classifier or EventClassifier()
    event_resolver = event_resolver or EventResolver()

    features: List[DocumentFeatures] = []
    sentiment_by_document_id: Dict[str, SentimentResult] = {}
    failures: List[SignalBuildFailure] = []

    for document in documents:
        try:
            entities = resolver.resolve(document)
            sentiment_result = sentiment_analyzer.analyze(document.text)
            event_result = event_classifier.classify(document)
        except Exception as exc:  # noqa: BLE001 - intentionally broad: ANY single-document
            # failure (malformed model output, unexpected empty field, a
            # transient model/network error, ...) must not take down the
            # rest of the batch. The failure is still recorded, not hidden.
            warnings.warn(f"Skipping document {document.document_id}: {exc}", stacklevel=2)
            failures.append(SignalBuildFailure(document_id=document.document_id, error=str(exc)))
            continue
        features.append(DocumentFeatures(document=document, entities=entities, event_classification=event_result))
        sentiment_by_document_id[document.document_id] = sentiment_result

    # Cross-document step - needs the whole successful batch at once.
    resolved_map = event_resolver.resolve(features)

    signals: List[Signal] = []
    for feat in features:
        document_id = feat.document.document_id
        resolved_event = resolved_map[document_id]
        impact_result = compute_impact(
            event_type=feat.event_classification.event_type,
            entities=feat.entities,
            novelty=resolved_event.novelty,
            velocity=resolved_event.velocity,
            corroboration=resolved_event.corroboration,
        )
        signals.append(
            _build_resolved_signal(feat, sentiment_by_document_id[document_id], resolved_event, impact_result)
        )

    return signals, failures


def run_replay_pipeline(
    output_path: Optional[Path] = None,
    resolver: Optional[EntityResolver] = None,
    sentiment_analyzer: Optional[SentimentAnalyzer] = None,
    event_classifier: Optional[EventClassifier] = None,
    event_resolver: Optional[EventResolver] = None,
) -> Tuple[List[Signal], List[SignalBuildFailure]]:
    """
    End-to-end replay pipeline: load GDELT + SEC replay documents
    (same normalized UnifiedDocument schema for both - Section 5 of the
    master context), run them through `process_documents()` (now
    resolution- and impact-aware), and persist the resulting signals to
    `output_path` (default: data/signals.jsonl) via the existing
    `write_jsonl` helper (no duplicate persistence logic).

    Deterministic by construction: replay documents have fixed,
    deterministic document_ids, and resolved event IDs are derived
    deterministically from the sorted set of member document_ids (see
    resolution.py), so running this twice with the same component
    instances/behavior produces byte-for-byte identical
    signal_id/event_id values.
    """
    documents: List[UnifiedDocument] = list(load_gdelt_replay()) + list(load_sec_replay())
    signals, failures = process_documents(documents, resolver, sentiment_analyzer, event_classifier, event_resolver)
    write_jsonl(output_path or DEFAULT_SIGNALS_PATH, signals)
    return signals, failures
