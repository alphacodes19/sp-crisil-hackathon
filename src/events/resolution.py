"""
Event resolution.

Groups multiple documents that plausibly describe the SAME underlying
real-world event into one resolved event, replacing the provisional,
one-document-per-event placeholder used directly by `build_signal()`
in src/impact/signal_builder.py (Milestone 6) with a genuine (if still
heuristic) grouping, for documents that go through the batch pipeline.

Per the master context: "Do not merge documents merely because they
mention the same company." Merging two documents requires ALL of:
  1. overlapping resolved entity tickers (non-empty intersection);
  2. the SAME classified event_type, and it must not be UNCLASSIFIED
     (an unclassified document carries too little signal to safely
     group - it always stays in its own singleton event);
  3. temporal proximity (within `max_time_gap`, default 72h);
  4. sufficient title similarity - a lightweight, deterministic
     Jaccard token-overlap measure (see `_text_similarity`), NOT an
     embedding model, per the instruction to prefer a lightweight
     deterministic method over downloading another model.
All four must hold; any one failing keeps the two documents separate.
This is intentionally conservative (favors precision over recall).

Clustering is a single-pass, deterministic greedy grouping over
documents sorted by (timestamp, document_id) for full reproducibility:
for each document, check it against every existing cluster's member
documents, in cluster-creation order; join the first cluster where it
matches ANY member, else start a new cluster. This is
O(n * avg_cluster_size) - fine at replay/demo scale - and deterministic
run-over-run regardless of input ordering (inputs are always re-sorted
first). Known limitation, not solved here: this greedy single-pass
approach can leave two clusters separate that a smarter transitive/
connected-components merge might combine (e.g. A matches B, B matches
C, but A doesn't directly match C) - acceptable for a prototype,
documented rather than silently assumed away.

Processing documents in timestamp order also directly drives novelty
(see `_compute_novelty`): a resolved event's novelty is judged, via
lightweight text similarity, only against RELEVANT prior events (same
event_type, overlapping tickers, within a lookback window) that were
already "known" - i.e. whose first document appeared earlier in time -
never against events that are chronologically later but happened to be
processed earlier in some other ordering. Reordering the input does
not change the result because we always re-sort before clustering.

Known data-availability limitation: this prototype has no persistent
cross-run history store. "Previously observed relevant events" for
novelty purposes means "earlier, within this same batch/replay run" -
not a durable history across separate pipeline runs. This is
documented, not hidden.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, FrozenSet, List, Optional

from src.entities.resolver import EntityMatch
from src.events.classifier import EventClassification, UNCLASSIFIED
from src.utils.schema import UnifiedDocument

# --- clustering thresholds (documented assumptions, not measured) ----------

DEFAULT_MAX_TIME_GAP_HOURS = 72.0
# Deliberately a low bar: titles describing the same real-world event
# from different publishers are often phrased quite differently, so
# title similarity is used as ONE of four required signals, not the
# sole decider.
DEFAULT_MIN_TITLE_SIMILARITY = 0.2

# --- novelty (documented heuristic, not a trained/calibrated model) --------
#
# Novelty is judged by TEXT SIMILARITY against relevant prior resolved
# events (same event_type, overlapping entity tickers, first_seen
# within `novelty_lookback_hours` of this event), using the same
# lightweight, deterministic Jaccard token-overlap measure as merge
# decisions (`_text_similarity`) - no embedding model. The most similar
# relevant prior event sets the novelty: a near-duplicate headline
# about the same company/event-type gets LOW novelty (it's not new
# information), while a genuinely different headline about the same
# company/event-type gets HIGH novelty (same recurring category, but a
# distinct development) - novelty = 1.0 - max_similarity, floored so it
# is never exactly 0 (we can't be certain no new facet exists even for
# a near-identical headline).
NOVELTY_FLOOR = 0.1
# Used when we can't meaningfully compare at all (UNCLASSIFIED
# event_type, no entities resolved, or no representative title text) -
# a neutral midpoint, not a guess at either extreme.
NOVELTY_NEUTRAL_DEFAULT = 0.5
# How far back "previously observed relevant events" looks for novelty
# purposes. Deliberately much longer than the merge time-gap (72h):
# merging requires near-term proximity, but judging whether something
# is "novel" reasonably looks back further (here, 30 days) - documented
# assumption, not a measured value.
DEFAULT_NOVELTY_LOOKBACK_HOURS = 24.0 * 30.0

# --- velocity (documented heuristic, not a measured rate model) ------------

# "Many mentions per hour" reference point that saturates velocity to
# 1.0. Chosen to be sensibly sensitive at small-scale replay/demo
# volumes, not derived from real mention-rate statistics.
DEFAULT_VELOCITY_REFERENCE_RATE_PER_HOUR = 2.0
# Floor for the time span used in the mentions-per-hour calculation,
# so two documents that land at (or extremely near) the exact same
# timestamp never cause a division by zero or an unbounded spike.
DEFAULT_MIN_TIME_SPAN_HOURS = 1.0

# --- corroboration (documented heuristic) -----------------------------------

# Distinct publishers beyond the first that saturate corroboration to
# 1.0 (e.g. 4 distinct publishers -> (4-1)/3 = 1.0).
DEFAULT_CORROBORATION_REFERENCE_PUBLISHERS = 3


@dataclass(frozen=True)
class DocumentFeatures:
    """
    Per-document inputs to event resolution: the already-resolved
    entities and already-classified event type for one document. This
    is a pure projection - resolution never re-runs entity resolution
    or event classification itself (no duplicated logic).
    """

    document: UnifiedDocument
    entities: List[EntityMatch]
    event_classification: EventClassification


@dataclass(frozen=True)
class _EventHistorySignature:
    """One previously-resolved event's signature, kept only for novelty
    comparisons against later events. Built incrementally in
    chronological order by `EventResolver.resolve()` - a signature is
    only ever appended AFTER its event has been fully resolved, so a
    later event's novelty computation can never see a signature for an
    event that (chronologically) hasn't happened yet."""

    event_type: str
    tickers: FrozenSet[str]
    representative_title: str
    first_seen: datetime


@dataclass
class ResolvedEvent:
    """One cluster's aggregated, resolved-event-level metadata."""

    resolved_event_id: str
    event_type: str
    member_document_ids: List[str]
    first_seen: datetime
    latest_seen: datetime
    document_count: int  # raw count, including exact-URL duplicates
    distinct_source_count: int  # after de-duplicating exact-URL duplicates
    distinct_publisher_count: int  # among the de-duplicated sources
    novelty: float
    velocity: float
    corroboration: float


def _tokenize(text: str) -> frozenset:
    return frozenset(re.findall(r"[a-z0-9]+", text.lower()))


def _text_similarity(title_a: str, title_b: str) -> float:
    """Jaccard similarity over normalized unigram tokens. Deterministic,
    no model/embedding involved. Returns 0.0 if either title has no
    tokens at all (can't meaningfully compare)."""
    tokens_a, tokens_b = _tokenize(title_a), _tokenize(title_b)
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def _should_merge(
    a: DocumentFeatures,
    b: DocumentFeatures,
    max_time_gap: timedelta,
    min_title_similarity: float,
) -> bool:
    tickers_a = {e.ticker for e in a.entities}
    tickers_b = {e.ticker for e in b.entities}
    if not tickers_a or not tickers_b or not (tickers_a & tickers_b):
        return False  # no shared entity - "mentions the same company" alone is not evaluated here anyway

    if a.event_classification.event_type != b.event_classification.event_type:
        return False
    if a.event_classification.event_type == UNCLASSIFIED:
        return False  # too little signal to safely group

    if abs(a.document.timestamp - b.document.timestamp) > max_time_gap:
        return False

    if _text_similarity(a.document.title, b.document.title) < min_title_similarity:
        return False

    return True


def _distinct_sources(cluster: List[DocumentFeatures]) -> List[DocumentFeatures]:
    """
    One representative per distinct source. De-duplicates by exact URL
    (an exact-URL repeat is the same article re-ingested, e.g. a
    re-crawl - never independent corroboration). Known limitation, not
    solved here: full syndication of identical text under *different*
    URLs/publishers is not detected by this simple heuristic.
    Documents with no URL are never treated as duplicates of each
    other (each gets its own key).
    """
    seen_keys = set()
    distinct: List[DocumentFeatures] = []
    for feat in cluster:
        key = feat.document.url or f"__no_url__:{feat.document.document_id}"
        if key not in seen_keys:
            seen_keys.add(key)
            distinct.append(feat)
    return distinct


def _compute_novelty(
    event_type: str,
    tickers: FrozenSet[str],
    representative_title: str,
    event_timestamp: datetime,
    history: List[_EventHistorySignature],
    lookback: timedelta,
) -> float:
    """
    Lightweight, deterministic, text-similarity-based novelty (see the
    module-level comment above `NOVELTY_FLOOR` for the full rationale).

    Handles the required edge cases explicitly rather than letting them
    fall through to an arithmetic accident:
    - UNCLASSIFIED event_type, no resolved entities, or a blank/missing
      representative title -> NOVELTY_NEUTRAL_DEFAULT (can't judge
      reliably either way).
    - Empty history, or no RELEVANT prior event within the lookback
      window -> 1.0 (nothing comparable to judge it against, so treated
      as fully novel - not an error, not a 0).
    - `history` is always built chronologically by the caller (see
      `_EventHistorySignature`), so only genuinely earlier events are
      ever compared against - this function never needs to filter out
      "future" entries itself, but the `abs()` below is still a
      defensive guard for direct/unit-test callers that might pass an
      out-of-order history on purpose.
    """
    if event_type == UNCLASSIFIED or not tickers or not representative_title.strip():
        return NOVELTY_NEUTRAL_DEFAULT

    relevant_prior_titles = [
        h.representative_title
        for h in history
        if h.event_type == event_type
        and (h.tickers & tickers)
        and abs(event_timestamp - h.first_seen) <= lookback
        and h.representative_title.strip()
    ]
    if not relevant_prior_titles:
        return 1.0

    max_similarity = max(_text_similarity(representative_title, prior_title) for prior_title in relevant_prior_titles)
    return max(NOVELTY_FLOOR, 1.0 - max_similarity)


def _compute_velocity(
    document_count: int,
    first_seen: datetime,
    latest_seen: datetime,
    reference_rate_per_hour: float,
    min_time_span_hours: float,
) -> float:
    if document_count <= 1:
        return 0.0  # a single mention has no observed "increase" yet
    span_hours = max(abs((latest_seen - first_seen).total_seconds()) / 3600.0, min_time_span_hours)
    mentions_per_hour = (document_count - 1) / span_hours
    return max(0.0, min(1.0, mentions_per_hour / reference_rate_per_hour))


def _compute_corroboration(distinct_publisher_count: int, reference_publishers: float) -> float:
    if distinct_publisher_count <= 1:
        return 0.0  # never treat a single source as self-corroborating
    return max(0.0, min(1.0, (distinct_publisher_count - 1) / reference_publishers))


def _make_resolved_event_id(member_document_ids_sorted: List[str]) -> str:
    """Deterministic regardless of processing order: derived from the
    sorted set of member document IDs, so the same underlying event
    always gets the same ID across runs."""
    digest = hashlib.sha1("|".join(member_document_ids_sorted).encode("utf-8")).hexdigest()
    return f"resolved-event-{digest[:12]}"


class EventResolver:
    """Clusters documents into resolved events and computes per-event
    novelty/velocity/corroboration. See module docstring for the full
    merge-criteria and limitations."""

    def __init__(
        self,
        max_time_gap_hours: float = DEFAULT_MAX_TIME_GAP_HOURS,
        min_title_similarity: float = DEFAULT_MIN_TITLE_SIMILARITY,
        velocity_reference_rate_per_hour: float = DEFAULT_VELOCITY_REFERENCE_RATE_PER_HOUR,
        min_time_span_hours: float = DEFAULT_MIN_TIME_SPAN_HOURS,
        corroboration_reference_publishers: float = DEFAULT_CORROBORATION_REFERENCE_PUBLISHERS,
        novelty_lookback_hours: float = DEFAULT_NOVELTY_LOOKBACK_HOURS,
    ) -> None:
        self._max_time_gap = timedelta(hours=max_time_gap_hours)
        self._min_title_similarity = min_title_similarity
        self._velocity_reference_rate = velocity_reference_rate_per_hour
        self._min_time_span_hours = min_time_span_hours
        self._corroboration_reference_publishers = corroboration_reference_publishers
        self._novelty_lookback = timedelta(hours=novelty_lookback_hours)

    def resolve(self, features: List[DocumentFeatures]) -> Dict[str, ResolvedEvent]:
        """Returns {document_id: ResolvedEvent}; every input document
        appears as a key, even if its event is a singleton (document
        that matched no one else still gets a valid resolved_event_id,
        just with document_count=1 and corroboration=0.0)."""
        sorted_features = sorted(features, key=lambda f: (f.document.timestamp, f.document.document_id))

        clusters: List[List[DocumentFeatures]] = []
        for feat in sorted_features:
            target_cluster: Optional[List[DocumentFeatures]] = None
            for cluster in clusters:
                if any(
                    _should_merge(feat, member, self._max_time_gap, self._min_title_similarity)
                    for member in cluster
                ):
                    target_cluster = cluster
                    break
            if target_cluster is not None:
                target_cluster.append(feat)
            else:
                clusters.append([feat])

        resolved_map: Dict[str, ResolvedEvent] = {}
        # Chronological because `clusters` is built from timestamp-sorted
        # input: a new cluster's first document's timestamp is always >=
        # every previously-processed document's timestamp. A signature is
        # appended only AFTER its own event is fully resolved, so later
        # events' novelty never "sees" a chronologically-future event.
        history_signatures: List[_EventHistorySignature] = []

        for cluster in clusters:
            representative = cluster[0]  # earliest document in the cluster, by construction
            representative_event_type = representative.event_classification.event_type
            representative_title = representative.document.title
            # Safe: merging requires exact event_type equality, so every
            # member of a cluster shares the same event_type already.
            ticker_sets = [frozenset(e.ticker for e in feat.entities) for feat in cluster]
            all_tickers: FrozenSet[str] = frozenset().union(*ticker_sets) if ticker_sets else frozenset()

            timestamps = [feat.document.timestamp for feat in cluster]
            first_seen = min(timestamps)
            latest_seen = max(timestamps)

            distinct_sources = _distinct_sources(cluster)
            distinct_source_count = len(distinct_sources)
            distinct_publisher_count = len({feat.document.publisher for feat in distinct_sources})

            novelty = _compute_novelty(
                representative_event_type,
                all_tickers,
                representative_title,
                first_seen,
                history_signatures,
                self._novelty_lookback,
            )
            velocity = _compute_velocity(
                len(cluster), first_seen, latest_seen, self._velocity_reference_rate, self._min_time_span_hours
            )
            corroboration = _compute_corroboration(distinct_publisher_count, self._corroboration_reference_publishers)

            member_ids_sorted = sorted(feat.document.document_id for feat in cluster)
            resolved_event = ResolvedEvent(
                resolved_event_id=_make_resolved_event_id(member_ids_sorted),
                event_type=representative_event_type,
                member_document_ids=member_ids_sorted,
                first_seen=first_seen,
                latest_seen=latest_seen,
                document_count=len(cluster),
                distinct_source_count=distinct_source_count,
                distinct_publisher_count=distinct_publisher_count,
                novelty=novelty,
                velocity=velocity,
                corroboration=corroboration,
            )

            for feat in cluster:
                resolved_map[feat.document.document_id] = resolved_event

            history_signatures.append(
                _EventHistorySignature(
                    event_type=representative_event_type,
                    tickers=all_tickers,
                    representative_title=representative_title,
                    first_seen=first_seen,
                )
            )

        return resolved_map
