"""
Entity resolution.

Maps raw company mentions in a UnifiedDocument's title/text to a
canonical entity (name + ticker), using a controlled alias/ticker table
(data/entities/alias_table.csv). Per Section 7 of the master context:
deterministic rules first, avoid excessive model complexity, and do not
build a giant entity knowledge graph (~20-30 companies is enough for
the demo/portfolio universe).

Design:
- This is a separate enrichment layer, not a change to UnifiedDocument.
  Callers get back a list of EntityMatch records alongside the
  untouched document.
- Matching priority, highest first: exact ticker > exact canonical name
  > known alias > fuzzy fallback. A document can mention more than one
  entity; each distinct canonical entity is returned at most once, with
  its single best (highest-priority) match type.
- Ticker matching is case-sensitive and word-bounded on the raw text
  (tickers conventionally appear uppercase, e.g. "NVDA", "(TSLA)").
  Company-name/alias matching is case- and punctuation-insensitive
  (normalized: lowercased, punctuation stripped, whitespace collapsed).
- The fuzzy fallback (stdlib difflib only - no new dependency) only
  tries to catch near-miss misspellings of a canonical company name
  that the deterministic pass completely missed; it never overrides a
  deterministic match, and uses a conservative similarity threshold to
  avoid false positives.

Known limitation (documented, not solved here): very short tickers
("C", "V") carry an elevated false-positive risk against ordinary
uppercase single letters (e.g. list markers). Acceptable for prototype
scope; would need entity disambiguation (e.g. co-occurring context) to
tighten further in a later milestone.
"""
from __future__ import annotations

import csv
import difflib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel

from src.utils.schema import UnifiedDocument

MatchType = Literal["ticker", "exact_name", "alias", "fuzzy"]

_MATCH_TYPE_PRIORITY: Dict[MatchType, int] = {
    "ticker": 0,
    "exact_name": 1,
    "alias": 2,
    "fuzzy": 3,
}

DEFAULT_ALIAS_TABLE_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "entities" / "alias_table.csv"
)

DEFAULT_FUZZY_THRESHOLD = 0.88


class EntityMatch(BaseModel):
    """One resolved entity mention in a document."""

    canonical_name: str
    ticker: str
    matched_alias: str
    match_type: MatchType


@dataclass(frozen=True)
class _EntityRecord:
    canonical_name: str
    ticker: str


@dataclass(frozen=True)
class _AliasTable:
    ticker_index: Dict[str, _EntityRecord]  # ticker (as-is, uppercase) -> record
    alias_index: Dict[str, _EntityRecord]  # normalized alias text -> record
    canonical_index: Dict[str, _EntityRecord]  # canonical_name -> record


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _word_boundary_search(pattern_text: str, haystack: str) -> bool:
    if not pattern_text:
        return False
    pattern = r"\b" + re.escape(pattern_text) + r"\b"
    return re.search(pattern, haystack) is not None


def load_alias_table(path: Optional[Path] = None) -> _AliasTable:
    """Build ticker/alias/canonical-name indices from the CSV table."""
    path = path or DEFAULT_ALIAS_TABLE_PATH
    ticker_index: Dict[str, _EntityRecord] = {}
    alias_index: Dict[str, _EntityRecord] = {}
    canonical_index: Dict[str, _EntityRecord] = {}

    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            canonical_name = row["canonical_name"].strip()
            ticker = row["ticker"].strip().upper()
            alias = row["alias"].strip()
            record = _EntityRecord(canonical_name=canonical_name, ticker=ticker)

            ticker_index.setdefault(ticker, record)
            canonical_index.setdefault(canonical_name, record)
            alias_index[_normalize(alias)] = record

    return _AliasTable(
        ticker_index=ticker_index,
        alias_index=alias_index,
        canonical_index=canonical_index,
    )


class EntityResolver:
    """
    Loads the alias table once and resolves entities for many
    documents. Prefer this over the module-level convenience function
    when resolving a batch (e.g. an ingestion pipeline).
    """

    def __init__(
        self,
        alias_table_path: Optional[Path] = None,
        enable_fuzzy_fallback: bool = True,
        fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
    ) -> None:
        self._table = load_alias_table(alias_table_path)
        self._enable_fuzzy_fallback = enable_fuzzy_fallback
        self._fuzzy_threshold = fuzzy_threshold

    def resolve(self, document: UnifiedDocument) -> List[EntityMatch]:
        raw_text = f"{document.title} {document.text}"
        normalized_text = _normalize(raw_text)

        matches: Dict[str, EntityMatch] = {}

        # 1. Ticker matches: case-sensitive, word-bounded, on raw text.
        for ticker, record in self._table.ticker_index.items():
            if _word_boundary_search(ticker, raw_text):
                self._record_match(matches, record, ticker, "ticker")

        # 2. Alias / exact-name matches: case/punctuation-insensitive,
        #    on normalized text.
        for norm_alias, record in self._table.alias_index.items():
            if _word_boundary_search(norm_alias, normalized_text):
                match_type: MatchType = (
                    "exact_name"
                    if norm_alias == _normalize(record.canonical_name)
                    else "alias"
                )
                self._record_match(matches, record, norm_alias, match_type)

        # 3. Fuzzy fallback: only for canonical entities not already
        #    found deterministically.
        if self._enable_fuzzy_fallback:
            self._apply_fuzzy_fallback(matches, normalized_text)

        return list(matches.values())

    def _record_match(
        self,
        matches: Dict[str, EntityMatch],
        record: _EntityRecord,
        matched_alias: str,
        match_type: MatchType,
    ) -> None:
        existing = matches.get(record.canonical_name)
        if existing is None or (
            _MATCH_TYPE_PRIORITY[match_type] < _MATCH_TYPE_PRIORITY[existing.match_type]
        ):
            matches[record.canonical_name] = EntityMatch(
                canonical_name=record.canonical_name,
                ticker=record.ticker,
                matched_alias=matched_alias,
                match_type=match_type,
            )

    def _apply_fuzzy_fallback(
        self, matches: Dict[str, EntityMatch], normalized_text: str
    ) -> None:
        """
        Compare candidate word n-grams from the text against every known
        alias (not just the full canonical name), so a typo of a short
        alias like "Microsoft" or "Tesla" is still catchable even though
        it would score poorly against a longer canonical form like
        "Microsoft Corporation".
        """
        words = normalized_text.split()
        if not words:
            return
        candidates = set(words)
        for n in (2, 3):
            candidates.update(
                " ".join(words[i : i + n]) for i in range(len(words) - n + 1)
            )

        for norm_alias, record in self._table.alias_index.items():
            if record.canonical_name in matches:
                continue
            best = difflib.get_close_matches(
                norm_alias, candidates, n=1, cutoff=self._fuzzy_threshold
            )
            if best:
                matches[record.canonical_name] = EntityMatch(
                    canonical_name=record.canonical_name,
                    ticker=record.ticker,
                    matched_alias=best[0],
                    match_type="fuzzy",
                )


_default_resolver: Optional[EntityResolver] = None


def _get_default_resolver() -> EntityResolver:
    global _default_resolver
    if _default_resolver is None:
        _default_resolver = EntityResolver()
    return _default_resolver


def resolve_entities(
    document: UnifiedDocument, resolver: Optional[EntityResolver] = None
) -> List[EntityMatch]:
    """Convenience wrapper for one-off calls; reuses a lazily-built default resolver."""
    return (resolver or _get_default_resolver()).resolve(document)
