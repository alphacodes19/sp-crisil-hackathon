"""
GDELT ingestion.

Two entry points, both returning the same UnifiedDocument type so no
downstream stage needs to know which path produced a document:

- fetch_live(): calls the real GDELT 2.0 DOC API. Requires open outbound
  internet access. Not exercised in the dev sandbox used to build this
  project (egress there is restricted to a short allow-list that does
  not include api.gdeltproject.org); intended to run on a normal machine
  with internet (e.g. during the actual demo/live mode).
- load_replay(): reads a local JSONL file of pre-captured, schema-valid
  GDELT-style articles for deterministic offline/demo use. This is the
  path exercised by tests and by the mandatory Replay Mode
  (Section 5 / Section 29 Day 1 of the master context).

Replay data in data/replay/gdelt/sample.jsonl was seeded from real,
public news articles (fetched via research tooling outside this
sandbox's network restriction, not via the live GDELT API itself), so
the shape and content are real-world-representative while remaining
reproducible without network access.
"""
from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import requests

from src.utils.io import read_jsonl
from src.utils.schema import UnifiedDocument

GDELT_DOC_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
DEFAULT_REPLAY_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "replay" / "gdelt" / "sample.jsonl"
)


def _make_document_id(url: str) -> str:
    """Deterministic ID derived from the article URL (idempotent re-ingestion)."""
    return "gdelt-" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]


def _parse_gdelt_seendate(raw: str) -> datetime:
    """GDELT 'seendate' format is e.g. '20261008T103100Z'."""
    return datetime.strptime(raw, "%Y%m%dT%H%M%SZ")


def fetch_live(
    query: str,
    max_records: int = 25,
    timespan: str = "1d",
) -> List[UnifiedDocument]:
    """
    Query the GDELT 2.0 DOC API (mode=artlist) and normalize results.

    NOTE: GDELT's artlist mode returns headline/metadata, not full
    article body text; `text` is populated from the title as a
    placeholder until a full-text fetch step is added (future
    milestone, not required for Day 1 Milestone 1).
    """
    params = {
        "query": query,
        "mode": "artlist",
        "maxrecords": str(max_records),
        "timespan": timespan,
        "format": "json",
    }
    resp = requests.get(GDELT_DOC_API_URL, params=params, timeout=15)
    resp.raise_for_status()
    payload = resp.json()

    docs: List[UnifiedDocument] = []
    for article in payload.get("articles", []):
        url = article.get("url", "")
        title = article.get("title", "")
        docs.append(
            UnifiedDocument(
                document_id=_make_document_id(url),
                source_type="news",
                source_name="GDELT",
                publisher=article.get("domain", "unknown"),
                timestamp=_parse_gdelt_seendate(article["seendate"]),
                title=title,
                text=title,
                url=url,
            )
        )
    return docs


def load_replay(path: Optional[Path] = None) -> List[UnifiedDocument]:
    """Load a pre-captured GDELT-style replay file as UnifiedDocuments."""
    return list(read_jsonl(path or DEFAULT_REPLAY_PATH, UnifiedDocument))
