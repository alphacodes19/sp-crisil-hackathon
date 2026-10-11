"""
Synthetic high-impact demo fixture loader.

Loads data/replay/synthetic/high_impact_demo.jsonl: a SMALL, ENTIRELY
FICTIONAL set of documents, clearly labeled as synthetic in every
field that could otherwise be mistaken for real data:
  - the "company" ("Fictional Demo Holdings Inc. (SYNTHETIC DEMO
    COMPANY - NOT REAL)", ticker "ZZDEMO") exists only in
    data/entities/alias_table.csv for this purpose and matches no real
    issuer;
  - every "publisher" domain uses the ".test" top-level domain, which
    is reserved by IANA/RFC 2606 specifically for testing and is
    guaranteed to never resolve to a real website;
  - every document_id/url is prefixed "synthetic-demo-".

Purpose: exercise the velocity + corroboration + impact pipeline
end-to-end with a deterministic fixture designed to produce multiple
independent "publishers" reporting the "same event" in a tight time
window - something the real GDELT/SEC replay data (by design, since
it's real and small) doesn't reliably do. This fixture is NEVER mixed
into `run_replay_pipeline()`'s default document set - it is loaded only
by the dedicated test that exercises it
(tests/test_synthetic_high_impact_fixture.py). The ordinary GDELT/SEC
replay data and pipeline defaults are unaffected by its existence.

This is a `load_replay()`-only module (no `fetch_live()`): there is no
live source for fictional data, so unlike gdelt.py/sec_edgar.py there
is nothing to fetch.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from src.utils.io import read_jsonl
from src.utils.schema import UnifiedDocument

DEFAULT_REPLAY_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "replay" / "synthetic" / "high_impact_demo.jsonl"
)


def load_replay(path: Optional[Path] = None) -> List[UnifiedDocument]:
    """Load the synthetic high-impact demo fixture as UnifiedDocuments."""
    return list(read_jsonl(path or DEFAULT_REPLAY_PATH, UnifiedDocument))
