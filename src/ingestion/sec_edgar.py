"""
SEC EDGAR 8-K ingestion.

Mirrors the GDELT ingestion pattern (src/ingestion/gdelt.py): two entry
points, both returning UnifiedDocument, so downstream stages never need
to know which path produced a document.

- fetch_live(): calls the real SEC EDGAR submissions API
  (data.sec.gov/submissions/CIK##########.json) to list a company's
  recent 8-K filings, then follows each filing's accession/primary
  document info to the actual filing document and extracts readable
  text, returning a fully populated UnifiedDocument per filing.
  Requires open outbound internet access and a descriptive User-Agent
  (SEC's fair-access policy) - see SEC_USER_AGENT env var below. NOT
  exercised in the dev sandbox used to build this project (egress there
  is restricted to a short allow-list that does not include
  data.sec.gov / sec.gov); intended to run on a normal machine with
  internet. UNVERIFIED in-sandbox - flagged for you to test end-to-end
  before relying on it.
- load_replay(): reads a local JSONL file of real 8-K filings with full
  text, pre-captured for deterministic offline/demo use (Section 5,
  "Replay Mode", in the master context). This is the path exercised by
  tests.

Provenance: every UnifiedDocument produced here keeps the company name
in `publisher`, the source agency in `source_name` ("SEC_EDGAR"), the
original filing URL in `url`, and the filing date in `timestamp` - no
new schema fields were added. This is enough to recover the full chain:

    Risk Signal -> SEC Filing -> Company -> original filing URL
"""
from __future__ import annotations

import os
import re
import time
import warnings
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import List, Optional

import requests

from src.utils.io import read_jsonl
from src.utils.schema import UnifiedDocument

SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"

# SEC EDGAR's fair-access policy requires a descriptive User-Agent that
# identifies the requester (name + contact). Configure via the
# SEC_USER_AGENT environment variable rather than hardcoding one - e.g.:
#   export SEC_USER_AGENT="YourOrgName your-email@example.com"
# If unset, fetch_live() falls back to a placeholder and warns loudly;
# the placeholder is NOT suitable for real use against sec.gov.
_FALLBACK_USER_AGENT = "SP-CRISIL-Hackathon-Prototype UNSET-CONTACT-SET-SEC_USER_AGENT-ENV-VAR"

# Conservative pacing between successive requests to sec.gov /
# data.sec.gov, well under SEC's stated fair-access limits, so a batch
# `fetch_live()` call doesn't hammer the API.
DEFAULT_REQUEST_DELAY_SECONDS = 0.3

# Defensive cap on extracted filing text so one oversized exhibit-laden
# filing can't blow up downstream memory/token budgets.
MAX_FILING_TEXT_CHARS = 20_000

DEFAULT_REPLAY_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "replay" / "sec" / "sample.jsonl"
)


def _get_user_agent() -> str:
    """Resolve the SEC User-Agent from the SEC_USER_AGENT env var."""
    ua = os.environ.get("SEC_USER_AGENT")
    if not ua:
        warnings.warn(
            "SEC_USER_AGENT environment variable is not set. Falling back to "
            "a placeholder User-Agent. SEC's fair-access policy requires a "
            "descriptive User-Agent identifying the requester (org/name + "
            "contact email) - set SEC_USER_AGENT before calling fetch_live() "
            "for real.",
            stacklevel=2,
        )
        return _FALLBACK_USER_AGENT
    return ua


class _HTMLTextExtractor(HTMLParser):
    """
    Minimal HTML -> plain text extractor, stdlib-only (no new
    dependency such as BeautifulSoup/lxml, per "keep it minimal").
    Skips <script>/<style> content and collapses whitespace.
    """

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._chunks: List[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        if tag in ("script", "style"):
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style") and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._chunks.append(data)

    def get_text(self) -> str:
        raw = " ".join(self._chunks)
        return re.sub(r"\s+", " ", raw).strip()


def extract_text_from_html(html: str) -> str:
    """Pure function: HTML string -> cleaned plain text. No network I/O."""
    parser = _HTMLTextExtractor()
    parser.feed(html)
    return parser.get_text()


def _make_document_id(cik: str, accession_number: str) -> str:
    """Deterministic ID from CIK + accession number (both already unique
    within EDGAR, so no hashing is needed, unlike the GDELT URL-hash case)."""
    clean_accession = accession_number.replace("-", "")
    return f"sec-8k-{int(cik)}-{clean_accession}"


def _build_filing_url(cik: str, accession_number: str, primary_document: str) -> str:
    clean_accession = accession_number.replace("-", "")
    return (
        f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
        f"{clean_accession}/{primary_document}"
    )


def _fetch_filing_text(url: str, user_agent: str, timeout: int = 15) -> str:
    """
    Download one filing document (typically .htm) and return extracted,
    whitespace-collapsed plain text. Raises if the document can't be
    fetched or yields no extractable text, so a bad filing surfaces
    clearly instead of silently producing an empty UnifiedDocument.
    """
    resp = requests.get(
        url,
        headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
        timeout=timeout,
    )
    resp.raise_for_status()
    text = extract_text_from_html(resp.text)
    if not text:
        raise ValueError(f"No extractable text found in filing document: {url}")
    return text[:MAX_FILING_TEXT_CHARS]


def fetch_live(
    cik: str,
    max_filings: int = 5,
    request_delay_seconds: float = DEFAULT_REQUEST_DELAY_SECONDS,
) -> List[UnifiedDocument]:
    """
    List a company's most recent 8-K filings via the SEC EDGAR
    submissions API, then follow each filing's accession/primary
    document to the actual filing document and extract its text.
    `cik` may be given with or without leading zeros (e.g. "1318605" or
    "0001318605").

    Conservative by design: one submissions-API call, then one
    document-fetch per filing with a small delay between requests
    (`request_delay_seconds`), well under SEC's stated fair-access
    limits. A filing whose document can't be fetched/parsed is skipped
    with a warning rather than aborting the whole batch.
    """
    user_agent = _get_user_agent()
    url = SEC_SUBMISSIONS_URL.format(cik=int(cik))
    resp = requests.get(url, headers={"User-Agent": user_agent}, timeout=15)
    resp.raise_for_status()
    payload = resp.json()

    company_name = payload.get("name", "unknown")
    recent = payload["filings"]["recent"]

    docs: List[UnifiedDocument] = []
    for i, form in enumerate(recent["form"]):
        if form != "8-K":
            continue
        accession_number = recent["accessionNumber"][i]
        filing_date = recent["filingDate"][i]
        primary_document = recent["primaryDocument"][i]
        filing_url = _build_filing_url(cik, accession_number, primary_document)
        title = f"{company_name} - Form 8-K - filed {filing_date}"

        try:
            text = _fetch_filing_text(filing_url, user_agent)
        except (requests.RequestException, ValueError) as exc:
            warnings.warn(f"Skipping filing {filing_url}: {exc}", stacklevel=2)
            continue

        docs.append(
            UnifiedDocument(
                document_id=_make_document_id(cik, accession_number),
                source_type="filing",
                source_name="SEC_EDGAR",
                publisher=company_name,
                timestamp=datetime.fromisoformat(filing_date),
                title=title,
                text=text,
                url=filing_url,
            )
        )
        if len(docs) >= max_filings:
            break
        time.sleep(request_delay_seconds)
    return docs


def load_replay(path: Optional[Path] = None) -> List[UnifiedDocument]:
    """Load pre-captured, full-text 8-K filings as UnifiedDocuments."""
    return list(read_jsonl(path or DEFAULT_REPLAY_PATH, UnifiedDocument))
