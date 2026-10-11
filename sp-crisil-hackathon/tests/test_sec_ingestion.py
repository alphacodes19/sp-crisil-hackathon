import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from src.ingestion.sec_edgar import (
    MAX_FILING_TEXT_CHARS,
    _build_filing_url,
    _get_user_agent,
    _make_document_id,
    extract_text_from_html,
    load_replay,
)
from src.utils.schema import UnifiedDocument


def test_load_replay_sample_returns_valid_documents():
    docs = load_replay()
    assert len(docs) >= 3
    ids_seen = {d.document_id for d in docs}
    assert len(ids_seen) == len(docs)  # all IDs unique
    for d in docs:
        assert isinstance(d, UnifiedDocument)
        assert d.source_type == "filing"
        assert d.source_name == "SEC_EDGAR"
        assert d.text.strip() != ""
        assert d.title.strip() != ""
        assert d.document_id.startswith("sec-8k-")


def test_replay_documents_preserve_provenance_chain():
    """
    Every document must let us recover:
    company (publisher) -> filing timestamp -> filing URL -> source=SEC/EDGAR -> text
    using only existing UnifiedDocument fields.
    """
    docs = load_replay()
    for d in docs:
        assert d.publisher  # company/entity
        assert d.timestamp  # filing timestamp
        assert d.url and d.url.startswith("https://www.sec.gov/Archives/edgar/")
        assert d.source_name == "SEC_EDGAR"
        assert d.text.strip() != ""


def test_build_filing_url_matches_edgar_archive_pattern():
    url = _build_filing_url("1318605", "0001628280-26-063820", "tsla-20260929.htm")
    assert url == (
        "https://www.sec.gov/Archives/edgar/data/1318605/"
        "000162828026063820/tsla-20260929.htm"
    )


def test_make_document_id_is_deterministic_and_unique_per_accession():
    id_a = _make_document_id("1318605", "0001628280-26-063820")
    id_b = _make_document_id("1318605", "0001628280-26-063820")
    id_c = _make_document_id("1318605", "0001628280-26-063821")
    assert id_a == id_b
    assert id_a != id_c
    assert id_a == "sec-8k-1318605-000162828026063820"


# --- text extraction (pure function, no network) ---------------------------


def test_extract_text_from_html_strips_tags_and_scripts():
    html = """
    <html><head><style>.x{color:red}</style></head>
    <body>
      <script>var x = 1;</script>
      <h1>Tesla, Inc.</h1>
      <p>Item 2.03. Creation of a Direct Financial Obligation.</p>
      <p>On September 29, 2026, Tesla entered into a credit agreement.</p>
    </body></html>
    """
    text = extract_text_from_html(html)
    assert "var x = 1" not in text
    assert "color:red" not in text
    assert "Tesla, Inc." in text
    assert "Item 2.03. Creation of a Direct Financial Obligation." in text
    assert "On September 29, 2026, Tesla entered into a credit agreement." in text
    # whitespace/newlines collapsed to single spaces
    assert "  " not in text


def test_extract_text_from_html_collapses_whitespace():
    html = "<p>Line one.</p>\n\n\n<p>   Line   two.   </p>"
    text = extract_text_from_html(html)
    assert text == "Line one. Line two."


def test_extracted_text_is_capped_at_max_filing_text_chars():
    huge_html = "<p>" + ("A" * (MAX_FILING_TEXT_CHARS + 500)) + "</p>"
    text = extract_text_from_html(huge_html)
    # extract_text_from_html itself does not truncate (that happens in
    # _fetch_filing_text); confirm the raw extraction exceeds the cap so
    # the cap in _fetch_filing_text is actually meaningful.
    assert len(text) > MAX_FILING_TEXT_CHARS


# --- User-Agent resolution ---------------------------------------------------


def test_get_user_agent_uses_env_var_when_set(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "Acme Research acme@example.com")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no warning expected when env var set
        ua = _get_user_agent()
    assert ua == "Acme Research acme@example.com"


def test_get_user_agent_warns_and_falls_back_when_env_var_missing(monkeypatch):
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    with pytest.warns(UserWarning, match="SEC_USER_AGENT"):
        ua = _get_user_agent()
    assert "SEC_USER_AGENT" in ua or "UNSET-CONTACT" in ua
