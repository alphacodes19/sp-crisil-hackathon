import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from pydantic import ValidationError

from src.ingestion.gdelt import load_replay
from src.utils.schema import UnifiedDocument


def test_load_replay_sample_returns_valid_documents():
    docs = load_replay()
    assert len(docs) >= 3
    tickers_seen = {d.document_id for d in docs}
    assert len(tickers_seen) == len(docs)  # all IDs unique
    for d in docs:
        assert isinstance(d, UnifiedDocument)
        assert d.source_type == "news"
        assert d.source_name == "GDELT"
        assert d.text.strip() != ""
        assert d.title.strip() != ""
        assert d.document_id.startswith("gdelt-")


def test_schema_rejects_empty_text():
    with pytest.raises(ValidationError):
        UnifiedDocument(
            document_id="x",
            source_type="news",
            source_name="GDELT",
            publisher="test",
            timestamp="2026-10-08T10:00:00Z",
            title="title",
            text="   ",
            url=None,
        )


def test_schema_rejects_empty_document_id():
    with pytest.raises(ValidationError):
        UnifiedDocument(
            document_id="  ",
            source_type="news",
            source_name="GDELT",
            publisher="test",
            timestamp="2026-10-08T10:00:00Z",
            title="title",
            text="some text",
            url=None,
        )


def test_schema_accepts_minimal_valid_document():
    doc = UnifiedDocument(
        document_id="gdelt-x",
        source_type="news",
        source_name="GDELT",
        publisher="test",
        timestamp="2026-10-08T10:00:00Z",
        title="title",
        text="some text",
        url=None,
    )
    assert doc.url is None
