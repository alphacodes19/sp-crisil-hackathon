"""
Unified input document schema.

Canonical normalized form that every ingestion source (GDELT, SEC 8-K,
and any future source) must produce. Downstream stages (entity
resolution, sentiment, event classification, ...) only ever depend on
this type, never on a source-specific format.

Source: SP_CRISIL_Hackathon_Master_Project_Context.md, Section 6.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, field_validator

SourceType = Literal["news", "filing", "social", "other"]


class UnifiedDocument(BaseModel):
    """Normalized representation of any ingested document."""

    document_id: str
    source_type: SourceType
    source_name: str
    publisher: str
    timestamp: datetime
    title: str
    text: str
    url: Optional[str] = None

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("text must not be empty")
        return v

    @field_validator("document_id")
    @classmethod
    def document_id_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("document_id must not be empty")
        return v
