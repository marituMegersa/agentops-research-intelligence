"""Data models for document retrieval, chunking, and evidence extraction."""

from __future__ import annotations

import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid
from pydantic import BaseModel, Field


class VerificationStatus(str, Enum):
    UNVERIFIED = "unverified"
    VERIFIED = "verified"
    CONTRADICTED = "contradicted"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class Document(BaseModel):
    """Represents a full source document ingested into the research platform."""

    id: str = Field(default_factory=lambda: f"doc-{uuid.uuid4().hex[:8]}")
    title: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )


class Chunk(BaseModel):
    """Represents an atomic chunk of a source document with optional embeddings."""

    id: str = Field(default_factory=lambda: f"chk-{uuid.uuid4().hex[:8]}")
    document_id: str
    text: str
    chunk_index: int = 0
    token_count: int = 0
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """Search query match returned by the retrieval engine."""

    chunk_id: str
    document_id: str
    title: str
    text: str
    score: float
    chunk_index: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Claim(BaseModel):
    """Extracted claim or factual statement from research sources."""

    id: str = Field(default_factory=lambda: f"clm-{uuid.uuid4().hex[:8]}")
    text: str
    document_id: Optional[str] = None
    chunk_id: Optional[str] = None
    confidence: float = 1.0
    status: VerificationStatus = VerificationStatus.UNVERIFIED
    supporting_evidence_ids: List[str] = Field(default_factory=list)


class Evidence(BaseModel):
    """Structured evidence item linking claims to source quotes and provenance."""

    id: str = Field(default_factory=lambda: f"evi-{uuid.uuid4().hex[:8]}")
    claim_id: str
    source_id: str
    chunk_id: str
    quote: str
    similarity_score: float = 0.0
    status: VerificationStatus = VerificationStatus.VERIFIED
