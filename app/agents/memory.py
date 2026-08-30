"""Short-term and episodic long-term memory systems for research agents."""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional
import uuid
from pydantic import BaseModel, Field

from app.retrieval.models import Chunk, Document
from app.retrieval.vector_store import VectorStore


class MemoryRecord(BaseModel):
    """An individual episodic memory item stored in long-term memory."""

    id: str = Field(default_factory=lambda: f"mem-{uuid.uuid4().hex[:8]}")
    query: str
    summary: str
    evidence_quotes: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )


class ShortTermMemory(BaseModel):
    """Ephemeral scratchpad and context tracking within a single research session."""

    session_id: str = Field(default_factory=lambda: f"sess-{uuid.uuid4().hex[:8]}")
    scratchpad: List[str] = Field(default_factory=list)
    context_variables: Dict[str, Any] = Field(default_factory=dict)

    def add_note(self, note: str) -> None:
        """Add a scratchpad reasoning note."""
        self.scratchpad.append(note)

    def set_var(self, key: str, value: Any) -> None:
        """Store a temporary variable in session context."""
        self.context_variables[key] = value

    def get_var(self, key: str, default: Any = None) -> Any:
        """Retrieve a session variable."""
        return self.context_variables.get(key, default)

    def clear(self) -> None:
        """Reset the scratchpad."""
        self.scratchpad.clear()
        self.context_variables.clear()


class LongTermMemory:
    """Episodic memory store backed by vector similarity for cross-session recall."""

    def __init__(self, vector_store: Optional[VectorStore] = None):
        self.vector_store = vector_store or VectorStore()
        self.records: Dict[str, MemoryRecord] = {}

    def store_research(
        self,
        query: str,
        summary: str,
        evidence_quotes: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryRecord:
        """Save a completed research finding into long-term episodic memory."""
        record = MemoryRecord(
            query=query,
            summary=summary,
            evidence_quotes=evidence_quotes or [],
            metadata=metadata or {},
        )
        self.records[record.id] = record

        # Index into vector store for semantic recall
        doc = Document(
            id=record.id,
            title=f"Memory: {query}",
            content=f"{query}\n{summary}\n{' '.join(record.evidence_quotes)}",
            metadata={"type": "episodic_memory", "memory_id": record.id},
        )
        chunk = Chunk(
            document_id=doc.id,
            text=doc.content,
            metadata=doc.metadata,
        )
        self.vector_store.add_documents([doc], [chunk])
        return record

    def recall(self, query: str, limit: int = 3) -> List[MemoryRecord]:
        """Retrieve relevant past research memories matching the current query."""
        search_results = self.vector_store.search_hybrid(query=query, limit=limit)
        memories: List[MemoryRecord] = []

        for res in search_results:
            if res.document_id in self.records:
                memories.append(self.records[res.document_id])

        return memories
