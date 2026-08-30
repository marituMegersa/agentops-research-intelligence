"""Unit tests for ShortTermMemory and LongTermMemory."""

import pytest
from app.agents.memory import LongTermMemory, ShortTermMemory
from app.retrieval.vector_store import VectorStore


def test_short_term_memory():
    mem = ShortTermMemory()
    mem.add_note("Decomposing question into 2 sub-queries")
    mem.set_var("target_domain", "enterprise_rag")

    assert len(mem.scratchpad) == 1
    assert mem.get_var("target_domain") == "enterprise_rag"
    assert mem.get_var("missing", default=10) == 10

    mem.clear()
    assert len(mem.scratchpad) == 0
    assert mem.get_var("target_domain") is None


def test_long_term_memory_recall():
    store = VectorStore()
    ltm = LongTermMemory(vector_store=store)

    # Store memory
    record = ltm.store_research(
        query="Enterprise RAG Architecture",
        summary="Use hybrid vector search with BM25 reranking for high precision retrieval.",
        evidence_quotes=["Hybrid retrieval combines dense cosine similarity with BM25."],
        metadata={"category": "architecture"},
    )
    assert record.id in ltm.records

    # Recall memory
    recalled = ltm.recall(query="RAG Architecture hybrid retrieval", limit=1)
    assert len(recalled) == 1
    assert recalled[0].id == record.id
    assert "hybrid vector search" in recalled[0].summary
