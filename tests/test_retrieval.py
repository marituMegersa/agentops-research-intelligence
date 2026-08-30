"""Unit tests for Document Ingestion, VectorStore, and EvidenceManager."""

import pytest
from app.retrieval.models import Document, VerificationStatus
from app.retrieval.ingestion import TextSplitter, DocumentIngestor
from app.retrieval.vector_store import VectorStore, LocalBM25Embeddings
from app.retrieval.evidence import EvidenceManager


def test_text_splitter_basic():
    splitter = TextSplitter(chunk_size=50, chunk_overlap=10)
    text = "Machine learning is a field of AI. It allows computers to learn from data. Deep learning is a subset."
    chunks = splitter.split_text(text)
    assert len(chunks) >= 2
    assert all(len(c) <= 60 for c in chunks)


def test_text_splitter_empty():
    splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
    assert splitter.split_text("") == []


def test_document_ingestor():
    ingestor = DocumentIngestor(text_splitter=TextSplitter(chunk_size=100, chunk_overlap=20))
    doc = Document(
        title="Sample Architecture",
        content="Retrieval-Augmented Generation enhances generative AI. It retrieves relevant documents from a database.",
        metadata={"category": "ai"},
    )
    chunks = ingestor.ingest_document(doc)
    assert len(chunks) >= 1
    assert chunks[0].document_id == doc.id
    assert chunks[0].metadata["category"] == "ai"
    assert chunks[0].token_count > 0


def test_vector_store_dense_and_hybrid():
    store = VectorStore(embedder=LocalBM25Embeddings(dimension=64))
    ingestor = DocumentIngestor(text_splitter=TextSplitter(chunk_size=200, chunk_overlap=20))

    doc1 = Document(
        id="d1",
        title="Agent Workflows",
        content="Autonomous agents plan multi-step trajectories and call external tools.",
    )
    doc2 = Document(
        id="d2",
        title="Database Optimization",
        content="Relational databases use B-tree indexing to speed up queries.",
    )

    chunks1 = ingestor.ingest_document(doc1)
    chunks2 = ingestor.ingest_document(doc2)

    store.add_documents([doc1, doc2], chunks1 + chunks2)

    # Dense search
    dense_results = store.search_dense(query="autonomous agent tool calling", limit=1)
    assert len(dense_results) == 1
    assert dense_results[0].document_id == "d1"

    # Hybrid search
    hybrid_results = store.search_hybrid(query="relational database indexing", limit=1)
    assert len(hybrid_results) == 1
    assert hybrid_results[0].document_id == "d2"


def test_evidence_manager():
    store = VectorStore()
    ingestor = DocumentIngestor()
    doc = Document(
        id="d-100",
        title="LLM Reliability",
        content="Hallucinations occur when models generate unsupported facts. Grounding with citations reduces error rates.",
    )
    chunks = ingestor.ingest_document(doc)
    store.add_documents([doc], chunks)

    manager = EvidenceManager(vector_store=store)
    claims = manager.extract_claims_from_text(doc.content, document_id=doc.id, chunk_id=chunks[0].id)
    assert len(claims) >= 2
    assert claims[0].status == VerificationStatus.UNVERIFIED

    # Attach evidence
    claim_to_verify = claims[0]
    evidence = manager.attach_evidence(
        claim_id=claim_to_verify.id,
        source_id=doc.id,
        chunk_id=chunks[0].id,
        quote="Hallucinations occur when models generate unsupported facts.",
        status=VerificationStatus.VERIFIED,
    )
    assert evidence is not None
    assert evidence.status == VerificationStatus.VERIFIED
    assert claim_to_verify.status == VerificationStatus.VERIFIED

    retrieved_evidences = manager.get_evidence_for_claim(claim_to_verify.id)
    assert len(retrieved_evidences) == 1
    assert retrieved_evidences[0].quote == evidence.quote
