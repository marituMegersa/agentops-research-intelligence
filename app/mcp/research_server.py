"""FastMCP Research Server powered by dynamic hybrid vector retrieval and evidence tracking."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from mcp.server.mcpserver import MCPServer

from app.retrieval.evidence import EvidenceManager
from app.retrieval.ingestion import DocumentIngestor, TextSplitter
from app.retrieval.models import Document, VerificationStatus
from app.retrieval.vector_store import VectorStore

mcp = MCPServer(
    "research-server",
)

# Initialize Core Retrieval & Evidence Infrastructure
vector_store = VectorStore()
ingestor = DocumentIngestor(text_splitter=TextSplitter(chunk_size=300, chunk_overlap=50))
evidence_manager = EvidenceManager(vector_store=vector_store)

DEFAULT_DOCUMENTS = [
    {
        "id": "doc-001",
        "title": "Introduction to Retrieval-Augmented Generation",
        "content": (
            "RAG combines retrieval with language generation "
            "to provide models with external context."
        ),
    },
    {
        "id": "doc-002",
        "title": "Agentic AI Systems",
        "content": (
            "Agentic AI systems use models to plan tasks, "
            "select tools, execute actions, and adapt to results."
        ),
    },
    {
        "id": "doc-003",
        "title": "AI Evaluation",
        "content": (
            "Evaluation measures the quality, reliability, "
            "and performance of AI systems."
        ),
    },
]


def _seed_documents():
    """Seed initial research documents into the dynamic vector store."""
    for item in DEFAULT_DOCUMENTS:
        doc = Document(id=item["id"], title=item["title"], content=item["content"])
        chunks = ingestor.ingest_document(doc)
        vector_store.add_documents([doc], chunks)


_seed_documents()


@mcp.tool()
def search_sources(query: str, limit: int = 5) -> dict:
    """Search research sources using hybrid semantic vector and keyword search."""
    results = vector_store.search_hybrid(query=query, limit=limit)
    formatted = [
        {
            "id": r.document_id,
            "chunk_id": r.chunk_id,
            "title": r.title,
            "text": r.text,
            "score": round(r.score, 4),
        }
        for r in results
    ]

    return {
        "query": query,
        "results": formatted,
    }


@mcp.tool()
def get_source(source_id: str) -> dict:
    """Retrieve a research source by its identifier."""
    doc = vector_store.get_document(source_id)
    if doc:
        return {
            "id": doc.id,
            "title": doc.title,
            "content": doc.content,
            "metadata": doc.metadata,
        }

    return {"error": f"Source '{source_id}' was not found."}


@mcp.tool()
def ingest_source(title: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> dict:
    """Ingest and index a new research document into the vector store."""
    doc = Document(title=title, content=content, metadata=metadata or {})
    chunks = ingestor.ingest_document(doc)
    vector_store.add_documents([doc], chunks)

    return {
        "id": doc.id,
        "title": doc.title,
        "chunks_indexed": len(chunks),
        "status": "indexed",
    }


@mcp.tool()
def extract_claims(text: str) -> dict:
    """Extract candidate factual claims from text."""
    claims = evidence_manager.extract_claims_from_text(text)
    return {
        "claims": [c.text for c in claims],
        "claim_objects": [c.model_dump() for c in claims],
        "count": len(claims),
    }


@mcp.tool()
def record_evidence(
    claim_id: str,
    source_id: str,
    chunk_id: str,
    quote: str,
    status: str = "verified",
) -> dict:
    """Record evidence linking a claim to a source citation."""
    verif_status = (
        VerificationStatus.VERIFIED
        if status.lower() == "verified"
        else VerificationStatus.CONTRADICTED
    )
    evidence = evidence_manager.attach_evidence(
        claim_id=claim_id,
        source_id=source_id,
        chunk_id=chunk_id,
        quote=quote,
        status=verif_status,
    )
    if not evidence:
        return {"error": f"Claim '{claim_id}' not found."}

    return {
        "evidence_id": evidence.id,
        "claim_id": claim_id,
        "status": evidence.status.value,
        "similarity_score": round(evidence.similarity_score, 4),
    }


if __name__ == "__main__":
    mcp.run()