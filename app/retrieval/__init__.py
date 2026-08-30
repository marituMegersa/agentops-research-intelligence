"""Retrieval and Evidence Management Module."""

from app.retrieval.models import Chunk, Claim, Document, Evidence, SearchResult, VerificationStatus
from app.retrieval.ingestion import DocumentIngestor, TextSplitter
from app.retrieval.vector_store import BaseEmbeddings, LocalBM25Embeddings, VectorStore
from app.retrieval.evidence import EvidenceManager

__all__ = [
    "Document",
    "Chunk",
    "SearchResult",
    "Claim",
    "Evidence",
    "VerificationStatus",
    "TextSplitter",
    "DocumentIngestor",
    "BaseEmbeddings",
    "LocalBM25Embeddings",
    "VectorStore",
    "EvidenceManager",
]
