"""Embeddings, Vector Store, and Hybrid Retrieval Engine."""

from __future__ import annotations

import abc
import math
import re
from typing import Any, Dict, List, Optional
import numpy as np

from app.retrieval.models import Chunk, Document, SearchResult


class BaseEmbeddings(abc.ABC):
    """Abstract base class for text embedding models."""

    @abc.abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding vector for a query string."""
        pass

    @abc.abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embedding vectors for a list of document strings."""
        pass


class LocalBM25Embeddings(BaseEmbeddings):
    """Fast, deterministic character/token n-gram embedding for local evaluation without external APIs."""

    def __init__(self, dimension: int = 128):
        self.dimension = dimension

    def _hash_token(self, token: str) -> int:
        hash_val = 5381
        for ch in token:
            hash_val = ((hash_val << 5) + hash_val) + ord(ch)
        return hash_val % self.dimension

    def _vectorize(self, text: str) -> List[float]:
        tokens = re.findall(r"\w+", text.lower())
        if not tokens:
            return [0.0] * self.dimension

        vec = np.zeros(self.dimension, dtype=np.float32)
        for i, token in enumerate(tokens):
            idx = self._hash_token(token)
            vec[idx] += 1.0
            if i + 1 < len(tokens):
                bigram_idx = self._hash_token(f"{token}_{tokens[i+1]}")
                vec[bigram_idx] += 0.5

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_query(self, text: str) -> List[float]:
        return self._vectorize(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._vectorize(t) for t in texts]


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Calculate cosine similarity between two float vectors."""
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class VectorStore:
    """In-memory Vector Store with support for dense semantic search, BM25 keyword matching, and hybrid RRF search."""

    def __init__(self, embedder: Optional[BaseEmbeddings] = None):
        self.embedder = embedder or LocalBM25Embeddings()
        self.documents: Dict[str, Document] = {}
        self.chunks: Dict[str, Chunk] = {}

    def add_documents(self, documents: List[Document], chunks: List[Chunk]) -> None:
        """Index documents and their corresponding chunks."""
        for doc in documents:
            self.documents[doc.id] = doc

        texts_to_embed = []
        chunks_to_embed = []

        for chunk in chunks:
            self.chunks[chunk.id] = chunk
            if chunk.embedding is None:
                texts_to_embed.append(chunk.text)
                chunks_to_embed.append(chunk)

        if texts_to_embed:
            embeddings = self.embedder.embed_documents(texts_to_embed)
            for chunk, emb in zip(chunks_to_embed, embeddings):
                chunk.embedding = emb

    def get_document(self, document_id: str) -> Optional[Document]:
        """Retrieve a document by ID."""
        return self.documents.get(document_id)

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """Retrieve a chunk by ID."""
        return self.chunks.get(chunk_id)

    def search_dense(
        self,
        query: str,
        limit: int = 5,
        document_id: Optional[str] = None,
    ) -> List[SearchResult]:
        """Perform dense semantic similarity search."""
        if not self.chunks:
            return []

        query_emb = self.embedder.embed_query(query)
        scored_results: List[SearchResult] = []

        for chunk in self.chunks.values():
            if document_id and chunk.document_id != document_id:
                continue

            if chunk.embedding is None:
                continue

            score = cosine_similarity(query_emb, chunk.embedding)
            if score > 0:
                doc = self.documents.get(chunk.document_id)
                title = doc.title if doc else chunk.metadata.get("title", "")
                scored_results.append(
                    SearchResult(
                        chunk_id=chunk.id,
                        document_id=chunk.document_id,
                        title=title,
                        text=chunk.text,
                        score=score,
                        chunk_index=chunk.chunk_index,
                        metadata=chunk.metadata,
                    )
                )

        scored_results.sort(key=lambda r: r.score, reverse=True)
        return scored_results[:limit]

    def search_keyword(
        self,
        query: str,
        limit: int = 5,
        document_id: Optional[str] = None,
    ) -> List[SearchResult]:
        """Perform BM25/keyword-based search over chunk texts."""
        query_terms = set(re.findall(r"\w+", query.lower()))
        if not query_terms or not self.chunks:
            return []

        results: List[SearchResult] = []
        for chunk in self.chunks.values():
            if document_id and chunk.document_id != document_id:
                continue

            chunk_terms = re.findall(r"\w+", chunk.text.lower())
            if not chunk_terms:
                continue

            term_counts: Dict[str, int] = {}
            for t in chunk_terms:
                term_counts[t] = term_counts.get(t, 0) + 1

            matched_score = 0.0
            for term in query_terms:
                if term in term_counts:
                    tf = term_counts[term] / len(chunk_terms)
                    matched_score += 1.0 + math.log(1.0 + tf)

            if matched_score > 0:
                doc = self.documents.get(chunk.document_id)
                title = doc.title if doc else chunk.metadata.get("title", "")
                results.append(
                    SearchResult(
                        chunk_id=chunk.id,
                        document_id=chunk.document_id,
                        title=title,
                        text=chunk.text,
                        score=matched_score,
                        chunk_index=chunk.chunk_index,
                        metadata=chunk.metadata,
                    )
                )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    def search_hybrid(
        self,
        query: str,
        limit: int = 5,
        document_id: Optional[str] = None,
        rrf_k: int = 60,
    ) -> List[SearchResult]:
        """Combine dense and keyword search using Reciprocal Rank Fusion (RRF)."""
        dense_results = self.search_dense(query, limit=limit * 2, document_id=document_id)
        keyword_results = self.search_keyword(query, limit=limit * 2, document_id=document_id)

        rrf_scores: Dict[str, float] = {}
        lookup: Dict[str, SearchResult] = {}

        for rank, res in enumerate(dense_results):
            rrf_scores[res.chunk_id] = rrf_scores.get(res.chunk_id, 0.0) + (1.0 / (rrf_k + rank + 1))
            lookup[res.chunk_id] = res

        for rank, res in enumerate(keyword_results):
            rrf_scores[res.chunk_id] = rrf_scores.get(res.chunk_id, 0.0) + (1.0 / (rrf_k + rank + 1))
            lookup[res.chunk_id] = res

        sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

        final_results = []
        for cid in sorted_chunk_ids[:limit]:
            original = lookup[cid]
            final_results.append(
                SearchResult(
                    chunk_id=original.chunk_id,
                    document_id=original.document_id,
                    title=original.title,
                    text=original.text,
                    score=rrf_scores[cid],
                    chunk_index=original.chunk_index,
                    metadata=original.metadata,
                )
            )

        return final_results
