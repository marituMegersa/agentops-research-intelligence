"""Evidence management, claim extraction, and verification tracking."""

from __future__ import annotations

import re
from typing import Dict, List, Optional
from app.retrieval.models import Claim, Evidence, VerificationStatus
from app.retrieval.vector_store import VectorStore, cosine_similarity


class EvidenceManager:
    """Manages claims, evidence references, and verification status."""

    def __init__(self, vector_store: Optional[VectorStore] = None):
        self.vector_store = vector_store or VectorStore()
        self.claims: Dict[str, Claim] = {}
        self.evidence_items: Dict[str, Evidence] = {}

    def extract_claims_from_text(
        self,
        text: str,
        document_id: Optional[str] = None,
        chunk_id: Optional[str] = None,
    ) -> List[Claim]:
        """Extract candidate claim statements from text."""
        sentences = [
            s.strip()
            for s in re.split(r"[.!?]\s+", text.strip())
            if len(s.strip()) > 10
        ]

        extracted = []
        for sentence in sentences:
            claim = Claim(
                text=sentence,
                document_id=document_id,
                chunk_id=chunk_id,
                confidence=1.0,
                status=VerificationStatus.UNVERIFIED,
            )
            self.claims[claim.id] = claim
            extracted.append(claim)

        return extracted

    def add_claim(self, claim: Claim) -> Claim:
        """Register a claim in the evidence store."""
        self.claims[claim.id] = claim
        return claim

    def get_claim(self, claim_id: str) -> Optional[Claim]:
        """Retrieve a claim by ID."""
        return self.claims.get(claim_id)

    def attach_evidence(
        self,
        claim_id: str,
        source_id: str,
        chunk_id: str,
        quote: str,
        status: VerificationStatus = VerificationStatus.VERIFIED,
    ) -> Optional[Evidence]:
        """Attach a supporting or contradicting piece of evidence to a claim."""
        claim = self.claims.get(claim_id)
        if not claim:
            return None

        # Compute similarity score if vector store has embedding
        similarity = 0.0
        chunk = self.vector_store.get_chunk(chunk_id)
        if chunk and chunk.embedding:
            claim_emb = self.vector_store.embedder.embed_query(claim.text)
            similarity = cosine_similarity(claim_emb, chunk.embedding)

        evidence = Evidence(
            claim_id=claim_id,
            source_id=source_id,
            chunk_id=chunk_id,
            quote=quote,
            similarity_score=similarity,
            status=status,
        )
        self.evidence_items[evidence.id] = evidence
        claim.supporting_evidence_ids.append(evidence.id)
        claim.status = status
        return evidence

    def get_evidence_for_claim(self, claim_id: str) -> List[Evidence]:
        """List all evidence linked to a given claim."""
        return [
            evi
            for evi in self.evidence_items.values()
            if evi.claim_id == claim_id
        ]
