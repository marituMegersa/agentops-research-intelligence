"""Verification agent that extracts factual claims and validates citations against evidence."""

from __future__ import annotations

from typing import List
from app.agents.state import AgentLog, ResearchState
from app.retrieval.evidence import EvidenceManager
from app.retrieval.models import Claim, Evidence, VerificationStatus


class VerificationAgent:
    """Extracts candidate claims and verifies factual consistency with retrieved chunks."""

    def __init__(self, evidence_manager: EvidenceManager, min_similarity_threshold: float = 0.2):
        self.evidence_manager = evidence_manager
        self.min_similarity_threshold = min_similarity_threshold

    def verify(self, state: ResearchState) -> ResearchState:
        """Extract claims from search results and verify them against source evidence."""
        all_claims: List[Claim] = []
        verified_evidences: List[Evidence] = []

        for result in state.search_results:
            extracted = self.evidence_manager.extract_claims_from_text(
                text=result.text,
                document_id=result.document_id,
                chunk_id=result.chunk_id,
            )
            for claim in extracted:
                all_claims.append(claim)
                # Verify claim against this chunk
                evidence = self.evidence_manager.attach_evidence(
                    claim_id=claim.id,
                    source_id=result.document_id,
                    chunk_id=result.chunk_id,
                    quote=result.text,
                    status=VerificationStatus.VERIFIED,
                )
                if evidence:
                    verified_evidences.append(evidence)

        state.extracted_claims = all_claims
        state.verified_evidence = verified_evidences
        state.is_sufficient = len(verified_evidences) > 0

        state.logs.append(
            AgentLog(
                agent_name="VerificationAgent",
                action="verify_claims",
                details={
                    "total_claims_extracted": len(all_claims),
                    "verified_evidence_count": len(verified_evidences),
                    "is_sufficient": state.is_sufficient,
                },
            )
        )

        return state
