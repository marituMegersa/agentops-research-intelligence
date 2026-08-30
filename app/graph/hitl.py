"""Human-in-the-loop (HITL) research orchestration with review and approval gates."""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field

from app.agents.planner import PlanningAgent
from app.agents.researcher import ResearchAgent
from app.agents.state import AgentLog, ResearchPlanItem, ResearchState
from app.agents.synthesizer import SynthesisAgent
from app.agents.verifier import VerificationAgent
from app.retrieval.evidence import EvidenceManager
from app.retrieval.vector_store import VectorStore


class HITLReviewCheckpoint(BaseModel):
    """Encapsulates a human review stage for external approval."""

    stage: str  # "plan_review", "synthesis_review", "completed"
    is_approved: bool = False
    human_feedback: Optional[str] = None
    state: ResearchState


class HITLOrchestrator:
    """Orchestrates multi-agent research with interactive human-in-the-loop checkpoints."""

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        evidence_manager: Optional[EvidenceManager] = None,
    ):
        self.vector_store = vector_store or VectorStore()
        self.evidence_manager = evidence_manager or EvidenceManager(vector_store=self.vector_store)

        self.planner = PlanningAgent()
        self.researcher = ResearchAgent(vector_store=self.vector_store)
        self.verifier = VerificationAgent(evidence_manager=self.evidence_manager)
        self.synthesizer = SynthesisAgent()

    def start_research(self, query: str) -> HITLReviewCheckpoint:
        """Step 1: Run PlanningAgent and pause for human review of proposed sub-queries."""
        initial_state = ResearchState(user_query=query)
        planned_state = self.planner.plan(initial_state)

        return HITLReviewCheckpoint(
            stage="plan_review",
            is_approved=False,
            state=planned_state,
        )

    def submit_plan_approval(
        self,
        checkpoint: HITLReviewCheckpoint,
        approved: bool = True,
        override_sub_queries: Optional[List[str]] = None,
    ) -> HITLReviewCheckpoint:
        """Step 2: Process human plan approval/edits, execute research and verification, pause before synthesis."""
        state = checkpoint.state

        if not approved:
            state.logs.append(
                AgentLog(
                    agent_name="HumanOperator",
                    action="reject_plan",
                    details={"feedback": checkpoint.human_feedback or "Plan rejected"},
                )
            )
            checkpoint.stage = "plan_rejected"
            return checkpoint

        if override_sub_queries:
            state.plan = [
                ResearchPlanItem(sub_query=q, rationale="Human operator custom objective")
                for q in override_sub_queries
            ]

        state.logs.append(
            AgentLog(
                agent_name="HumanOperator",
                action="approve_plan",
                details={"sub_queries": [p.sub_query for p in state.plan]},
            )
        )

        # Execute Research & Verification
        state = self.researcher.research(state)
        state = self.verifier.verify(state)

        return HITLReviewCheckpoint(
            stage="synthesis_review",
            is_approved=False,
            state=state,
        )

    def finalize_synthesis(
        self,
        checkpoint: HITLReviewCheckpoint,
        approved: bool = True,
    ) -> HITLReviewCheckpoint:
        """Step 3: Execute synthesis and deliver the verified brief upon operator confirmation."""
        state = checkpoint.state
        if approved:
            state = self.synthesizer.synthesize(state)
            state.logs.append(
                AgentLog(
                    agent_name="HumanOperator",
                    action="approve_synthesis",
                    details={"status": "completed"},
                )
            )
            checkpoint.stage = "completed"
            checkpoint.is_approved = True
        else:
            checkpoint.stage = "synthesis_rejected"
            checkpoint.is_approved = False

        return checkpoint
