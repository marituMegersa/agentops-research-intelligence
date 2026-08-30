"""LangGraph workflow orchestrating Planning, Research, Verification, and Synthesis agents."""

from __future__ import annotations

from typing import Optional
from langgraph.graph import END, START, StateGraph

from app.agents.planner import PlanningAgent
from app.agents.researcher import ResearchAgent
from app.agents.state import ResearchState
from app.agents.synthesizer import SynthesisAgent
from app.agents.verifier import VerificationAgent
from app.retrieval.evidence import EvidenceManager
from app.retrieval.vector_store import VectorStore


def should_continue(state: ResearchState) -> str:
    """Conditional edge router checking if evidence is sufficient or retry limit reached."""
    if state.is_sufficient or state.iteration_count >= state.max_iterations:
        return "synthesizer"

    state.iteration_count += 1
    return "researcher"


def create_research_graph(
    vector_store: Optional[VectorStore] = None,
    evidence_manager: Optional[EvidenceManager] = None,
):
    """Build and compile the multi-agent research LangGraph."""
    store = vector_store or VectorStore()
    ev_manager = evidence_manager or EvidenceManager(vector_store=store)

    planner = PlanningAgent()
    researcher = ResearchAgent(vector_store=store)
    verifier = VerificationAgent(evidence_manager=ev_manager)
    synthesizer = SynthesisAgent()

    # Define Node Handlers
    def plan_node(state: ResearchState) -> ResearchState:
        return planner.plan(state)

    def research_node(state: ResearchState) -> ResearchState:
        return researcher.research(state)

    def verify_node(state: ResearchState) -> ResearchState:
        return verifier.verify(state)

    def synthesize_node(state: ResearchState) -> ResearchState:
        return synthesizer.synthesize(state)

    # Construct StateGraph
    workflow = StateGraph(ResearchState)

    workflow.add_node("planner", plan_node)
    workflow.add_node("researcher", research_node)
    workflow.add_node("verifier", verify_node)
    workflow.add_node("synthesizer", synthesize_node)

    # Define Edges
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "verifier")
    workflow.add_conditional_edges(
        "verifier",
        should_continue,
        {
            "synthesizer": "synthesizer",
            "researcher": "researcher",
        },
    )
    workflow.add_edge("synthesizer", END)

    return workflow.compile()


class ResearchOrchestrator:
    """High-level facade to execute end-to-end multi-agent research workflows."""

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        evidence_manager: Optional[EvidenceManager] = None,
    ):
        self.vector_store = vector_store or VectorStore()
        self.evidence_manager = evidence_manager or EvidenceManager(vector_store=self.vector_store)
        self.graph = create_research_graph(
            vector_store=self.vector_store,
            evidence_manager=self.evidence_manager,
        )

    def run(self, query: str, max_iterations: int = 2) -> ResearchState:
        """Run the full research pipeline on a user query."""
        initial_state = ResearchState(
            user_query=query,
            max_iterations=max_iterations,
        )
        final_state_dict = self.graph.invoke(initial_state)
        return (
            final_state_dict
            if isinstance(final_state_dict, ResearchState)
            else ResearchState(**final_state_dict)
        )
