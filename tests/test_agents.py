"""Unit tests for individual research agents."""

import pytest
from app.agents.planner import PlanningAgent
from app.agents.researcher import ResearchAgent
from app.agents.verifier import VerificationAgent
from app.agents.synthesizer import SynthesisAgent
from app.agents.state import ResearchState
from app.retrieval.models import Document
from app.retrieval.ingestion import DocumentIngestor
from app.retrieval.vector_store import VectorStore
from app.retrieval.evidence import EvidenceManager


def test_planning_agent():
    agent = PlanningAgent(max_subqueries=3)
    state = ResearchState(user_query="How does Retrieval-Augmented Generation compare to Fine-Tuning?")
    updated = agent.plan(state)

    assert len(updated.plan) >= 1
    assert any("retrieval-augmented generation" in p.sub_query.lower() for p in updated.plan)
    assert len(updated.logs) == 1
    assert updated.logs[0].agent_name == "PlanningAgent"


def test_research_and_verification_agents():
    store = VectorStore()
    ingestor = DocumentIngestor()
    doc = Document(
        id="doc-rag",
        title="RAG vs Fine Tuning",
        content="Retrieval-Augmented Generation combines real-time search with generative models to reduce hallucination.",
    )
    store.add_documents([doc], ingestor.ingest_document(doc))

    ev_manager = EvidenceManager(vector_store=store)

    research_agent = ResearchAgent(vector_store=store)
    verify_agent = VerificationAgent(evidence_manager=ev_manager)

    # Initial state with plan
    state = ResearchState(user_query="Retrieval-Augmented Generation")
    state.plan = PlanningAgent().plan(state).plan

    # Run Research
    state = research_agent.research(state)
    assert len(state.search_results) >= 1
    assert state.search_results[0].document_id == "doc-rag"

    # Run Verification
    state = verify_agent.verify(state)
    assert len(state.extracted_claims) >= 1
    assert len(state.verified_evidence) >= 1
    assert state.is_sufficient is True


def test_synthesis_agent():
    synthesizer = SynthesisAgent()
    store = VectorStore()
    ingestor = DocumentIngestor()
    doc = Document(
        id="doc-agentic",
        title="Agentic Frameworks",
        content="Agents use memory and tool execution to solve complex multi-step tasks.",
    )
    store.add_documents([doc], ingestor.ingest_document(doc))

    ev_manager = EvidenceManager(vector_store=store)
    researcher = ResearchAgent(vector_store=store)
    verifier = VerificationAgent(evidence_manager=ev_manager)

    state = ResearchState(user_query="Agentic Frameworks")
    state = PlanningAgent().plan(state)
    state = researcher.research(state)
    state = verifier.verify(state)
    state = synthesizer.synthesize(state)

    assert state.final_report is not None
    assert "# Research Intelligence Brief" in state.final_report
    assert "Agentic Frameworks" in state.final_report
    assert "[1]" in state.final_report
