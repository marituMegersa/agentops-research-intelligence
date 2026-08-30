"""End-to-end tests for the LangGraph multi-agent research workflow."""

import pytest
from app.graph.workflow import ResearchOrchestrator
from app.retrieval.models import Document
from app.retrieval.ingestion import DocumentIngestor
from app.retrieval.vector_store import VectorStore
from app.retrieval.evidence import EvidenceManager


def test_research_orchestrator_end_to_end():
    # Setup VectorStore with test knowledge base
    store = VectorStore()
    ingestor = DocumentIngestor()

    doc1 = Document(
        id="doc-rag-arch",
        title="Enterprise RAG Architecture",
        content="Enterprise RAG architectures utilize semantic chunking, dense embeddings, and reranking pipelines to deliver accurate grounded responses.",
    )
    doc2 = Document(
        id="doc-eval",
        title="Agent Evaluation Metrics",
        content="Evaluation frameworks measure faithfulness, answer relevancy, context precision, and latency.",
    )

    store.add_documents(
        [doc1, doc2],
        ingestor.ingest_document(doc1) + ingestor.ingest_document(doc2),
    )

    ev_manager = EvidenceManager(vector_store=store)
    orchestrator = ResearchOrchestrator(vector_store=store, evidence_manager=ev_manager)

    result_state = orchestrator.run(
        query="Explain Enterprise RAG Architecture and Agent Evaluation Metrics",
        max_iterations=2,
    )

    assert result_state.final_report is not None
    assert "Enterprise RAG Architecture" in result_state.final_report
    assert len(result_state.logs) >= 4
    agent_names_logged = {log.agent_name for log in result_state.logs}
    assert "PlanningAgent" in agent_names_logged
    assert "ResearchAgent" in agent_names_logged
    assert "VerificationAgent" in agent_names_logged
    assert "SynthesisAgent" in agent_names_logged
