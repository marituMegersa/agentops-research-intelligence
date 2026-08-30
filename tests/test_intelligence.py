"""Unit tests for dynamic tool registry, failure recovery, and human-in-the-loop orchestration."""

import pytest
from app.agents.recovery import FallbackExecutor, QueryReformulator
from app.agents.tools import ToolRegistry
from app.graph.hitl import HITLOrchestrator
from app.retrieval.models import Document
from app.retrieval.ingestion import DocumentIngestor
from app.retrieval.vector_store import VectorStore


def test_tool_registry_registration_and_execution():
    registry = ToolRegistry()

    @registry.register(name="calculator", description="Add two integers", tags=["math"])
    def add_numbers(a: int, b: int) -> int:
        return a + b

    tool_def = registry.get_tool("calculator")
    assert tool_def is not None
    assert tool_def.name == "calculator"
    assert "math" in tool_def.tags

    result = registry.execute("calculator", a=10, b=25)
    assert result == 35

    # Match tool
    selected = registry.select_tools_for_query("need to calculate math score", limit=1)
    assert len(selected) == 1
    assert selected[0].name == "calculator"


def test_query_reformulator():
    reformulator = QueryReformulator()
    variations = reformulator.reformulate("Please research enterprise RAG and agent systems")
    assert len(variations) >= 2
    assert any("retrieval augmented generation" in v for v in variations)


def test_fallback_executor():
    executor = FallbackExecutor(max_retries=1, backoff_seconds=0.01)

    # Success on fallback
    def failing_primary():
        raise ConnectionError("Primary database timeout")

    def fallback():
        return "fallback_success"

    result = executor.execute_with_fallback(failing_primary, fallback)
    assert result == "fallback_success"


def test_hitl_orchestration_flow():
    store = VectorStore()
    ingestor = DocumentIngestor()
    doc = Document(
        id="doc-hitl",
        title="HITL Evaluation Guide",
        content="Human-in-the-loop workflows enable manual approval of autonomous agent decision trees.",
    )
    store.add_documents([doc], ingestor.ingest_document(doc))

    hitl = HITLOrchestrator(vector_store=store)

    # 1. Start research -> pauses at plan_review
    cp1 = hitl.start_research("HITL Evaluation Guide")
    assert cp1.stage == "plan_review"
    assert not cp1.is_approved
    assert len(cp1.state.plan) >= 1

    # 2. Human approves with custom plan sub-queries -> advances to synthesis_review
    cp2 = hitl.submit_plan_approval(
        cp1,
        approved=True,
        override_sub_queries=["HITL Evaluation Guide decision trees"],
    )
    assert cp2.stage == "synthesis_review"
    assert len(cp2.state.search_results) >= 1

    # 3. Finalize synthesis
    cp3 = hitl.finalize_synthesis(cp2, approved=True)
    assert cp3.stage == "completed"
    assert cp3.is_approved is True
    assert cp3.state.final_report is not None
    assert "HITL Evaluation Guide" in cp3.state.final_report
