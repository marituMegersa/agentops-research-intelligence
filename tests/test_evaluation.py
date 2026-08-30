"""Unit tests for Evaluation Suite: Metrics, Dataset, Tracer, and Benchmark Runner."""

import pytest
from app.evaluation.dataset import BenchmarkDataset, EvaluationSample
from app.evaluation.metrics import (
    calculate_answer_relevance,
    calculate_citation_accuracy,
    calculate_context_precision,
    calculate_context_recall,
    calculate_faithfulness,
    calculate_reciprocal_rank,
    evaluate_research_sample,
)
from app.evaluation.tracer import ExecutionTracer
from app.evaluation.runner import EvaluationRunner
from app.retrieval.models import Claim, Evidence, SearchResult, VerificationStatus


def test_retrieval_metrics():
    retrieved = [
        SearchResult(
            chunk_id="c1",
            document_id="doc-1",
            title="Doc 1",
            text="text 1",
            score=0.9,
        ),
        SearchResult(
            chunk_id="c2",
            document_id="doc-2",
            title="Doc 2",
            text="text 2",
            score=0.8,
        ),
        SearchResult(
            chunk_id="c3",
            document_id="doc-3",
            title="Doc 3",
            text="text 3",
            score=0.5,
        ),
    ]

    expected = ["doc-1", "doc-2"]

    prec = calculate_context_precision(retrieved, expected)
    rec = calculate_context_recall(retrieved, expected)
    mrr = calculate_reciprocal_rank(retrieved, expected)

    assert round(prec, 2) == 0.67
    assert rec == 1.0
    assert mrr == 1.0


def test_faithfulness_and_citation_metrics():
    claim1 = Claim(
        id="clm-1",
        text="RAG reduces hallucinations.",
        status=VerificationStatus.VERIFIED,
        supporting_evidence_ids=["evi-1"],
    )
    claim2 = Claim(
        id="clm-2",
        text="Agents can plan multi-step workflows.",
        status=VerificationStatus.VERIFIED,
        supporting_evidence_ids=["evi-2"],
    )

    faithfulness = calculate_faithfulness([claim1, claim2], [])
    assert faithfulness == 1.0

    report = """# Brief
- Verified finding: RAG reduces hallucinations [[1]](#ref-1)
<a id="ref-1"></a>[1] **RAG Paper** (`doc-1`)
"""
    citation_acc = calculate_citation_accuracy(report, ["doc-1"])
    assert citation_acc == 1.0

    relevance = calculate_answer_relevance("Explain RAG hallucinations", report)
    assert relevance >= 0.5


def test_execution_tracer():
    tracer = ExecutionTracer()
    span1 = tracer.start_span("planning_step", "PlanningAgent", {"query": "test query"})
    tracer.end_span(span1.span_id, {"status": "ok"}, estimated_tokens=150)

    span2 = tracer.start_span("research_step", "ResearchAgent")
    tracer.end_span(span2.span_id, estimated_tokens=300)

    summary = tracer.get_summary()
    assert summary["total_spans"] == 2
    assert summary["total_tokens"] == 450
    assert summary["all_spans_successful"] is True
    assert "PlanningAgent" in summary["latency_breakdown_by_agent"]


def test_evaluation_runner_benchmark():
    runner = EvaluationRunner()
    dataset = BenchmarkDataset(
        name="test-mini-benchmark",
        samples=[
            EvaluationSample(
                id="t-001",
                query="What is Retrieval-Augmented Generation?",
                expected_source_ids=["doc-001"],
            )
        ],
    )

    report = runner.run_benchmark(dataset)
    assert report.total_samples == 1
    assert report.avg_composite_score > 0.5
    assert report.avg_context_recall == 1.0

    md = report.to_markdown()
    assert "# 📊 Benchmark Evaluation Report" in md
    assert "| `t-001` |" in md
