"""Evaluation, Benchmarks, Metrics, and Observability Suite."""

from app.evaluation.dataset import BenchmarkDataset, EvaluationSample
from app.evaluation.metrics import (
    EvaluationScorecard,
    calculate_answer_relevance,
    calculate_citation_accuracy,
    calculate_context_precision,
    calculate_context_recall,
    calculate_faithfulness,
    calculate_reciprocal_rank,
    evaluate_research_sample,
)
from app.evaluation.tracer import ExecutionTracer, TraceSpan
from app.evaluation.runner import BenchmarkRunReport, EvaluationRunner, SampleEvaluationResult

__all__ = [
    "EvaluationSample",
    "BenchmarkDataset",
    "EvaluationScorecard",
    "calculate_context_precision",
    "calculate_context_recall",
    "calculate_reciprocal_rank",
    "calculate_faithfulness",
    "calculate_citation_accuracy",
    "calculate_answer_relevance",
    "evaluate_research_sample",
    "TraceSpan",
    "ExecutionTracer",
    "SampleEvaluationResult",
    "BenchmarkRunReport",
    "EvaluationRunner",
]
