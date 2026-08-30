"""Benchmark evaluation dataset and ground-truth sample models."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import uuid
from pydantic import BaseModel, Field


class EvaluationSample(BaseModel):
    """An individual research benchmark question with ground-truth verification targets."""

    id: str = Field(default_factory=lambda: f"eval-{uuid.uuid4().hex[:8]}")
    query: str
    ground_truth_claims: List[str] = Field(default_factory=list)
    expected_source_ids: List[str] = Field(default_factory=list)
    category: str = "general"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BenchmarkDataset(BaseModel):
    """Collection of evaluation samples for systematic agent benchmarking."""

    name: str = "standard-research-benchmarks"
    version: str = "1.0.0"
    samples: List[EvaluationSample] = Field(default_factory=list)

    @classmethod
    def get_default_benchmarks(cls) -> BenchmarkDataset:
        """Provide a curated set of enterprise research benchmark questions."""
        return cls(
            name="enterprise-ai-research-v1",
            samples=[
                EvaluationSample(
                    id="eval-001",
                    query="What is Retrieval-Augmented Generation and how does it combine search with LLMs?",
                    ground_truth_claims=[
                        "RAG combines retrieval with language generation to provide models with external context.",
                        "Grounding with citations reduces error rates and hallucinations.",
                    ],
                    expected_source_ids=["doc-001"],
                    category="retrieval",
                ),
                EvaluationSample(
                    id="eval-002",
                    query="How do agentic AI systems execute autonomous planning and tool selection?",
                    ground_truth_claims=[
                        "Agentic AI systems use models to plan tasks, select tools, execute actions, and adapt to results.",
                    ],
                    expected_source_ids=["doc-002"],
                    category="agentic_ai",
                ),
                EvaluationSample(
                    id="eval-003",
                    query="How is AI evaluation performed and what reliability metrics are tracked?",
                    ground_truth_claims=[
                        "Evaluation measures the quality, reliability, and performance of AI systems.",
                    ],
                    expected_source_ids=["doc-003"],
                    category="evaluation",
                ),
            ],
        )
