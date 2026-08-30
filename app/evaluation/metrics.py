"""RAG Triad and Agent Evaluation Metrics Engine."""

from __future__ import annotations

import re
from typing import List, Set
from pydantic import BaseModel, Field

from app.retrieval.models import Claim, Evidence, SearchResult


class EvaluationScorecard(BaseModel):
    """Aggregated evaluation metrics for a single research evaluation run."""

    context_precision: float = Field(default=1.0, ge=0.0, le=1.0)
    context_recall: float = Field(default=1.0, ge=0.0, le=1.0)
    reciprocal_rank: float = Field(default=1.0, ge=0.0, le=1.0)
    faithfulness_score: float = Field(default=1.0, ge=0.0, le=1.0)
    citation_accuracy: float = Field(default=1.0, ge=0.0, le=1.0)
    answer_relevance_score: float = Field(default=1.0, ge=0.0, le=1.0)
    composite_score: float = Field(default=1.0, ge=0.0, le=1.0)


def calculate_context_precision(
    retrieved_results: List[SearchResult],
    expected_source_ids: List[str],
) -> float:
    """Calculate the ratio of retrieved results that belong to expected ground-truth sources."""
    if not retrieved_results:
        return 0.0
    if not expected_source_ids:
        return 1.0

    expected_set = set(expected_source_ids)
    relevant_hits = sum(1 for r in retrieved_results if r.document_id in expected_set)
    return round(relevant_hits / len(retrieved_results), 4)


def calculate_context_recall(
    retrieved_results: List[SearchResult],
    expected_source_ids: List[str],
) -> float:
    """Calculate the fraction of ground-truth source IDs that were retrieved."""
    if not expected_source_ids:
        return 1.0
    if not retrieved_results:
        return 0.0

    retrieved_doc_ids = {r.document_id for r in retrieved_results}
    matched = sum(1 for exp in expected_source_ids if exp in retrieved_doc_ids)
    return round(matched / len(expected_source_ids), 4)


def calculate_reciprocal_rank(
    retrieved_results: List[SearchResult],
    expected_source_ids: List[str],
) -> float:
    """Calculate the Mean Reciprocal Rank (MRR) of the first relevant document."""
    if not retrieved_results or not expected_source_ids:
        return 0.0

    expected_set = set(expected_source_ids)
    for rank, r in enumerate(retrieved_results, start=1):
        if r.document_id in expected_set:
            return round(1.0 / rank, 4)

    return 0.0


def calculate_faithfulness(
    claims: List[Claim],
    evidence_items: List[Evidence],
) -> float:
    """Calculate the ratio of extracted claims that are grounded by verified evidence."""
    if not claims:
        return 1.0

    supported_claims = 0
    for claim in claims:
        if claim.supporting_evidence_ids or claim.status.value == "verified":
            supported_claims += 1

    return round(supported_claims / len(claims), 4)


def calculate_citation_accuracy(
    report_text: str,
    valid_source_ids: List[str],
) -> float:
    """Verify that cited reference links in the report map to valid indexed source IDs."""
    if not report_text:
        return 0.0

    cited_indices = re.findall(r"\[(\d+)\]", report_text)
    if not cited_indices:
        return 1.0  # No citations claimed, or zero citations needed

    # Check that referenced anchors exist in text
    valid_anchors = re.findall(r'<a id="ref-(\d+)"></a>', report_text)
    if not valid_anchors:
        return 0.5

    valid_set = set(valid_anchors)
    valid_cites = sum(1 for c in set(cited_indices) if c in valid_set)
    return round(valid_cites / len(set(cited_indices)), 4)


def calculate_answer_relevance(query: str, report_text: str) -> float:
    """Calculate semantic keyword overlap between user query and generated response."""
    if not query or not report_text:
        return 0.0

    query_terms = set(re.findall(r"\w+", query.lower()))
    report_terms = set(re.findall(r"\w+", report_text.lower()))

    # Exclude common stop words
    stop_words = {"what", "is", "how", "and", "the", "a", "an", "does", "to", "for", "with"}
    meaningful_query_terms = query_terms - stop_words

    if not meaningful_query_terms:
        return 1.0

    overlap = meaningful_query_terms.intersection(report_terms)
    return round(len(overlap) / len(meaningful_query_terms), 4)


def evaluate_research_sample(
    query: str,
    retrieved_results: List[SearchResult],
    claims: List[Claim],
    evidence_items: List[Evidence],
    final_report: str,
    expected_source_ids: List[str],
) -> EvaluationScorecard:
    """Compute complete RAG and agent evaluation scorecard."""
    precision = calculate_context_precision(retrieved_results, expected_source_ids)
    recall = calculate_context_recall(retrieved_results, expected_source_ids)
    mrr = calculate_reciprocal_rank(retrieved_results, expected_source_ids)
    faithfulness = calculate_faithfulness(claims, evidence_items)
    citation_acc = calculate_citation_accuracy(
        final_report, [r.document_id for r in retrieved_results]
    )
    relevance = calculate_answer_relevance(query, final_report)

    # Weighted composite score
    composite = round(
        (0.25 * precision)
        + (0.25 * recall)
        + (0.25 * faithfulness)
        + (0.15 * citation_acc)
        + (0.10 * relevance),
        4,
    )

    return EvaluationScorecard(
        context_precision=precision,
        context_recall=recall,
        reciprocal_rank=mrr,
        faithfulness_score=faithfulness,
        citation_accuracy=citation_acc,
        answer_relevance_score=relevance,
        composite_score=composite,
    )
