"""Shared state schema for multi-agent research orchestration."""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional
from typing_extensions import Annotated
import operator
from pydantic import BaseModel, Field

from app.retrieval.models import Claim, Evidence, SearchResult


class ResearchPlanItem(BaseModel):
    """An individual sub-task or search query created by the planning agent."""

    sub_query: str
    rationale: str = ""
    status: str = "pending"  # pending, completed, failed


class AgentLog(BaseModel):
    """Execution step log emitted by an agent."""

    agent_name: str
    action: str
    details: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )


class ResearchState(BaseModel):
    """Global state passed across all nodes in the LangGraph research graph."""

    user_query: str
    plan: List[ResearchPlanItem] = Field(default_factory=list)
    search_results: List[SearchResult] = Field(default_factory=list)
    extracted_claims: List[Claim] = Field(default_factory=list)
    verified_evidence: List[Evidence] = Field(default_factory=list)
    final_report: Optional[str] = None
    iteration_count: int = 0
    max_iterations: int = 3
    is_sufficient: bool = False
    logs: Annotated[List[AgentLog], operator.add] = Field(default_factory=list)
