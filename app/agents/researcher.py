"""Research and Retrieval Agent that executes searches across vector and knowledge stores."""

from __future__ import annotations

from typing import Dict, List, Optional
from app.agents.state import AgentLog, ResearchState
from app.retrieval.models import SearchResult
from app.retrieval.vector_store import VectorStore


class ResearchAgent:
    """Executes planned queries against the Vector Store and retrieval layer."""

    def __init__(self, vector_store: VectorStore, search_limit_per_query: int = 3):
        self.vector_store = vector_store
        self.search_limit_per_query = search_limit_per_query

    def research(self, state: ResearchState) -> ResearchState:
        """Execute searches for all pending sub-queries in state.plan."""
        existing_results_by_chunk: Dict[str, SearchResult] = {
            r.chunk_id: r for r in state.search_results
        }
        queries_executed: List[str] = []

        for plan_item in state.plan:
            if plan_item.status != "completed":
                matches = self.vector_store.search_hybrid(
                    query=plan_item.sub_query,
                    limit=self.search_limit_per_query,
                )
                for match in matches:
                    if match.chunk_id not in existing_results_by_chunk:
                        existing_results_by_chunk[match.chunk_id] = match

                plan_item.status = "completed"
                queries_executed.append(plan_item.sub_query)

        state.search_results = list(existing_results_by_chunk.values())
        state.logs.append(
            AgentLog(
                agent_name="ResearchAgent",
                action="execute_searches",
                details={
                    "executed_queries": queries_executed,
                    "total_unique_chunks_found": len(state.search_results),
                },
            )
        )

        return state
