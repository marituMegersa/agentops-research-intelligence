"""Planning agent that decomposes complex research topics into targeted sub-queries."""

from __future__ import annotations

import re
from typing import List, Optional
from app.agents.state import AgentLog, ResearchPlanItem, ResearchState


class PlanningAgent:
    """Decomposes a user query into discrete research sub-goals."""

    def __init__(self, max_subqueries: int = 3):
        self.max_subqueries = max_subqueries

    def plan(self, state: ResearchState) -> ResearchState:
        """Analyze the query and populate the research plan."""
        query = state.user_query.strip()
        sub_queries: List[ResearchPlanItem] = []

        # Always include the core query
        sub_queries.append(
            ResearchPlanItem(
                sub_query=query,
                rationale="Core research prompt lookup",
            )
        )

        # Decompose compound phrases / clauses if present
        conjunction_parts = re.split(r"\b(?:and|vs|versus|compared to|recommend|how to|what is)\b", query, flags=re.IGNORECASE)
        for part in conjunction_parts:
            cleaned = part.strip()
            if len(cleaned) > 5 and cleaned.lower() != query.lower():
                sub_queries.append(
                    ResearchPlanItem(
                        sub_query=cleaned,
                        rationale=f"Specific sub-topic exploration for: '{cleaned}'",
                    )
                )

        # Limit to max_subqueries
        planned_items = sub_queries[: self.max_subqueries]
        state.plan = planned_items

        state.logs.append(
            AgentLog(
                agent_name="PlanningAgent",
                action="create_plan",
                details={
                    "query": query,
                    "sub_queries_count": len(planned_items),
                    "sub_queries": [p.sub_query for p in planned_items],
                },
            )
        )

        return state
