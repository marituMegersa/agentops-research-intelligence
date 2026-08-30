"""Failure recovery, query reformulation, and fallback execution handlers."""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class QueryReformulator:
    """Reformulates search queries when initial searches yield insufficient or empty results."""

    STOP_PHRASES = [
        "can you", "please", "research", "tell me about", "what are the details of",
        "find information on", "look up", "how does", "what is", "why is",
    ]

    SYNONYM_EXPANSIONS = {
        "rag": ["retrieval augmented generation", "vector search context"],
        "agent": ["autonomous system", "agentic tool execution"],
        "eval": ["benchmarking", "faithfulness metrics", "reliability"],
        "llm": ["large language model", "generative ai"],
    }

    def reformulate(self, query: str) -> List[str]:
        """Generate alternative search query variations."""
        cleaned = query.strip().lower()
        for phrase in self.STOP_PHRASES:
            cleaned = re.sub(rf"^{phrase}\s+", "", cleaned, flags=re.IGNORECASE)

        variations: List[str] = [cleaned]

        # Add synonym-expanded variations
        tokens = cleaned.split()
        for token in tokens:
            if token in self.SYNONYM_EXPANSIONS:
                for syn in self.SYNONYM_EXPANSIONS[token]:
                    variations.append(cleaned.replace(token, syn))

        # Add single keyword isolate if multi-word
        keywords = [w for w in tokens if len(w) > 3]
        if len(keywords) > 1:
            variations.append(" ".join(keywords[:2]))

        # Deduplicate while preserving order
        seen = set()
        unique_variations = []
        for v in variations:
            if v and v not in seen:
                seen.add(v)
                unique_variations.append(v)

        return unique_variations


class FallbackExecutor:
    """Executes actions with retry policy and fallback degradation."""

    def __init__(self, max_retries: int = 2, backoff_seconds: float = 0.05):
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds

    def execute_with_fallback(
        self,
        primary_action: Callable[..., Any],
        fallback_action: Optional[Callable[..., Any]] = None,
        *args,
        **kwargs,
    ) -> Any:
        """Attempt primary action with retries; switch to fallback if primary fails."""
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return primary_action(*args, **kwargs)
            except Exception as exc:
                last_exception = exc
                logger.warning(
                    f"Action failed on attempt {attempt + 1}/{self.max_retries + 1}: {exc}"
                )
                if attempt < self.max_retries and self.backoff_seconds > 0:
                    time.sleep(self.backoff_seconds * (2**attempt))

        if fallback_action:
            logger.info("Engaging fallback action.")
            return fallback_action(*args, **kwargs)

        raise last_exception or RuntimeError("Execution failed without specific exception.")
