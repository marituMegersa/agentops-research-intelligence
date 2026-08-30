"""Execution tracer, OpenTelemetry-aligned span profiler, and latency observability."""

from __future__ import annotations

import datetime
import time
from typing import Any, Dict, List, Optional
import uuid
from pydantic import BaseModel, Field


class TraceSpan(BaseModel):
    """Execution span recording a discrete agent step or tool call."""

    span_id: str = Field(default_factory=lambda: f"span-{uuid.uuid4().hex[:8]}")
    name: str
    agent_name: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: float = 0.0
    status: str = "running"  # running, success, error
    input_payload: Dict[str, Any] = Field(default_factory=dict)
    output_payload: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    estimated_tokens: int = 0


class ExecutionTracer:
    """Collects, profiles, and analyzes execution spans across multi-agent workflows."""

    def __init__(self, trace_id: Optional[str] = None):
        self.trace_id = trace_id or f"trc-{uuid.uuid4().hex[:8]}"
        self.spans: List[TraceSpan] = []
        self._active_spans: Dict[str, TraceSpan] = {}

    def start_span(
        self,
        name: str,
        agent_name: str,
        input_payload: Optional[Dict[str, Any]] = None,
    ) -> TraceSpan:
        """Begin tracking an execution span."""
        span = TraceSpan(
            name=name,
            agent_name=agent_name,
            start_time=time.perf_counter(),
            input_payload=input_payload or {},
        )
        self.spans.append(span)
        self._active_spans[span.span_id] = span
        return span

    def end_span(
        self,
        span_id: str,
        output_payload: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        estimated_tokens: int = 0,
    ) -> Optional[TraceSpan]:
        """Complete an execution span and record duration."""
        span = self._active_spans.pop(span_id, None)
        if not span:
            return None

        span.end_time = time.perf_counter()
        span.duration_ms = round((span.end_time - span.start_time) * 1000, 2)
        span.output_payload = output_payload or {}
        span.estimated_tokens = estimated_tokens

        if error_message:
            span.status = "error"
            span.error_message = error_message
        else:
            span.status = "success"

        return span

    def get_total_duration_ms(self) -> float:
        """Return total cumulative runtime in milliseconds."""
        return sum(s.duration_ms for s in self.spans)

    def get_total_tokens(self) -> int:
        """Return cumulative estimated token usage."""
        return sum(s.estimated_tokens for s in self.spans)

    def get_summary(self) -> Dict[str, Any]:
        """Generate a complete telemetry summary profile."""
        durations_by_agent: Dict[str, float] = {}
        for s in self.spans:
            durations_by_agent[s.agent_name] = round(
                durations_by_agent.get(s.agent_name, 0.0) + s.duration_ms, 2
            )

        return {
            "trace_id": self.trace_id,
            "total_spans": len(self.spans),
            "total_duration_ms": round(self.get_total_duration_ms(), 2),
            "total_tokens": self.get_total_tokens(),
            "latency_breakdown_by_agent": durations_by_agent,
            "all_spans_successful": all(s.status == "success" for s in self.spans),
        }
