"""Type definitions for self-healing-router."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable


class ToolStatus(Enum):
    """Health status of a tool node."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    UNKNOWN = "unknown"


class CircuitBreakerState(Enum):
    """Circuit breaker states for tool health."""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"            # Tool confirmed down, weight = infinity
    HALF_OPEN = "half_open"  # Recovery suspected, send one probe


class ExecutionOutcome(Enum):
    """Outcome of a routing execution."""
    SUCCESS = "success"            # Completed via rerouted or primary path
    ESCALATED = "escalated"        # LLM was called, may have demoted goal
    MAX_REROUTES = "max_reroutes"  # Hit reroute limit


@dataclass
class ToolNode:
    """A tool in the routing graph."""
    name: str
    handler: Callable[..., Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    status: ToolStatus = ToolStatus.HEALTHY


@dataclass
class Edge:
    """A weighted directed edge between two tools."""
    source: str
    target: str
    base_weight: float = 1.0
    current_weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def fail(self) -> None:
        """Mark edge as failed (infinite weight)."""
        self.current_weight = float("inf")

    def recover(self) -> None:
        """Reset edge to base weight."""
        self.current_weight = self.base_weight

    @property
    def is_failed(self) -> bool:
        return self.current_weight == float("inf")


@dataclass
class HealthReport:
    """Health report from a monitor."""
    tool_name: str
    status: ToolStatus
    latency_ms: float | None = None
    error_rate: float | None = None
    priority: int = 0  # Higher = more urgent
    detail: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class GoalDemotion:
    """Represents a demotion from original goal to a lesser goal."""
    original_goal: str
    demoted_goal: str
    reason: str
    fallback_action: str | None = None


@dataclass
class RouteResult:
    """Result of a routing attempt."""
    path: list[str]
    total_weight: float
    success: bool
    outputs: dict[str, Any] = field(default_factory=dict)
    reroutes: int = 0
    llm_calls: int = 0
    errors: list[str] = field(default_factory=list)
    outcome: ExecutionOutcome = ExecutionOutcome.SUCCESS
    execution_log: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class EscalationResult:
    """Result from LLM escalation."""
    action: str  # "retry", "skip", "abort", "alternative"
    detail: str = ""
    alternative_path: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    demotion: GoalDemotion | None = None


# Type aliases for callbacks
ToolHandler = Callable[..., Any]
AsyncToolHandler = Callable[..., Awaitable[Any]]
EscalationCallback = Callable[[str, list[str], dict[str, Any]], EscalationResult]
