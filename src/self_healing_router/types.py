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
class RouteResult:
    """Result of a routing attempt."""
    path: list[str]
    total_weight: float
    success: bool
    outputs: dict[str, Any] = field(default_factory=dict)
    reroutes: int = 0
    llm_calls: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class EscalationResult:
    """Result from LLM escalation."""
    action: str  # "retry", "skip", "abort", "alternative"
    detail: str = ""
    alternative_path: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# Type aliases for callbacks
ToolHandler = Callable[..., Any]
AsyncToolHandler = Callable[..., Awaitable[Any]]
EscalationCallback = Callable[[str, list[str], dict[str, Any]], EscalationResult]
