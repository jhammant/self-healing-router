"""Self-Healing Router — Graph-based tool routing for LLM agents.

93% fewer LLM calls than ReAct. Same correctness. Automatic recovery.

    from self_healing_router import SelfHealingRouter

    router = SelfHealingRouter()
    router.add_tool("fetch", handler=fetch_fn)
    router.add_tool("parse", handler=parse_fn)
    router.add_tool("store", handler=store_fn)
    router.add_edge("fetch", "parse")
    router.add_edge("parse", "store")

    result = router.route("fetch", "store")
    # Zero LLM calls for normal operation
    # Auto-reroutes on failure
    # LLM called only when no path exists
"""

__version__ = "0.1.0"

from .types import (
    ToolStatus,
    ToolNode,
    Edge,
    HealthReport,
    RouteResult,
    EscalationResult,
    CircuitBreakerState,
    GoalDemotion,
    ExecutionOutcome,
)
from .graph import ToolGraph
from .monitor import HealthMonitor, MonitorRegistry
from .escalation import EscalationHandler, default_escalation
from .priority import PrioritySignal, PriorityArbiter
from .router import SelfHealingRouter

__all__ = [
    "SelfHealingRouter",
    "ToolGraph",
    "HealthMonitor",
    "MonitorRegistry",
    "EscalationHandler",
    "default_escalation",
    "ToolStatus",
    "ToolNode",
    "Edge",
    "HealthReport",
    "RouteResult",
    "EscalationResult",
    "CircuitBreakerState",
    "GoalDemotion",
    "ExecutionOutcome",
    "PrioritySignal",
    "PriorityArbiter",
]
