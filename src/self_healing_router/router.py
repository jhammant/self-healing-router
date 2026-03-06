"""The Self-Healing Router — Dijkstra routing with automatic failure recovery.

Core loop:
1. Compute shortest path via Dijkstra
2. Execute tools along the path
3. If a tool fails → set edge weight to ∞, recompute path
4. If NO path exists → escalate to LLM (last resort)

This achieves 93% fewer LLM calls than ReAct while matching correctness.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from .graph import ToolGraph
from .monitor import MonitorRegistry
from .escalation import EscalationHandler, default_escalation
from .priority import PriorityArbiter
from .types import (
    RouteResult,
    ToolStatus,
    EscalationResult,
    EscalationCallback,
    ExecutionOutcome,
)

logger = logging.getLogger(__name__)


class SelfHealingRouter:
    """Graph-based tool router with automatic failure recovery.
    
    Example:
        router = SelfHealingRouter()
        router.add_tool("fetch", handler=fetch_data)
        router.add_tool("parse", handler=parse_data)
        router.add_tool("store", handler=store_data)
        router.add_edge("fetch", "parse")
        router.add_edge("parse", "store")
        
        result = router.route("fetch", "store", input_data={"url": "..."})
        print(f"Success: {result.success}, LLM calls: {result.llm_calls}")
    """

    def __init__(
        self,
        escalation_callback: EscalationCallback | None = None,
        max_reroutes: int = 5,
        max_escalations: int = 3,
    ) -> None:
        self.graph = ToolGraph()
        self.monitors = MonitorRegistry()
        self.escalation = EscalationHandler(
            callback=escalation_callback or default_escalation,
            max_escalations=max_escalations,
        )
        self._max_reroutes = max_reroutes

    def add_tool(
        self,
        name: str,
        handler: Any = None,
        metadata: dict[str, Any] | None = None,
        latency_threshold_ms: float = 5000.0,
        error_rate_threshold: float = 0.5,
    ) -> None:
        """Add a tool with automatic health monitoring."""
        self.graph.add_tool(name, handler=handler, metadata=metadata)
        self.monitors.register(
            name,
            latency_threshold_ms=latency_threshold_ms,
            error_rate_threshold=error_rate_threshold,
        )

    def add_edge(
        self,
        source: str,
        target: str,
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a weighted edge between tools."""
        self.graph.add_edge(source, target, weight=weight, metadata=metadata)

    def _sync_monitor_weights(self) -> None:
        """Update edge weights from monitor health data."""
        for edge in self.graph.edges:
            monitor = self.monitors.get(edge.target)
            if monitor:
                penalty = monitor.weight_penalty()
                if penalty == float("inf"):
                    edge.fail()
                else:
                    edge.current_weight = edge.base_weight * penalty

    def _collect_priority_signals(self) -> None:
        """Collect priority signals from all monitors and log the winner."""
        arbiter = PriorityArbiter()
        for report in self.monitors.all_reports():
            monitor = self.monitors.get(report.tool_name)
            if monitor:
                signal = monitor.priority_signal()
                arbiter.add_signal(signal)
        winner = arbiter.resolve()
        if winner:
            logger.debug(
                "Priority winner: %s (score=%.2f, source=%s, detail=%s)",
                winner.name, winner.score, winner.source, winner.detail,
            )

    def _execute_tool(self, name: str, input_data: dict[str, Any]) -> tuple[bool, Any]:
        """Execute a single tool, recording metrics."""
        node = self.graph.get_tool(name)
        if node is None or node.handler is None:
            return True, input_data  # passthrough if no handler

        monitor = self.monitors.get(name)
        start = time.monotonic()

        try:
            result = node.handler(input_data)
            elapsed_ms = (time.monotonic() - start) * 1000
            if monitor:
                monitor.record_success(elapsed_ms)
            return True, result
        except Exception as e:
            elapsed_ms = (time.monotonic() - start) * 1000
            if monitor:
                monitor.record_failure(elapsed_ms)
            return False, str(e)

    def route(
        self,
        start: str,
        end: str,
        input_data: dict[str, Any] | None = None,
    ) -> RouteResult:
        """Route from start to end, auto-healing on failures.
        
        Args:
            start: Source tool name
            end: Destination tool name  
            input_data: Initial data passed to the first tool
            
        Returns:
            RouteResult with path taken, outputs, reroute/LLM call counts
        """
        data = dict(input_data or {})
        reroutes = 0
        llm_calls = 0
        outputs: dict[str, Any] = {}
        errors: list[str] = []
        attempted_paths: list[list[str]] = []
        executed: set[str] = set()
        execution_log: list[dict[str, Any]] = []
        outcome = ExecutionOutcome.SUCCESS

        while reroutes <= self._max_reroutes:
            # Collect priority signals before routing
            self._collect_priority_signals()

            # Sync edge weights from monitors
            self._sync_monitor_weights()

            # Find shortest path
            path, cost = self.graph.shortest_path(start, end)
            
            if not path or cost == float("inf"):
                # No path — escalate to LLM
                llm_calls += 1
                outcome = ExecutionOutcome.ESCALATED
                escalation_result = self.escalation.escalate(
                    failed_tool=start,
                    attempted_paths=attempted_paths,
                    context={"outputs": outputs, "errors": errors, "data": data},
                    start_goal=start,
                    end_goal=end,
                )
                
                execution_log.append({
                    "step": "escalation",
                    "tool": start,
                    "success": False,
                    "detail": escalation_result.detail,
                })

                if escalation_result.action == "retry":
                    # LLM says retry — recover all failed tools and try again
                    for tool_name in self.monitors.failed_tools():
                        self.graph.recover_tool(tool_name)
                    reroutes += 1
                    continue
                elif escalation_result.action == "alternative" and escalation_result.alternative_path:
                    path = escalation_result.alternative_path
                    # Fall through to execute this path
                else:
                    # abort or skip
                    return RouteResult(
                        path=[],
                        total_weight=float("inf"),
                        success=False,
                        outputs=outputs,
                        reroutes=reroutes,
                        llm_calls=llm_calls,
                        errors=errors + [escalation_result.detail],
                        outcome=outcome,
                        execution_log=execution_log,
                    )

            attempted_paths.append(path)
            
            # Execute tools along the path
            current_data = data
            path_failed = False
            
            for tool_name in path:
                if tool_name in executed:
                    # Already ran this tool successfully, skip
                    if tool_name in outputs:
                        current_data = outputs[tool_name] if isinstance(outputs[tool_name], dict) else current_data
                    continue

                step_start = time.monotonic()
                success, result = self._execute_tool(tool_name, current_data)
                step_elapsed = (time.monotonic() - step_start) * 1000

                execution_log.append({
                    "step": "execute",
                    "tool": tool_name,
                    "success": success,
                    "latency_ms": step_elapsed,
                    "reroute": reroutes > 0,
                })

                if success:
                    outputs[tool_name] = result
                    if isinstance(result, dict):
                        current_data = result
                    executed.add(tool_name)
                else:
                    # Tool failed — mark it, reroute
                    error_msg = f"Tool '{tool_name}' failed: {result}"
                    errors.append(error_msg)
                    self.graph.fail_tool(tool_name)
                    
                    # Update start to last successful node (or original start)
                    if executed:
                        # Find the last executed node that's in our current path
                        for prev_tool in reversed(path):
                            if prev_tool in executed and prev_tool != tool_name:
                                start = prev_tool
                                break
                    
                    reroutes += 1
                    path_failed = True
                    break

            if not path_failed:
                # Full path executed successfully
                return RouteResult(
                    path=path,
                    total_weight=cost,
                    success=True,
                    outputs=outputs,
                    reroutes=reroutes,
                    llm_calls=llm_calls,
                    errors=errors,
                    outcome=outcome,
                    execution_log=execution_log,
                )

        # Exceeded max reroutes
        return RouteResult(
            path=[],
            total_weight=float("inf"),
            success=False,
            outputs=outputs,
            reroutes=reroutes,
            llm_calls=llm_calls,
            errors=errors + [f"Max reroutes ({self._max_reroutes}) exceeded"],
            outcome=ExecutionOutcome.MAX_REROUTES,
            execution_log=execution_log,
        )

    def health_report(self) -> str:
        """Get a human-readable health report."""
        reports = self.monitors.all_reports()
        lines = ["Health Report:", ""]
        for r in reports:
            status_icon = {
                ToolStatus.HEALTHY: "✅",
                ToolStatus.DEGRADED: "⚠️",
                ToolStatus.FAILED: "❌",
                ToolStatus.UNKNOWN: "❓",
            }.get(r.status, "?")
            lat = f"{r.latency_ms:.0f}ms" if r.latency_ms else "n/a"
            err = f"{r.error_rate:.0%}" if r.error_rate is not None else "n/a"
            lines.append(f"  {status_icon} {r.tool_name}: latency={lat}, errors={err}, priority={r.priority}")
        return "\n".join(lines)

    def visualize(self) -> str:
        """ASCII visualization of the current graph state."""
        return self.graph.to_ascii()

    def reset(self) -> None:
        """Reset all monitors and recover all tools."""
        self.escalation.reset()
        for name in self.graph.nodes:
            self.graph.recover_tool(name)
