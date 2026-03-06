"""Health monitors for tool nodes — latency, error rates, custom signals."""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Callable

from .types import ToolStatus, HealthReport, CircuitBreakerState


class HealthMonitor:
    """Tracks health metrics for a tool and produces HealthReports.
    
    Monitors run in parallel (conceptually), feeding priority scores
    back to the router which adjusts edge weights accordingly.
    """

    def __init__(
        self,
        tool_name: str,
        latency_threshold_ms: float = 5000.0,
        error_rate_threshold: float = 0.5,
        window_size: int = 20,
    ) -> None:
        self.tool_name = tool_name
        self.latency_threshold_ms = latency_threshold_ms
        self.error_rate_threshold = error_rate_threshold
        self._window_size = window_size
        
        self._latencies: deque[float] = deque(maxlen=window_size)
        self._outcomes: deque[bool] = deque(maxlen=window_size)  # True=success
        self._custom_signals: dict[str, float] = {}

        # Circuit breaker state
        self._circuit_state: CircuitBreakerState = CircuitBreakerState.CLOSED
        self._open_since: float | None = None
        self._recovery_timeout: float = 30.0
        self._consecutive_failures: int = 0
        self._failure_threshold: int = 3

        # Rate limit tracking
        self._rate_limit_remaining: int | None = None
        self._rate_limit_total: int | None = None

        # Proactive health check
        self._health_check: Callable[[], bool] | None = None

    def record_success(self, latency_ms: float) -> None:
        """Record a successful tool invocation."""
        self._latencies.append(latency_ms)
        self._outcomes.append(True)
        self._consecutive_failures = 0
        if self._circuit_state == CircuitBreakerState.HALF_OPEN:
            self._circuit_state = CircuitBreakerState.CLOSED
            self._open_since = None

    def record_failure(self, latency_ms: float = 0.0) -> None:
        """Record a failed tool invocation."""
        if latency_ms > 0:
            self._latencies.append(latency_ms)
        self._outcomes.append(False)
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._failure_threshold:
            self._circuit_state = CircuitBreakerState.OPEN
            self._open_since = time.monotonic()

    def set_signal(self, name: str, value: float) -> None:
        """Set a custom health signal (e.g., queue depth, memory usage)."""
        self._custom_signals[name] = value

    @property
    def avg_latency_ms(self) -> float | None:
        """Average latency over the sliding window."""
        if not self._latencies:
            return None
        return sum(self._latencies) / len(self._latencies)

    @property
    def error_rate(self) -> float | None:
        """Error rate over the sliding window (0.0-1.0)."""
        if not self._outcomes:
            return None
        failures = sum(1 for o in self._outcomes if not o)
        return failures / len(self._outcomes)

    @property
    def circuit_state(self) -> CircuitBreakerState:
        """Current circuit breaker state, with auto-transition from OPEN to HALF_OPEN."""
        if (
            self._circuit_state == CircuitBreakerState.OPEN
            and self._open_since is not None
            and (time.monotonic() - self._open_since) >= self._recovery_timeout
        ):
            self._circuit_state = CircuitBreakerState.HALF_OPEN
        return self._circuit_state

    def trip(self) -> None:
        """Manually trip the circuit breaker to OPEN."""
        self._circuit_state = CircuitBreakerState.OPEN
        self._open_since = time.monotonic()

    def reset_circuit(self) -> None:
        """Manually reset the circuit breaker to CLOSED."""
        self._circuit_state = CircuitBreakerState.CLOSED
        self._open_since = None
        self._consecutive_failures = 0

    @property
    def status(self) -> ToolStatus:
        """Compute current health status from metrics."""
        err = self.error_rate
        lat = self.avg_latency_ms

        # No data yet
        if err is None:
            return ToolStatus.UNKNOWN

        # Hard failure
        if err >= self.error_rate_threshold:
            return ToolStatus.FAILED

        # Degraded
        if err > 0.1 or (lat is not None and lat > self.latency_threshold_ms):
            return ToolStatus.DEGRADED

        return ToolStatus.HEALTHY

    def report(self) -> HealthReport:
        """Generate a health report with priority score."""
        status = self.status
        
        # Priority: higher = more urgent attention needed
        priority = 0
        if status == ToolStatus.FAILED:
            priority = 100
        elif status == ToolStatus.DEGRADED:
            priority = 50
        elif status == ToolStatus.UNKNOWN:
            priority = 10

        return HealthReport(
            tool_name=self.tool_name,
            status=status,
            latency_ms=self.avg_latency_ms,
            error_rate=self.error_rate,
            priority=priority,
            detail=f"signals={self._custom_signals}" if self._custom_signals else "",
        )

    def weight_penalty(self) -> float:
        """Compute weight penalty for edge reweighting.
        
        Returns a multiplier: 1.0 = healthy, inf = failed.
        Circuit breaker state takes precedence.
        """
        cs = self.circuit_state
        if cs == CircuitBreakerState.OPEN:
            return float("inf")
        if cs == CircuitBreakerState.HALF_OPEN:
            return 10.0

        # CLOSED — use normal status-based calculation
        status = self.status
        if status == ToolStatus.FAILED:
            return float("inf")
        if status == ToolStatus.DEGRADED:
            err = self.error_rate or 0.0
            lat_penalty = 1.0
            if self.avg_latency_ms and self.avg_latency_ms > self.latency_threshold_ms:
                lat_penalty = self.avg_latency_ms / self.latency_threshold_ms
            return 2.0 + (err * 5.0) + lat_penalty
        return 1.0

    # --- Rate Limit ---

    def set_rate_limit(self, remaining: int, total: int) -> None:
        """Update rate limit information."""
        self._rate_limit_remaining = remaining
        self._rate_limit_total = total

    def rate_limit_factor(self) -> float:
        """Rate limit penalty factor.
        
        Returns 1.0 when plenty of quota, spikes near limit, inf at 0.
        """
        if self._rate_limit_remaining is None or self._rate_limit_total is None:
            return 1.0
        if self._rate_limit_total == 0:
            return 1.0
        if self._rate_limit_remaining <= 0:
            return float("inf")
        ratio = self._rate_limit_remaining / self._rate_limit_total
        if ratio > 0.5:
            return 1.0
        if ratio > 0.2:
            return 2.0
        if ratio > 0.1:
            return 10.0
        if ratio > 0.05:
            return 50.0
        return 100.0

    # --- Composite Weight ---

    def composite_weight(self, base_cost: float = 1.0) -> float:
        """W(tool) = base_cost × latency(t) × reliability(t) × rate_limit(t) × availability(t)"""
        # Availability based on circuit state
        cs = self.circuit_state
        if cs == CircuitBreakerState.OPEN:
            return float("inf")
        availability = 10.0 if cs == CircuitBreakerState.HALF_OPEN else 1.0

        # Latency factor: ratio of avg latency to threshold, clamped [0.5, 10.0]
        lat = self.avg_latency_ms
        if lat is not None and self.latency_threshold_ms > 0:
            latency_f = max(0.5, min(10.0, lat / self.latency_threshold_ms))
        else:
            latency_f = 1.0

        # Reliability factor: 1/(1-error_rate), clamped [1.0, 50.0]
        err = self.error_rate
        if err is not None and err < 1.0:
            reliability_f = max(1.0, min(50.0, 1.0 / (1.0 - err)))
        elif err is not None:
            reliability_f = 50.0
        else:
            reliability_f = 1.0

        # Rate limit factor
        rl_f = self.rate_limit_factor()
        if rl_f == float("inf"):
            return float("inf")

        return base_cost * latency_f * reliability_f * rl_f * availability

    # --- Priority Signal ---

    def priority_signal(self) -> "PrioritySignal":
        """Produce a priority signal based on current state."""
        from .priority import PrioritySignal

        status = self.status
        cs = self.circuit_state

        if status == ToolStatus.FAILED or cs == CircuitBreakerState.OPEN:
            score = 0.99
        elif status == ToolStatus.DEGRADED or cs == CircuitBreakerState.HALF_OPEN:
            score = 0.70
        elif status == ToolStatus.UNKNOWN:
            score = 0.30
        else:
            score = 0.10

        return PrioritySignal(
            name=f"{self.tool_name}_health",
            score=score,
            source=self.tool_name,
            detail=f"status={status.value}, circuit={cs.value}",
        )

    # --- Proactive Health Check ---

    def set_health_check(self, fn: Callable[[], bool]) -> None:
        """Set a health check function that returns True if healthy."""
        self._health_check = fn

    def run_health_check(self) -> bool:
        """Run the health check, updating metrics."""
        if self._health_check is None:
            return True
        try:
            result = self._health_check()
            if result:
                self.record_success(0.0)
            else:
                self.record_failure(0.0)
            return result
        except Exception:
            self.record_failure(0.0)
            return False


class MonitorRegistry:
    """Registry of health monitors, one per tool."""

    def __init__(self) -> None:
        self._monitors: dict[str, HealthMonitor] = {}

    def register(
        self,
        tool_name: str,
        latency_threshold_ms: float = 5000.0,
        error_rate_threshold: float = 0.5,
        window_size: int = 20,
    ) -> HealthMonitor:
        """Register a health monitor for a tool."""
        monitor = HealthMonitor(
            tool_name=tool_name,
            latency_threshold_ms=latency_threshold_ms,
            error_rate_threshold=error_rate_threshold,
            window_size=window_size,
        )
        self._monitors[tool_name] = monitor
        return monitor

    def get(self, tool_name: str) -> HealthMonitor | None:
        return self._monitors.get(tool_name)

    def all_reports(self) -> list[HealthReport]:
        """Get health reports from all monitors, sorted by priority (highest first)."""
        reports = [m.report() for m in self._monitors.values()]
        reports.sort(key=lambda r: r.priority, reverse=True)
        return reports

    def failed_tools(self) -> list[str]:
        """List tools currently in FAILED status."""
        return [
            name for name, m in self._monitors.items()
            if m.status == ToolStatus.FAILED
        ]

    def run_all_health_checks(self) -> dict[str, bool]:
        """Run health checks on all registered monitors."""
        return {name: m.run_health_check() for name, m in self._monitors.items()}
