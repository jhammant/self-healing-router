"""Health monitors for tool nodes — latency, error rates, custom signals."""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Callable

from .types import ToolStatus, HealthReport


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

    def record_success(self, latency_ms: float) -> None:
        """Record a successful tool invocation."""
        self._latencies.append(latency_ms)
        self._outcomes.append(True)

    def record_failure(self, latency_ms: float = 0.0) -> None:
        """Record a failed tool invocation."""
        if latency_ms > 0:
            self._latencies.append(latency_ms)
        self._outcomes.append(False)

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
        """
        status = self.status
        if status == ToolStatus.FAILED:
            return float("inf")
        if status == ToolStatus.DEGRADED:
            # Scale penalty with error rate
            err = self.error_rate or 0.0
            lat_penalty = 1.0
            if self.avg_latency_ms and self.avg_latency_ms > self.latency_threshold_ms:
                lat_penalty = self.avg_latency_ms / self.latency_threshold_ms
            return 2.0 + (err * 5.0) + lat_penalty
        return 1.0


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
