"""Tests for priority competition system."""

from self_healing_router import (
    PrioritySignal,
    PriorityArbiter,
    HealthMonitor,
    ToolStatus,
    CircuitBreakerState,
)


class TestPriorityArbiter:
    """PriorityArbiter signal resolution."""

    def test_empty_returns_none(self):
        arbiter = PriorityArbiter()
        assert arbiter.resolve() is None

    def test_single_signal(self):
        arbiter = PriorityArbiter()
        s = PrioritySignal(name="test", score=0.5, source="tool_a", detail="ok")
        arbiter.add_signal(s)
        assert arbiter.resolve() is s

    def test_resolves_highest_score(self):
        arbiter = PriorityArbiter()
        low = PrioritySignal(name="low", score=0.1, source="a", detail="")
        high = PrioritySignal(name="high", score=0.9, source="b", detail="")
        mid = PrioritySignal(name="mid", score=0.5, source="c", detail="")
        arbiter.add_signal(low)
        arbiter.add_signal(high)
        arbiter.add_signal(mid)
        assert arbiter.resolve() is high

    def test_clear(self):
        arbiter = PriorityArbiter()
        arbiter.add_signal(PrioritySignal(name="x", score=0.5, source="a", detail=""))
        arbiter.clear()
        assert arbiter.resolve() is None


class TestPrioritySignalFromMonitor:
    """Priority signals produced by HealthMonitor."""

    def test_healthy_monitor_low_score(self):
        m = HealthMonitor("tool")
        for _ in range(5):
            m.record_success(100.0)
        sig = m.priority_signal()
        assert sig.score == 0.10
        assert sig.source == "tool"

    def test_failed_monitor_high_score(self):
        m = HealthMonitor("tool", error_rate_threshold=0.5)
        for _ in range(10):
            m.record_failure(100.0)
        sig = m.priority_signal()
        assert sig.score == 0.99

    def test_degraded_monitor_medium_score(self):
        m = HealthMonitor("tool", latency_threshold_ms=100.0)
        for _ in range(5):
            m.record_success(200.0)  # high latency -> degraded
        assert m.status == ToolStatus.DEGRADED
        sig = m.priority_signal()
        assert sig.score == 0.70

    def test_unknown_monitor_score(self):
        m = HealthMonitor("tool")
        # No data recorded
        sig = m.priority_signal()
        assert sig.score == 0.30

    def test_open_circuit_high_score(self):
        m = HealthMonitor("tool")
        m.trip()
        sig = m.priority_signal()
        assert sig.score == 0.99

    def test_half_open_circuit_medium_score(self):
        m = HealthMonitor("tool")
        for _ in range(5):
            m.record_success(100.0)
        m._circuit_state = CircuitBreakerState.HALF_OPEN
        sig = m.priority_signal()
        assert sig.score == 0.70
