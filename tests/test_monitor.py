"""Tests for HealthMonitor and MonitorRegistry."""

from self_healing_router import HealthMonitor, ToolStatus
from self_healing_router.monitor import MonitorRegistry


class TestHealthMonitor:
    def test_initial_status_unknown(self):
        m = HealthMonitor("test_tool")
        assert m.status == ToolStatus.UNKNOWN

    def test_healthy_after_successes(self):
        m = HealthMonitor("test_tool")
        for _ in range(5):
            m.record_success(100.0)
        assert m.status == ToolStatus.HEALTHY

    def test_failed_after_errors(self):
        m = HealthMonitor("test_tool", error_rate_threshold=0.5)
        for _ in range(10):
            m.record_failure()
        assert m.status == ToolStatus.FAILED

    def test_degraded_high_latency(self):
        m = HealthMonitor("test_tool", latency_threshold_ms=100.0)
        for _ in range(5):
            m.record_success(200.0)  # Over threshold
        assert m.status == ToolStatus.DEGRADED

    def test_degraded_some_errors(self):
        m = HealthMonitor("test_tool", error_rate_threshold=0.5)
        # 20% error rate — above 10% threshold for degraded
        for _ in range(8):
            m.record_success(50.0)
        for _ in range(2):
            m.record_failure()
        assert m.status == ToolStatus.DEGRADED

    def test_avg_latency(self):
        m = HealthMonitor("test_tool")
        m.record_success(100.0)
        m.record_success(200.0)
        assert m.avg_latency_ms == 150.0

    def test_error_rate(self):
        m = HealthMonitor("test_tool")
        m.record_success(50.0)
        m.record_failure()
        assert m.error_rate == 0.5

    def test_no_data_returns_none(self):
        m = HealthMonitor("test_tool")
        assert m.avg_latency_ms is None
        assert m.error_rate is None

    def test_weight_penalty_healthy(self):
        m = HealthMonitor("test_tool")
        for _ in range(5):
            m.record_success(50.0)
        assert m.weight_penalty() == 1.0

    def test_weight_penalty_failed(self):
        m = HealthMonitor("test_tool", error_rate_threshold=0.5)
        for _ in range(10):
            m.record_failure()
        assert m.weight_penalty() == float("inf")

    def test_weight_penalty_degraded(self):
        m = HealthMonitor("test_tool", error_rate_threshold=0.5)
        for _ in range(8):
            m.record_success(50.0)
        for _ in range(2):
            m.record_failure()
        penalty = m.weight_penalty()
        assert penalty > 1.0
        assert penalty < float("inf")

    def test_custom_signal(self):
        m = HealthMonitor("test_tool")
        m.set_signal("queue_depth", 42.0)
        report = m.report()
        assert "queue_depth" in report.detail

    def test_report_priority(self):
        m = HealthMonitor("test_tool", error_rate_threshold=0.5)
        for _ in range(10):
            m.record_failure()
        report = m.report()
        assert report.priority == 100

    def test_sliding_window(self):
        m = HealthMonitor("test_tool", window_size=5, error_rate_threshold=0.5)
        # Fill with failures
        for _ in range(5):
            m.record_failure()
        assert m.status == ToolStatus.FAILED
        # Now push successes — failures slide out
        for _ in range(5):
            m.record_success(50.0)
        assert m.status == ToolStatus.HEALTHY


class TestMonitorRegistry:
    def test_register_and_get(self):
        reg = MonitorRegistry()
        reg.register("tool_a")
        assert reg.get("tool_a") is not None
        assert reg.get("tool_z") is None

    def test_all_reports(self):
        reg = MonitorRegistry()
        reg.register("tool_a")
        reg.register("tool_b")
        reports = reg.all_reports()
        assert len(reports) == 2

    def test_failed_tools(self):
        reg = MonitorRegistry()
        ma = reg.register("tool_a", error_rate_threshold=0.5)
        reg.register("tool_b")
        for _ in range(10):
            ma.record_failure()
        assert "tool_a" in reg.failed_tools()
        assert "tool_b" not in reg.failed_tools()
