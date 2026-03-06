"""Tests for binary observability — ExecutionOutcome and execution_log."""

from self_healing_router import SelfHealingRouter, ExecutionOutcome
from self_healing_router.types import EscalationResult


class TestExecutionOutcome:
    """ExecutionOutcome set correctly on RouteResult."""

    def _make_router_linear(self):
        """Helper: A -> B -> C linear chain."""
        router = SelfHealingRouter()
        router.add_tool("A", handler=lambda d: d)
        router.add_tool("B", handler=lambda d: d)
        router.add_tool("C", handler=lambda d: d)
        router.add_edge("A", "B")
        router.add_edge("B", "C")
        return router

    def test_success_on_happy_path(self):
        router = self._make_router_linear()
        result = router.route("A", "C")
        assert result.success is True
        assert result.outcome == ExecutionOutcome.SUCCESS

    def test_escalated_when_llm_called(self):
        call_count = 0

        def escalation_cb(failed_tool, paths, context):
            nonlocal call_count
            call_count += 1
            return EscalationResult(action="abort", detail="no path")

        router = SelfHealingRouter(escalation_callback=escalation_cb)
        router.add_tool("A")
        router.add_tool("B")
        # No edge — will escalate immediately
        result = router.route("A", "B")
        assert result.success is False
        assert result.outcome == ExecutionOutcome.ESCALATED
        assert call_count == 1

    def test_max_reroutes_outcome(self):
        """When all tools keep failing and reroute limit is hit."""
        fail_count = 0

        def always_fail(data):
            nonlocal fail_count
            fail_count += 1
            raise RuntimeError("boom")

        router = SelfHealingRouter(max_reroutes=2)
        router.add_tool("A", handler=lambda d: d)
        router.add_tool("B1", handler=always_fail)
        router.add_tool("B2", handler=always_fail)
        router.add_tool("B3", handler=always_fail)
        router.add_tool("C", handler=lambda d: d)
        router.add_edge("A", "B1")
        router.add_edge("A", "B2")
        router.add_edge("A", "B3")
        router.add_edge("B1", "C")
        router.add_edge("B2", "C")
        router.add_edge("B3", "C")
        result = router.route("A", "C")
        assert result.success is False
        assert result.outcome == ExecutionOutcome.MAX_REROUTES


class TestExecutionLog:
    """execution_log populated with step details."""

    def test_log_populated_on_success(self):
        router = SelfHealingRouter()
        router.add_tool("A", handler=lambda d: d)
        router.add_tool("B", handler=lambda d: d)
        router.add_edge("A", "B")
        result = router.route("A", "B")
        assert result.success is True
        assert len(result.execution_log) >= 2
        for entry in result.execution_log:
            assert "tool" in entry
            assert "success" in entry

    def test_log_records_failure(self):
        def fail(data):
            raise RuntimeError("oops")

        router = SelfHealingRouter(max_reroutes=0)
        router.add_tool("A", handler=lambda d: d)
        router.add_tool("B", handler=fail)
        router.add_edge("A", "B")
        result = router.route("A", "B")
        # Should have A success and B failure logged
        successes = [e for e in result.execution_log if e.get("step") == "execute" and e["success"]]
        failures = [e for e in result.execution_log if e.get("step") == "execute" and not e["success"]]
        assert len(successes) >= 1
        assert len(failures) >= 1

    def test_log_includes_latency(self):
        router = SelfHealingRouter()
        router.add_tool("A", handler=lambda d: d)
        router.add_tool("B", handler=lambda d: d)
        router.add_edge("A", "B")
        result = router.route("A", "B")
        for entry in result.execution_log:
            if entry.get("step") == "execute":
                assert "latency_ms" in entry
                assert entry["latency_ms"] >= 0

    def test_escalation_logged(self):
        def escalation_cb(failed_tool, paths, context):
            return EscalationResult(action="abort", detail="giving up")

        router = SelfHealingRouter(escalation_callback=escalation_cb)
        router.add_tool("A")
        router.add_tool("B")
        result = router.route("A", "B")
        esc_entries = [e for e in result.execution_log if e.get("step") == "escalation"]
        assert len(esc_entries) >= 1
