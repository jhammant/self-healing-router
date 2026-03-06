"""Tests for EscalationHandler."""

from self_healing_router import EscalationHandler, EscalationResult
from self_healing_router.escalation import default_escalation


class TestEscalationHandler:
    def test_no_callback_aborts(self):
        handler = EscalationHandler(callback=None)
        result = handler.escalate("tool_x", [["A", "B"]])
        assert result.action == "abort"
        assert handler.escalation_count == 1

    def test_callback_called(self):
        called_with = {}
        
        def my_callback(tool, paths, ctx):
            called_with["tool"] = tool
            called_with["paths"] = paths
            return EscalationResult(action="retry")

        handler = EscalationHandler(callback=my_callback)
        result = handler.escalate("tool_x", [["A", "B"]], {"extra": True})
        assert result.action == "retry"
        assert called_with["tool"] == "tool_x"

    def test_max_escalations(self):
        handler = EscalationHandler(
            callback=lambda t, p, c: EscalationResult(action="retry"),
            max_escalations=2,
        )
        # First two succeed
        r1 = handler.escalate("t", [[]])
        assert r1.action == "retry"
        r2 = handler.escalate("t", [[]])
        assert r2.action == "retry"
        # Third exceeds max → forced abort
        r3 = handler.escalate("t", [[]])
        assert r3.action == "abort"
        assert "exceeded" in r3.detail.lower()

    def test_history_tracked(self):
        handler = EscalationHandler(callback=None)
        handler.escalate("tool_a", [["A"]])
        handler.escalate("tool_b", [["B"]])
        assert len(handler.history) == 2
        assert handler.history[0]["failed_tool"] == "tool_a"

    def test_reset(self):
        handler = EscalationHandler(callback=None)
        handler.escalate("t", [[]])
        assert handler.escalation_count == 1
        handler.reset()
        assert handler.escalation_count == 0
        assert len(handler.history) == 0

    def test_default_escalation(self):
        result = default_escalation("tool_x", [["A", "B"]], {})
        assert result.action == "abort"
        assert "tool_x" in result.detail
