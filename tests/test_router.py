"""Tests for SelfHealingRouter — the main integration tests."""

import pytest
from self_healing_router import SelfHealingRouter, EscalationResult


class TestSelfHealingRouter:
    """Core routing tests across all three topologies."""

    # --- Linear Pipeline ---

    def test_linear_success(self):
        router = SelfHealingRouter()
        router.add_tool("A", handler=lambda d: {**d, "a": True})
        router.add_tool("B", handler=lambda d: {**d, "b": True})
        router.add_tool("C", handler=lambda d: {**d, "c": True})
        router.add_edge("A", "B")
        router.add_edge("B", "C")

        result = router.route("A", "C")
        assert result.success
        assert result.path == ["A", "B", "C"]
        assert result.llm_calls == 0
        assert result.reroutes == 0

    def test_linear_failure_reroutes(self):
        router = SelfHealingRouter()
        router.add_tool("A", handler=lambda d: d)
        router.add_tool("B", handler=lambda d: (_ for _ in ()).throw(RuntimeError("down")))
        router.add_tool("C", handler=lambda d: d)
        router.add_tool("B2", handler=lambda d: {**d, "backup": True})
        
        router.add_edge("A", "B", weight=1.0)
        router.add_edge("B", "C", weight=1.0)
        router.add_edge("A", "B2", weight=3.0)
        router.add_edge("B2", "C", weight=1.0)

        result = router.route("A", "C")
        assert result.success
        assert result.reroutes == 1
        assert result.llm_calls == 0  # No LLM needed!
        assert "B2" in [r for r in result.outputs.keys()]

    # --- Dependency DAG ---

    def test_dag_primary_path(self):
        router = SelfHealingRouter()
        router.add_tool("src", handler=lambda d: {**d, "src": True})
        router.add_tool("enrich", handler=lambda d: {**d, "enriched": True})
        router.add_tool("validate", handler=lambda d: {**d, "valid": True})
        router.add_tool("merge", handler=lambda d: {**d, "merged": True})
        
        router.add_edge("src", "enrich", weight=1.0)
        router.add_edge("enrich", "merge", weight=1.0)
        router.add_edge("src", "validate", weight=2.0)
        router.add_edge("validate", "merge", weight=1.0)

        result = router.route("src", "merge")
        assert result.success
        assert result.path == ["src", "enrich", "merge"]  # Cheapest
        assert result.llm_calls == 0

    def test_dag_reroute_on_failure(self):
        router = SelfHealingRouter()
        router.add_tool("src", handler=lambda d: d)
        router.add_tool("enrich", handler=lambda d: (_ for _ in ()).throw(RuntimeError("down")))
        router.add_tool("validate", handler=lambda d: {**d, "valid": True})
        router.add_tool("merge", handler=lambda d: {**d, "merged": True})
        
        router.add_edge("src", "enrich", weight=1.0)
        router.add_edge("enrich", "merge", weight=1.0)
        router.add_edge("src", "validate", weight=2.0)
        router.add_edge("validate", "merge", weight=1.0)

        result = router.route("src", "merge")
        assert result.success
        assert result.reroutes == 1
        assert result.llm_calls == 0
        assert "validate" in result.outputs

    # --- Parallel Fan-Out ---

    def test_fanout_picks_cheapest(self):
        router = SelfHealingRouter()
        router.add_tool("dispatch", handler=lambda d: d)
        router.add_tool("w1", handler=lambda d: {**d, "w1": True})
        router.add_tool("w2", handler=lambda d: {**d, "w2": True})
        router.add_tool("agg", handler=lambda d: {**d, "done": True})
        
        router.add_edge("dispatch", "w1", weight=1.0)
        router.add_edge("dispatch", "w2", weight=3.0)
        router.add_edge("w1", "agg", weight=1.0)
        router.add_edge("w2", "agg", weight=1.0)

        result = router.route("dispatch", "agg")
        assert result.success
        assert "w1" in result.path  # Cheapest worker
        assert result.llm_calls == 0

    def test_fanout_reroutes_to_backup(self):
        router = SelfHealingRouter()
        router.add_tool("dispatch", handler=lambda d: d)
        router.add_tool("w1", handler=lambda d: (_ for _ in ()).throw(RuntimeError("down")))
        router.add_tool("w2", handler=lambda d: {**d, "w2": True})
        router.add_tool("backup", handler=lambda d: {**d, "backup": True})
        router.add_tool("agg", handler=lambda d: {**d, "done": True})
        
        router.add_edge("dispatch", "w1", weight=1.0)
        router.add_edge("dispatch", "w2", weight=2.0)
        router.add_edge("dispatch", "backup", weight=5.0)
        router.add_edge("w1", "agg", weight=1.0)
        router.add_edge("w2", "agg", weight=1.0)
        router.add_edge("backup", "agg", weight=1.0)

        result = router.route("dispatch", "agg")
        assert result.success
        assert result.reroutes >= 1
        assert result.llm_calls == 0
        assert "w1" not in result.outputs  # Failed

    # --- LLM Escalation ---

    def test_escalation_when_no_path(self):
        escalation_called = []
        
        def mock_escalation(failed, paths, ctx):
            escalation_called.append(True)
            return EscalationResult(action="abort", detail="No alternatives")

        router = SelfHealingRouter(escalation_callback=mock_escalation)
        router.add_tool("A", handler=lambda d: d)
        router.add_tool("B", handler=lambda d: (_ for _ in ()).throw(RuntimeError("down")))
        router.add_tool("C", handler=lambda d: d)
        router.add_edge("A", "B")
        router.add_edge("B", "C")
        # No alternative path!

        result = router.route("A", "C")
        assert not result.success
        assert result.llm_calls == 1
        assert len(escalation_called) == 1

    def test_no_escalation_when_reroute_works(self):
        escalation_called = []
        
        def mock_escalation(failed, paths, ctx):
            escalation_called.append(True)
            return EscalationResult(action="abort")

        router = SelfHealingRouter(escalation_callback=mock_escalation)
        router.add_tool("A", handler=lambda d: d)
        router.add_tool("B", handler=lambda d: (_ for _ in ()).throw(RuntimeError("down")))
        router.add_tool("B2", handler=lambda d: {**d, "backup": True})
        router.add_tool("C", handler=lambda d: d)
        router.add_edge("A", "B", weight=1.0)
        router.add_edge("B", "C", weight=1.0)
        router.add_edge("A", "B2", weight=3.0)
        router.add_edge("B2", "C", weight=1.0)

        result = router.route("A", "C")
        assert result.success
        assert result.llm_calls == 0
        assert len(escalation_called) == 0  # Never called!

    # --- Zero LLM calls ---

    def test_zero_llm_calls_normal_operation(self):
        """The whole point: normal routing uses ZERO LLM calls."""
        router = SelfHealingRouter()
        for name in ["t1", "t2", "t3", "t4", "t5"]:
            router.add_tool(name, handler=lambda d, n=name: {**d, n: True})
        router.add_edge("t1", "t2")
        router.add_edge("t2", "t3")
        router.add_edge("t3", "t4")
        router.add_edge("t4", "t5")

        result = router.route("t1", "t5")
        assert result.success
        assert result.llm_calls == 0
        assert result.reroutes == 0

    def test_zero_llm_calls_with_reroute(self):
        """Even with failures and reroutes: ZERO LLM calls if alt path exists."""
        router = SelfHealingRouter()
        router.add_tool("A", handler=lambda d: d)
        router.add_tool("B", handler=lambda d: (_ for _ in ()).throw(RuntimeError("fail")))
        router.add_tool("C", handler=lambda d: (_ for _ in ()).throw(RuntimeError("fail")))
        router.add_tool("D", handler=lambda d: {**d, "d": True})
        router.add_tool("E", handler=lambda d: {**d, "e": True})

        router.add_edge("A", "B", weight=1.0)
        router.add_edge("A", "C", weight=2.0)
        router.add_edge("A", "D", weight=5.0)
        router.add_edge("B", "E", weight=1.0)
        router.add_edge("C", "E", weight=1.0)
        router.add_edge("D", "E", weight=1.0)

        result = router.route("A", "E")
        assert result.success
        assert result.llm_calls == 0  # Still zero!
        assert result.reroutes == 2  # Failed B and C, landed on D

    # --- Health Report ---

    def test_health_report(self):
        router = SelfHealingRouter()
        router.add_tool("A", handler=lambda d: d)
        report = router.health_report()
        assert "A" in report

    def test_visualize(self):
        router = SelfHealingRouter()
        router.add_tool("A")
        router.add_tool("B")
        router.add_edge("A", "B")
        viz = router.visualize()
        assert "A" in viz
        assert "B" in viz

    def test_reset(self):
        router = SelfHealingRouter()
        router.add_tool("A", handler=lambda d: d)
        router.add_tool("B", handler=lambda d: (_ for _ in ()).throw(RuntimeError("fail")))
        router.add_edge("A", "B")
        
        router.route("A", "B")  # B will fail
        router.reset()
        
        # After reset, B should be recoverable
        assert router.graph.get_tool("B").status != "FAILED"
