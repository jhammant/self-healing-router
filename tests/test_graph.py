"""Tests for ToolGraph — Dijkstra routing, failure handling, recovery."""

import pytest
from self_healing_router import ToolGraph, ToolStatus


class TestToolGraph:
    def setup_method(self):
        self.g = ToolGraph()
        self.g.add_tool("A")
        self.g.add_tool("B")
        self.g.add_tool("C")
        self.g.add_tool("D")

    def test_add_tool(self):
        assert "A" in self.g.nodes
        assert self.g.get_tool("A").status == ToolStatus.HEALTHY

    def test_add_edge(self):
        self.g.add_edge("A", "B", weight=2.0)
        edges = self.g.get_edges("A")
        assert len(edges) == 1
        assert edges[0].target == "B"
        assert edges[0].current_weight == 2.0

    def test_add_edge_invalid_source(self):
        with pytest.raises(ValueError, match="Source tool"):
            self.g.add_edge("Z", "A")

    def test_add_edge_invalid_target(self):
        with pytest.raises(ValueError, match="Target tool"):
            self.g.add_edge("A", "Z")

    def test_shortest_path_simple(self):
        self.g.add_edge("A", "B", weight=1.0)
        self.g.add_edge("B", "C", weight=1.0)
        path, cost = self.g.shortest_path("A", "C")
        assert path == ["A", "B", "C"]
        assert cost == 2.0

    def test_shortest_path_picks_cheaper(self):
        self.g.add_edge("A", "B", weight=1.0)
        self.g.add_edge("B", "C", weight=1.0)
        self.g.add_edge("A", "C", weight=5.0)  # Direct but expensive
        path, cost = self.g.shortest_path("A", "C")
        assert path == ["A", "B", "C"]
        assert cost == 2.0

    def test_shortest_path_no_path(self):
        # No edges
        path, cost = self.g.shortest_path("A", "D")
        assert path == []
        assert cost == float("inf")

    def test_shortest_path_nonexistent_node(self):
        path, cost = self.g.shortest_path("A", "Z")
        assert path == []
        assert cost == float("inf")

    def test_fail_tool_makes_path_infinite(self):
        self.g.add_edge("A", "B", weight=1.0)
        self.g.add_edge("B", "C", weight=1.0)
        self.g.fail_tool("B")
        path, cost = self.g.shortest_path("A", "C")
        assert path == []
        assert cost == float("inf")

    def test_fail_tool_reroutes(self):
        self.g.add_edge("A", "B", weight=1.0)
        self.g.add_edge("B", "C", weight=1.0)
        self.g.add_edge("A", "D", weight=2.0)
        self.g.add_edge("D", "C", weight=1.0)
        
        # Fail B → should route through D
        self.g.fail_tool("B")
        path, cost = self.g.shortest_path("A", "C")
        assert path == ["A", "D", "C"]
        assert cost == 3.0

    def test_recover_tool(self):
        self.g.add_edge("A", "B", weight=1.0)
        self.g.add_edge("B", "C", weight=1.0)
        self.g.fail_tool("B")
        assert self.g.get_tool("B").status == ToolStatus.FAILED
        
        self.g.recover_tool("B")
        assert self.g.get_tool("B").status == ToolStatus.HEALTHY
        path, cost = self.g.shortest_path("A", "C")
        assert path == ["A", "B", "C"]

    def test_update_edge_weight(self):
        self.g.add_edge("A", "B", weight=1.0)
        self.g.update_edge_weight("A", "B", 10.0)
        edges = self.g.get_edges("A")
        assert edges[0].current_weight == 10.0

    def test_update_edge_weight_invalid(self):
        with pytest.raises(ValueError, match="No edge"):
            self.g.update_edge_weight("A", "B", 5.0)

    def test_all_paths(self):
        self.g.add_edge("A", "B", weight=1.0)
        self.g.add_edge("B", "C", weight=1.0)
        self.g.add_edge("A", "C", weight=5.0)
        paths = self.g.all_paths("A", "C")
        assert len(paths) == 2
        # Sorted by cost — cheapest first
        assert paths[0][0] == ["A", "B", "C"]
        assert paths[1][0] == ["A", "C"]

    def test_to_ascii(self):
        self.g.add_edge("A", "B", weight=1.0)
        viz = self.g.to_ascii()
        assert "A" in viz
        assert "B" in viz
