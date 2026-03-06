"""Tool graph with weighted directed edges and Dijkstra routing."""

from __future__ import annotations

import heapq
from typing import Any

from .types import ToolNode, Edge, ToolStatus


class ToolGraph:
    """A weighted directed graph of tools.
    
    Tools are nodes, dependencies are edges. Edge weights represent
    cost (latency, error risk, etc). Failed tools get infinite weight
    so Dijkstra naturally routes around them.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, ToolNode] = {}
        self._edges: dict[str, list[Edge]] = {}  # source -> [edges]
        self._reverse_edges: dict[str, list[Edge]] = {}  # target -> [edges]

    def add_tool(
        self,
        name: str,
        handler: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ToolNode:
        """Add a tool node to the graph."""
        node = ToolNode(name=name, handler=handler, metadata=metadata or {})
        self._nodes[name] = node
        if name not in self._edges:
            self._edges[name] = []
        if name not in self._reverse_edges:
            self._reverse_edges[name] = []
        return node

    def add_edge(
        self,
        source: str,
        target: str,
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> Edge:
        """Add a weighted directed edge between tools."""
        if source not in self._nodes:
            raise ValueError(f"Source tool '{source}' not in graph")
        if target not in self._nodes:
            raise ValueError(f"Target tool '{target}' not in graph")

        edge = Edge(
            source=source,
            target=target,
            base_weight=weight,
            current_weight=weight,
            metadata=metadata or {},
        )
        self._edges[source].append(edge)
        self._reverse_edges[target].append(edge)
        return edge

    def get_tool(self, name: str) -> ToolNode | None:
        """Get a tool node by name."""
        return self._nodes.get(name)

    def get_edges(self, source: str) -> list[Edge]:
        """Get all outgoing edges from a tool."""
        return self._edges.get(source, [])

    def fail_tool(self, name: str) -> None:
        """Mark a tool as failed — all edges to/from it get infinite weight."""
        if name in self._nodes:
            self._nodes[name].status = ToolStatus.FAILED
        for edge in self._edges.get(name, []):
            edge.fail()
        for edge in self._reverse_edges.get(name, []):
            edge.fail()

    def recover_tool(self, name: str) -> None:
        """Recover a tool — reset all its edges to base weight."""
        if name in self._nodes:
            self._nodes[name].status = ToolStatus.HEALTHY
        for edge in self._edges.get(name, []):
            edge.recover()
        for edge in self._reverse_edges.get(name, []):
            edge.recover()

    def update_edge_weight(self, source: str, target: str, weight: float) -> None:
        """Update the weight of a specific edge."""
        for edge in self._edges.get(source, []):
            if edge.target == target:
                edge.current_weight = weight
                return
        raise ValueError(f"No edge from '{source}' to '{target}'")

    def shortest_path(self, start: str, end: str) -> tuple[list[str], float]:
        """Find shortest path using Dijkstra's algorithm.
        
        Returns (path, total_weight). If no path exists, returns ([], inf).
        """
        if start not in self._nodes or end not in self._nodes:
            return [], float("inf")

        # Dijkstra's
        dist: dict[str, float] = {n: float("inf") for n in self._nodes}
        prev: dict[str, str | None] = {n: None for n in self._nodes}
        dist[start] = 0.0
        
        # Priority queue: (distance, node_name)
        pq: list[tuple[float, str]] = [(0.0, start)]

        while pq:
            d, u = heapq.heappop(pq)
            
            if d > dist[u]:
                continue
                
            if u == end:
                break

            for edge in self._edges.get(u, []):
                v = edge.target
                new_dist = dist[u] + edge.current_weight
                if new_dist < dist[v]:
                    dist[v] = new_dist
                    prev[v] = u
                    heapq.heappush(pq, (new_dist, v))

        # Reconstruct path
        if dist[end] == float("inf"):
            return [], float("inf")

        path = []
        node: str | None = end
        while node is not None:
            path.append(node)
            node = prev[node]
        path.reverse()

        return path, dist[end]

    def all_paths(self, start: str, end: str, max_paths: int = 10) -> list[tuple[list[str], float]]:
        """Find multiple paths (for debugging/visualization). Uses DFS with pruning."""
        if start not in self._nodes or end not in self._nodes:
            return []

        results: list[tuple[list[str], float]] = []
        
        def _dfs(current: str, target: str, visited: set[str], path: list[str], cost: float) -> None:
            if len(results) >= max_paths:
                return
            if current == target:
                results.append((list(path), cost))
                return
            for edge in self._edges.get(current, []):
                if edge.target not in visited and not edge.is_failed:
                    visited.add(edge.target)
                    path.append(edge.target)
                    _dfs(edge.target, target, visited, path, cost + edge.current_weight)
                    path.pop()
                    visited.discard(edge.target)

        _dfs(start, end, {start}, [start], 0.0)
        results.sort(key=lambda x: x[1])
        return results

    @property
    def nodes(self) -> dict[str, ToolNode]:
        return dict(self._nodes)

    @property
    def edges(self) -> list[Edge]:
        all_edges = []
        for edge_list in self._edges.values():
            all_edges.extend(edge_list)
        return all_edges

    def to_ascii(self) -> str:
        """Simple ASCII visualization of the graph."""
        lines = ["Tool Graph:", ""]
        for name, node in self._nodes.items():
            status = "✓" if node.status == ToolStatus.HEALTHY else "✗" if node.status == ToolStatus.FAILED else "~"
            edges_out = self._edges.get(name, [])
            if edges_out:
                for edge in edges_out:
                    w = "∞" if edge.is_failed else f"{edge.current_weight:.1f}"
                    lines.append(f"  [{status}] {name} --({w})--> {edge.target}")
            else:
                lines.append(f"  [{status}] {name} (sink)")
        return "\n".join(lines)
