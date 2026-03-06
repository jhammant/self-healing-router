# Self-Healing Router

**93% fewer LLM calls than ReAct. Same correctness. Automatic recovery from tool failures.**

A graph-based tool routing library for LLM agents. Instead of asking your LLM "what tool should I use next?" at every step, Self-Healing Router models your tools as a weighted directed graph and uses Dijkstra's algorithm for deterministic routing. When a tool fails, edge weights update to infinity and the path recomputes automatically. The LLM is only called as a last resort — when no feasible path exists.

Based on the paper: [Graph-Based Self-Healing Tool Routing for Cost-Efficient LLM Agents](https://arxiv.org/abs/2603.01548)

## Why?

Every ReAct-style agent makes an LLM call at **every single step** to decide what tool to use next. For a 10-step pipeline, that's 10 LLM calls minimum — and if tools fail, you're looking at 15-20+ calls for retries and replanning.

Self-Healing Router replaces that with:
- **Dijkstra's algorithm** for tool routing (zero LLM calls)
- **Automatic rerouting** on failure (still zero LLM calls)
- **LLM escalation** only when ALL paths are exhausted (rare)

```
┌──────────────────────────────────────────────────────┐
│                    COST COMPARISON                    │
│                                                      │
│  ReAct (10-step pipeline):                           │
│    Normal:    10 LLM calls                           │
│    1 failure: 13 LLM calls                           │
│    2 failures: 18 LLM calls                          │
│                                                      │
│  Self-Healing Router:                                │
│    Normal:    0 LLM calls  ← deterministic routing   │
│    1 failure: 0 LLM calls  ← auto-reroute            │
│    2 failures: 0 LLM calls ← still auto-reroute      │
│    All fail:  1 LLM call   ← escalation (last resort)│
│                                                      │
│  Savings: 93% fewer LLM calls                        │
└──────────────────────────────────────────────────────┘
```

## Quick Start

```bash
pip install self-healing-router
```

```python
from self_healing_router import SelfHealingRouter

router = SelfHealingRouter()

# Define tools
router.add_tool("fetch", handler=fetch_data)
router.add_tool("parse", handler=parse_data)
router.add_tool("transform", handler=backup_parser)  # Fallback
router.add_tool("store", handler=store_data)

# Define routing graph
router.add_edge("fetch", "parse", weight=1.0)        # Primary path
router.add_edge("parse", "store", weight=1.0)
router.add_edge("fetch", "transform", weight=3.0)     # Backup path
router.add_edge("transform", "store", weight=1.0)

# Route — zero LLM calls
result = router.route("fetch", "store", input_data={"url": "..."})

print(f"Success: {result.success}")
print(f"Path taken: {' → '.join(result.path)}")
print(f"LLM calls: {result.llm_calls}")  # 0 in normal operation
print(f"Reroutes: {result.reroutes}")     # Auto-healed failures
```

If `parse` fails at runtime, the router automatically:
1. Sets parse's edge weight to ∞
2. Recomputes shortest path via Dijkstra
3. Routes through `transform` instead
4. **Zero LLM calls** — all deterministic

## Architecture

```
                     ┌─────────────┐
                     │  Tool Graph  │
                     │  (Dijkstra)  │
                     └──────┬──────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │  Tool A  │ │  Tool B  │ │  Tool C  │
        └────┬─────┘ └────┬─────┘ └────┬─────┘
             │             │             │
        ┌────▼─────┐ ┌────▼─────┐ ┌────▼─────┐
        │ Monitor A│ │ Monitor B│ │ Monitor C│
        │ (health) │ │ (health) │ │ (health) │
        └────┬─────┘ └────┬─────┘ └────┬─────┘
             │             │             │
             └─────────────┼─────────────┘
                           ▼
                  ┌─────────────────┐
                  │  Edge Weights   │
                  │  (auto-update)  │
                  └────────┬────────┘
                           │
                    ┌──────▼──────┐
                    │ Reroute or  │
                    │ Escalate    │──→ LLM (last resort)
                    └─────────────┘
```

**Key components:**

- **`ToolGraph`** — Weighted directed graph with Dijkstra routing
- **`HealthMonitor`** — Tracks latency, error rates, custom signals per tool
- **`SelfHealingRouter`** — Orchestrates routing + auto-recovery
- **`EscalationHandler`** — LLM callback, only when no path exists

## Supported Topologies

### Linear Pipeline
```
fetch → parse → store
          ↓ (fails)
fetch → transform → store  (auto-rerouted)
```

### Dependency DAG
```
api_a → enrich → merge → output
                   ↑
api_b → validate ──┘
                   ↑
api_c ─────────────┘  (fallback)
```

### Parallel Fan-Out
```
dispatch → worker_1 → aggregate → report
         → worker_2 →
         → worker_3 →
         → backup_1 →  (standby)
```

## API Reference

### `SelfHealingRouter`

```python
router = SelfHealingRouter(
    escalation_callback=my_llm_fn,  # Optional: called when no path exists
    max_reroutes=5,                  # Max reroute attempts before giving up
    max_escalations=3,               # Max LLM escalation calls
)
```

| Method | Description |
|--------|-------------|
| `add_tool(name, handler, metadata)` | Add a tool node with health monitoring |
| `add_edge(source, target, weight)` | Add a weighted directed edge |
| `route(start, end, input_data)` | Route and execute, returns `RouteResult` |
| `health_report()` | Human-readable health status |
| `visualize()` | ASCII graph visualization |
| `reset()` | Recover all tools, reset escalation counter |

### `RouteResult`

```python
@dataclass
class RouteResult:
    path: list[str]           # Tools executed (in order)
    total_weight: float       # Path cost
    success: bool             # Did we reach the goal?
    outputs: dict[str, Any]   # Output from each tool
    reroutes: int             # Number of automatic reroutes
    llm_calls: int            # Number of LLM escalation calls
    errors: list[str]         # Error messages from failed tools
```

### LLM Escalation

```python
from self_healing_router import EscalationResult

def my_escalation(failed_tool, attempted_paths, context):
    # Call your LLM here
    response = openai.chat.completions.create(...)
    return EscalationResult(
        action="abort",  # or "retry", "skip", "alternative"
        detail="LLM's reasoning",
        alternative_path=["A", "D", "E"],  # Optional
    )

router = SelfHealingRouter(escalation_callback=my_escalation)
```

## Examples

```bash
# Run examples
python examples/linear_pipeline.py
python examples/dependency_dag.py
python examples/parallel_fanout.py
python examples/with_llm_escalation.py
```

## Development

```bash
git clone https://github.com/jhammant/self-healing-router
cd self-healing-router
pip install -e ".[dev]"
pytest
```

## How It Compares

| Feature | ReAct | Self-Healing Router |
|---------|-------|-------------------|
| LLM calls per step | 1 | 0 |
| Failure recovery | LLM replans | Dijkstra reroutes |
| LLM calls on failure | N (per retry) | 0 (graph handles it) |
| LLM calls when stuck | N | 1 (escalation) |
| Latency | High (LLM per step) | Low (graph lookup) |
| Cost | $$$ | $ |
| Deterministic | No | Yes (except escalation) |

## License

MIT

## Citation

Based on: [Graph-Based Self-Healing Tool Routing for Cost-Efficient LLM Agents](https://arxiv.org/abs/2603.01548) (Bholani, 2026)
