"""Parallel fan-out / fan-in with partial failure handling.

A dispatch node fans out to multiple workers. Results converge
at an aggregator. If some workers fail, the router finds 
alternative paths through backup workers.

    dispatch ──→ worker_1 ──→ aggregate ──→ report
             ──→ worker_2 ──→ aggregate
             ──→ worker_3 ──→ aggregate
             ──→ backup_1 ──→ aggregate  (standby)
             
If worker_2 fails, traffic reroutes through backup_1.
"""

from self_healing_router import SelfHealingRouter

call_log: list[str] = []
fail_workers: set[str] = set()


def make_worker(name: str):
    def worker(data: dict) -> dict:
        call_log.append(name)
        if name in fail_workers:
            raise RuntimeError(f"{name} is down")
        return {**data, name: f"result from {name}"}
    return worker


def dispatch(data: dict) -> dict:
    call_log.append("dispatch")
    return {**data, "dispatched": True}


def aggregate(data: dict) -> dict:
    call_log.append("aggregate")
    return {**data, "aggregated": True}


def report(data: dict) -> dict:
    call_log.append("report")
    return {**data, "reported": True}


def main():
    router = SelfHealingRouter()

    # Core tools
    router.add_tool("dispatch", handler=dispatch)
    router.add_tool("aggregate", handler=aggregate)
    router.add_tool("report", handler=report)
    
    # Workers (primary)
    for i in range(1, 4):
        name = f"worker_{i}"
        router.add_tool(name, handler=make_worker(name))
        router.add_edge("dispatch", name, weight=1.0)
        router.add_edge(name, "aggregate", weight=1.0)
    
    # Backup workers (higher weight = less preferred)
    for i in range(1, 3):
        name = f"backup_{i}"
        router.add_tool(name, handler=make_worker(name))
        router.add_edge("dispatch", name, weight=3.0)
        router.add_edge(name, "aggregate", weight=1.0)

    # Final edge
    router.add_edge("aggregate", "report", weight=1.0)

    print("=== Parallel Fan-Out Demo ===\n")
    print(router.visualize())
    print()

    # --- Run 1: Normal ---
    print("--- Run 1: Normal (cheapest worker picked) ---")
    call_log.clear()
    fail_workers.clear()
    result = router.route("dispatch", "report")
    print(f"  Path:      {' → '.join(result.path)}")
    print(f"  Success:   {result.success}")
    print(f"  Reroutes:  {result.reroutes}")
    print(f"  LLM calls: {result.llm_calls}")
    print(f"  Tools:     {call_log}")
    print()

    # --- Run 2: worker_1 fails → reroute to worker_2 ---
    print("--- Run 2: worker_1 fails → reroute ---")
    router.reset()
    call_log.clear()
    fail_workers.add("worker_1")
    result = router.route("dispatch", "report")
    print(f"  Path:      {' → '.join(result.path) if result.path else 'rerouted'}")
    print(f"  Success:   {result.success}")
    print(f"  Reroutes:  {result.reroutes}")
    print(f"  LLM calls: {result.llm_calls}")
    print(f"  Tools:     {call_log}")
    print()

    # --- Run 3: All primary workers fail → backup kicks in ---
    print("--- Run 3: All primary workers fail → backup ---")
    router.reset()
    call_log.clear()
    fail_workers.update(["worker_1", "worker_2", "worker_3"])
    result = router.route("dispatch", "report")
    print(f"  Path:      {' → '.join(result.path) if result.path else 'rerouted'}")
    print(f"  Success:   {result.success}")
    print(f"  Reroutes:  {result.reroutes}")
    print(f"  LLM calls: {result.llm_calls}")
    print(f"  Tools:     {call_log}")
    print(f"  Errors:    {result.errors}")
    print()
    print(router.health_report())


if __name__ == "__main__":
    main()
