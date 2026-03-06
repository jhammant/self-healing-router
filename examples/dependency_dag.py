"""Dependency DAG with parallel dependencies and fallback paths.

Multiple tools feed into a merge step. If one dependency fails,
the router finds an alternative path.

    api_a ──→ enrich ──→ merge ──→ output
    api_b ──→ validate ──→ merge
    api_c ──→ merge  (direct, lower quality fallback)
    
If enrich fails, route through api_c as fallback.
"""

from self_healing_router import SelfHealingRouter

call_log: list[str] = []


def api_a(data: dict) -> dict:
    call_log.append("api_a")
    return {**data, "source_a": "primary data"}


def api_b(data: dict) -> dict:
    call_log.append("api_b")
    return {**data, "source_b": "secondary data"}


def api_c(data: dict) -> dict:
    call_log.append("api_c")
    return {**data, "source_c": "fallback data"}


def enrich(data: dict) -> dict:
    call_log.append("enrich")
    if data.get("fail_enrich"):
        raise RuntimeError("Enrichment service down")
    return {**data, "enriched": True}


def validate(data: dict) -> dict:
    call_log.append("validate")
    return {**data, "validated": True}


def merge(data: dict) -> dict:
    call_log.append("merge")
    return {**data, "merged": True}


def output(data: dict) -> dict:
    call_log.append("output")
    return {**data, "final": True}


def main():
    router = SelfHealingRouter()

    # Tools
    for name, fn in [
        ("api_a", api_a), ("api_b", api_b), ("api_c", api_c),
        ("enrich", enrich), ("validate", validate),
        ("merge", merge), ("output", output),
    ]:
        router.add_tool(name, handler=fn)

    # Primary path: api_a → enrich → merge → output
    router.add_edge("api_a", "enrich", weight=1.0)
    router.add_edge("enrich", "merge", weight=1.0)
    router.add_edge("merge", "output", weight=1.0)
    
    # Secondary path: api_b → validate → merge
    router.add_edge("api_b", "validate", weight=1.5)
    router.add_edge("validate", "merge", weight=1.0)
    
    # Fallback: api_c → merge directly (cheap but lower quality)
    router.add_edge("api_c", "merge", weight=2.0)
    
    # Cross-links for rerouting
    router.add_edge("api_a", "merge", weight=4.0)  # skip enrich if needed

    print("=== Dependency DAG Demo ===\n")
    print(router.visualize())
    print()

    # --- Run 1: Normal ---
    print("--- Run 1: Normal (primary path) ---")
    call_log.clear()
    result = router.route("api_a", "output")
    print(f"  Path:      {' → '.join(result.path)}")
    print(f"  Success:   {result.success}")
    print(f"  Reroutes:  {result.reroutes}")
    print(f"  LLM calls: {result.llm_calls}")
    print(f"  Tools:     {call_log}")
    print()

    # --- Run 2: Enrich fails → reroute skipping enrich ---
    print("--- Run 2: Enrich fails → reroute ---")
    router.reset()
    call_log.clear()
    result = router.route("api_a", "output", input_data={"fail_enrich": True})
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
