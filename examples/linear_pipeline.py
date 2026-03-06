"""Linear pipeline: A → B → C with failure injection.

Demonstrates automatic reroute when a tool in a linear chain fails
and an alternative path exists.

    fetch → parse → store
               ↓ (fails)
    fetch → transform → store  (auto-rerouted)
"""

from self_healing_router import SelfHealingRouter

call_log: list[str] = []


def fetch(data: dict) -> dict:
    call_log.append("fetch")
    return {**data, "raw": f"fetched({data.get('url', '?')})"}


def parse(data: dict) -> dict:
    call_log.append("parse")
    # Simulate failure on second call
    if "fail_parse" in data:
        raise RuntimeError("Parse service unavailable")
    return {**data, "parsed": f"parsed({data.get('raw', '?')})"}


def transform(data: dict) -> dict:
    """Backup parser — slower but works."""
    call_log.append("transform")
    return {**data, "parsed": f"transformed({data.get('raw', '?')})"}


def store(data: dict) -> dict:
    call_log.append("store")
    return {**data, "stored": True}


def main():
    router = SelfHealingRouter()

    # Add tools
    router.add_tool("fetch", handler=fetch)
    router.add_tool("parse", handler=parse)
    router.add_tool("transform", handler=transform)
    router.add_tool("store", handler=store)

    # Primary path: fetch → parse → store
    router.add_edge("fetch", "parse", weight=1.0)
    router.add_edge("parse", "store", weight=1.0)
    # Backup path: fetch → transform → store (higher weight = less preferred)
    router.add_edge("fetch", "transform", weight=3.0)
    router.add_edge("transform", "store", weight=1.0)

    print("=== Linear Pipeline Demo ===\n")
    print(router.visualize())
    print()

    # --- Run 1: Normal execution ---
    print("--- Run 1: Normal (no failures) ---")
    call_log.clear()
    result = router.route("fetch", "store", input_data={"url": "https://api.example.com"})
    print(f"  Path:     {' → '.join(result.path)}")
    print(f"  Success:  {result.success}")
    print(f"  Reroutes: {result.reroutes}")
    print(f"  LLM calls: {result.llm_calls}")
    print(f"  Tools called: {call_log}")
    print()

    # --- Run 2: Parse fails → auto-reroute to transform ---
    print("--- Run 2: Parse fails → reroute through transform ---")
    router.reset()
    call_log.clear()
    result = router.route("fetch", "store", input_data={"url": "https://api.example.com", "fail_parse": True})
    print(f"  Path:     {' → '.join(result.path) if result.path else 'rerouted'}")
    print(f"  Success:  {result.success}")
    print(f"  Reroutes: {result.reroutes}")
    print(f"  LLM calls: {result.llm_calls}")
    print(f"  Tools called: {call_log}")
    print(f"  Errors:   {result.errors}")
    print()
    print(router.health_report())


if __name__ == "__main__":
    main()
