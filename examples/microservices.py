"""Microservices — Service mesh routing with degraded responses.

Models a typical microservices architecture where an API gateway
routes through multiple services. Demonstrates how Self-Healing
Router can replace hand-coded retry/circuit-breaker logic in
service mesh configurations.

This is essentially what Envoy/Istio do, but as a 50-line Python graph.

Graph:
    gateway → auth_service → user_service → recommendation_engine → response
                            → user_cache   → popular_items          → response
                                           → static_defaults        → response

When the recommendation engine is slow/down, the system automatically
degrades to cached popular items or static defaults — no config changes,
no LLM, just cheaper edges.

Try it:
    python examples/microservices.py
"""

import time
from self_healing_router import SelfHealingRouter, EscalationResult

services = {"auth": True, "user": True, "user_cache": True, "reco": True, "popular": True}


def gateway(data: dict) -> dict:
    user_id = data.get("user_id", "u_123")
    print(f"    🌐 Gateway: request for user {user_id}")
    return {**data, "request_id": f"req_{int(time.time())}", "gateway_ts": time.time()}


def auth_service(data: dict) -> dict:
    if not services["auth"]:
        raise ConnectionError("Auth service: 503 (Redis session store down)")
    user_id = data.get("user_id", "u_123")
    print(f"    🔐 Auth: token verified for {user_id}")
    return {**data, "authenticated": True, "token_valid": True}


def user_service(data: dict) -> dict:
    if not services["user"]:
        raise ConnectionError("User service: 504 (PostgreSQL timeout)")
    print(f"    👤 User service: profile loaded (preferences, history)")
    return {**data, "profile": {"tier": "premium", "history_items": 142}, "personalized": True}


def user_cache(data: dict) -> dict:
    if not services["user_cache"]:
        raise ConnectionError("User cache: Redis WRONGTYPE error")
    print(f"    📋 User cache: basic profile from Redis (no history)")
    return {**data, "profile": {"tier": "unknown", "history_items": 0}, "personalized": False}


def recommendation_engine(data: dict) -> dict:
    if not services["reco"]:
        raise ConnectionError("Reco engine: model server OOM killed")
    items = data.get("profile", {}).get("history_items", 0)
    print(f"    🎯 Recommendations: personalised from {items} history items")
    return {**data, "items": ["item_a", "item_b", "item_c"], "reco_type": "personalized"}


def popular_items(data: dict) -> dict:
    if not services["popular"]:
        raise ConnectionError("Popular items: cache miss")
    print(f"    📊 Popular items: trending this week (non-personalized)")
    return {**data, "items": ["trending_1", "trending_2", "trending_3"], "reco_type": "popular"}


def static_defaults(data: dict) -> dict:
    print(f"    📦 Static defaults: hardcoded editorial picks (zero latency)")
    return {**data, "items": ["editorial_1", "editorial_2"], "reco_type": "static"}


def response(data: dict) -> dict:
    reco_type = data.get("reco_type", "?")
    items = data.get("items", [])
    personalized = data.get("personalized", False)
    print(f"    📤 Response: {len(items)} items ({reco_type}), personalized={personalized}")
    return {**data, "status": 200, "served": True}


def total_failure(failed_tool, attempted_paths, context):
    return EscalationResult(
        action="abort",
        detail="Service mesh critically degraded. Returning 503 with retry-after.",
    )


def build_router():
    router = SelfHealingRouter(escalation_callback=total_failure)

    for name, fn in [
        ("gateway", gateway), ("auth_service", auth_service),
        ("user_service", user_service), ("user_cache", user_cache),
        ("recommendation_engine", recommendation_engine),
        ("popular_items", popular_items), ("static_defaults", static_defaults),
        ("response", response),
    ]:
        router.add_tool(name, handler=fn)

    # Gateway → Auth (always required)
    router.add_edge("gateway", "auth_service", weight=1.0)

    # Auth → User data (primary vs cache)
    router.add_edge("auth_service", "user_service", weight=1.0)     # full profile
    router.add_edge("auth_service", "user_cache", weight=2.0)       # cached basics

    # User → Recommendations (tiered quality)
    router.add_edge("user_service", "recommendation_engine", weight=1.0)  # best: personalized
    router.add_edge("user_service", "popular_items", weight=3.0)          # decent: trending
    router.add_edge("user_service", "static_defaults", weight=5.0)        # minimum: editorial

    router.add_edge("user_cache", "recommendation_engine", weight=2.0)  # reco without full profile
    router.add_edge("user_cache", "popular_items", weight=2.5)         # cache → popular is natural
    router.add_edge("user_cache", "static_defaults", weight=4.0)

    # All recommendations → response
    for src in ["recommendation_engine", "popular_items", "static_defaults"]:
        router.add_edge(src, "response", weight=1.0)

    return router


def run(title, desc, overrides):
    global services
    services = {k: True for k in services}
    services.update(overrides)

    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"  {desc}")
    print(f"{'─' * 60}")

    router = build_router()
    result = router.route("gateway", "response", input_data={"user_id": "u_42"})

    reco = result.outputs.get("response", {}).get("reco_type", "none")
    print(f"\n  {'✅' if result.success else '❌'} Quality: {reco} | Reroutes: {result.reroutes} | LLM: {result.llm_calls}")
    if result.errors:
        for e in result.errors:
            print(f"    └─ {e.split(': ')[0]}")
    return result


def main():
    print("\n" + " MICROSERVICES — Graceful Degradation ".center(60, "═"))
    print("Quality degrades smoothly as services fail:".center(60))
    print("personalized → popular → static defaults".center(60))

    # Tier 1: Full personalization
    run("Full Stack Healthy",
        "All services up → personalized recommendations",
        {})

    # Tier 2: Reco engine down → popular items
    run("Recommendation Engine Down",
        "ML model OOMed → auto-degrade to trending items",
        {"reco": False})

    # Tier 3: User service down → cache + popular
    run("User Service Down",
        "Postgres timeout → use Redis cache + popular items",
        {"user": False})

    # Tier 4: Multiple failures → static defaults
    run("User + Reco + Popular All Down",
        "Everything but auth and defaults → editorial picks",
        {"user": False, "reco": False, "popular": False})

    # Tier 5: Auth down = game over
    run("Auth Service Down",
        "Can't verify identity → must escalate (503)",
        {"auth": False})

    print(f"\n{'═' * 60}")
    print("  GRACEFUL DEGRADATION WITHOUT CODE CHANGES")
    print(f"{'═' * 60}")
    print("  The graph encodes your degradation policy:")
    print("    weight=1 → premium path (personalized reco)")
    print("    weight=3 → acceptable (popular items)")
    print("    weight=5 → minimum viable (static defaults)")
    print()
    print("  When services recover, traffic automatically flows")
    print("  back to the cheaper (= better quality) path.")
    print("  No deploy, no config change, no LLM call.")


if __name__ == "__main__":
    main()
