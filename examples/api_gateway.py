"""API Gateway — Multi-provider AI inference with cost-aware routing.

A practical example: routing LLM inference requests across multiple
providers with automatic failover. This is the kind of infrastructure
that companies like Martian, OpenRouter, and LiteLLM build — but as
a simple, deterministic graph.

Graph:
    receive_request → openai_gpt4 ──────→ cache_result → respond
                    → anthropic_claude ──→ cache_result
                    → google_gemini ────→ cache_result
                    → local_ollama ─────→ cache_result  (fallback)

Weights represent cost per 1K tokens. Router always picks cheapest
available provider. When one goes down, traffic flows to the next.

Try it:
    python examples/api_gateway.py
"""

import time
from self_healing_router import SelfHealingRouter, EscalationResult


# --- Provider state ---
providers = {
    "openai": {"up": True, "latency_ms": 450},
    "anthropic": {"up": True, "latency_ms": 380},
    "google": {"up": True, "latency_ms": 320},
    "ollama": {"up": True, "latency_ms": 150},
}


def receive_request(data: dict) -> dict:
    prompt = data.get("prompt", "Hello, world!")
    print(f"    📥 Request received: \"{prompt[:50]}...\"")
    return {**data, "received_at": time.time()}


def make_provider(name: str, display: str, model: str):
    def handler(data: dict) -> dict:
        info = providers[name]
        if not info["up"]:
            raise ConnectionError(f"{display}: 503 Service Unavailable")
        prompt = data.get("prompt", "")
        latency = info["latency_ms"]
        print(f"    🤖 {display} ({model}): {latency}ms, {len(prompt)} chars")
        return {
            **data,
            "response": f"[{model} response to: {prompt[:30]}...]",
            "provider": name,
            "model": model,
            "latency_ms": latency,
        }
    return handler


def cache_result(data: dict) -> dict:
    provider = data.get("provider", "?")
    print(f"    💾 Cached response from {provider}")
    return {**data, "cached": True}


def respond(data: dict) -> dict:
    provider = data.get("provider", "?")
    model = data.get("model", "?")
    latency = data.get("latency_ms", 0)
    print(f"    📤 Response sent (provider: {provider}, model: {model}, {latency}ms)")
    return {**data, "responded": True}


def total_outage(failed_tool, attempted_paths, context):
    return EscalationResult(
        action="abort",
        detail="All inference providers unavailable. Returning 503 to client.",
    )


def build_router():
    router = SelfHealingRouter(escalation_callback=total_outage)

    router.add_tool("receive_request", handler=receive_request)
    router.add_tool("openai", handler=make_provider("openai", "OpenAI", "gpt-4o"))
    router.add_tool("anthropic", handler=make_provider("anthropic", "Anthropic", "claude-sonnet-4-20250514"))
    router.add_tool("google", handler=make_provider("google", "Google", "gemini-2.0-flash"))
    router.add_tool("ollama", handler=make_provider("ollama", "Ollama", "llama3.2:3b"))
    router.add_tool("cache_result", handler=cache_result)
    router.add_tool("respond", handler=respond)

    # Costs per 1K tokens (weights = cost in cents)
    # OpenAI gpt-4o: $2.50/1M input → 0.25¢/1K
    # Anthropic Claude: $3.00/1M → 0.30¢/1K
    # Google Gemini: $0.075/1M → 0.0075¢/1K
    # Local Ollama: ~$0 (electricity only) → 0.001
    router.add_edge("receive_request", "google", weight=0.0075)      # cheapest
    router.add_edge("receive_request", "ollama", weight=0.01)        # free but lower quality
    router.add_edge("receive_request", "openai", weight=0.25)        # mid
    router.add_edge("receive_request", "anthropic", weight=0.30)     # most expensive

    for provider in ["openai", "anthropic", "google", "ollama"]:
        router.add_edge(provider, "cache_result", weight=0.001)

    router.add_edge("cache_result", "respond", weight=0.001)

    return router


def run(title, overrides, request):
    for k, v in overrides.items():
        providers[k]["up"] = v

    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")

    router = build_router()
    result = router.route("receive_request", "respond", input_data=request)

    print(f"\n  {'✅' if result.success else '❌'} Provider: {result.outputs.get('respond', result.outputs.get('cache_result', {})).get('provider', 'none')}")
    print(f"  LLM routing calls: {result.llm_calls}")
    print(f"  Reroutes: {result.reroutes}")
    if result.errors:
        print(f"  Errors: {[e.split(':')[0] for e in result.errors]}")
    return result


def main():
    print("\n" + " API GATEWAY — Cost-Aware Provider Routing ".center(60, "═"))
    print("Routes to cheapest available provider. Auto-failover on outage.".center(60))

    req = {"prompt": "Explain quantum computing in simple terms for a blog post about emerging technology trends"}

    # 1: All up → picks cheapest (Google)
    run("All providers up → cheapest wins (Gemini)", 
        {"openai": True, "anthropic": True, "google": True, "ollama": True}, req)

    # 2: Google down → next cheapest (Ollama)
    run("Google down → falls to Ollama (local)",
        {"google": False}, req)

    # 3: Google + Ollama down → OpenAI
    run("Google + Ollama down → OpenAI",
        {"google": False, "ollama": False}, req)

    # 4: Only Anthropic left
    run("Only Anthropic available",
        {"openai": False, "google": False, "ollama": False, "anthropic": True}, req)

    # 5: Everything down
    run("Total outage → LLM escalation",
        {"openai": False, "anthropic": False, "google": False, "ollama": False}, req)

    print(f"\n{'═' * 60}")
    print("  WHY THIS MATTERS")
    print(f"{'═' * 60}")
    print("  Without Self-Healing Router, an API gateway needs:")
    print("    - Hand-coded if/else chains for each provider")
    print("    - Separate health check threads")
    print("    - Manual failover priority lists")
    print()
    print("  With Self-Healing Router:")
    print("    - Add providers as nodes, costs as weights")
    print("    - Dijkstra picks cheapest automatically")
    print("    - Failures reroute in <1ms, no config changes")
    print("    - New providers = one add_tool() + add_edge()")


if __name__ == "__main__":
    main()
