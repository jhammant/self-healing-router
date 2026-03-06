"""LLM escalation demo — what happens when ALL paths fail.

This shows the escalation handler: the router exhausts all graph
paths, then calls an LLM exactly once to decide what to do.

Compare with ReAct which would make an LLM call at every single step.
"""

from self_healing_router import SelfHealingRouter, EscalationResult


def always_fail(data: dict) -> dict:
    raise RuntimeError("Service permanently down")


def my_llm_escalation(
    failed_tool: str,
    attempted_paths: list[list[str]],
    context: dict,
) -> EscalationResult:
    """Simulated LLM escalation.
    
    In production, this would call your LLM:
    
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"Tool {failed_tool} failed..."}]
        )
    """
    print(f"  🤖 LLM ESCALATION: Tool '{failed_tool}' has no viable path")
    print(f"     Paths tried: {len(attempted_paths)}")
    print(f"     Escalation #{context['escalation_number']}")
    
    # LLM decides to abort gracefully
    return EscalationResult(
        action="abort",
        detail=f"LLM decided to abort: '{failed_tool}' and all alternatives are down. "
               "Recommend manual intervention.",
    )


def main():
    # Router with custom LLM escalation
    router = SelfHealingRouter(escalation_callback=my_llm_escalation)

    router.add_tool("start", handler=lambda d: {**d, "started": True})
    router.add_tool("process_a", handler=always_fail)
    router.add_tool("process_b", handler=always_fail)
    router.add_tool("end", handler=lambda d: {**d, "done": True})

    router.add_edge("start", "process_a", weight=1.0)
    router.add_edge("start", "process_b", weight=2.0)
    router.add_edge("process_a", "end", weight=1.0)
    router.add_edge("process_b", "end", weight=1.0)

    print("=== LLM Escalation Demo ===\n")
    print("All processing tools will fail. Router exhausts graph paths,")
    print("then escalates to LLM exactly ONCE.\n")
    print(router.visualize())
    print()

    result = router.route("start", "end")
    
    print()
    print(f"  Path:       {' → '.join(result.path) if result.path else '(none)'}")
    print(f"  Success:    {result.success}")
    print(f"  Reroutes:   {result.reroutes}")
    print(f"  LLM calls:  {result.llm_calls}  ← (ReAct would use ~6+ here)")
    print(f"  Errors:     {result.errors}")
    print()

    # Compare: in a ReAct loop, every single step would trigger an LLM call:
    #   Step 1: LLM decides tool → process_a (1 call)
    #   Step 2: process_a fails, LLM decides → process_b (1 call)
    #   Step 3: process_b fails, LLM decides → retry? (1 call)
    #   ... and so on
    # Self-Healing Router: 2 reroutes (deterministic) + 1 LLM call (escalation)
    
    react_equivalent_calls = len(result.errors) * 2 + 1  # conservative
    print(f"  📊 ReAct equivalent: ~{react_equivalent_calls}+ LLM calls")
    print(f"  📊 Self-Healing Router: {result.llm_calls} LLM call(s)")
    savings = ((react_equivalent_calls - result.llm_calls) / react_equivalent_calls) * 100
    print(f"  📊 Savings: {savings:.0f}% fewer LLM calls")


if __name__ == "__main__":
    main()
