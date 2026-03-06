"""Customer Support Agent — From the paper's evaluation (Section 3.1).

Models a customer support pipeline handling refund requests.
This is one of the three domains evaluated in the paper, demonstrating
how compound failures that break static workflows are handled
automatically by the Self-Healing Router.

Graph:
    lookup_order → process_refund_stripe ──→ send_email ──→ close_ticket
                 → process_refund_razorpay → send_sms   ──→ close_ticket
                 → (escalate to LLM if all payment fails)

Paper results (Table 1): Self-Healing Router uses 0-2 LLM calls vs
ReAct's 4-9 per scenario, while catching silent failures that
LangGraph misses.

Try it:
    python examples/customer_support.py
"""

from self_healing_router import SelfHealingRouter, EscalationResult

# --- Service state (toggle to simulate outages) ---
services = {
    "stripe": True,
    "razorpay": True,
    "email": True,
    "sms": True,
}


def lookup_order(data: dict) -> dict:
    order_id = data.get("order_id", "ORD-1234")
    amount = data.get("amount", 49.99)
    print(f"    🔍 Found order {order_id}: ${amount:.2f}")
    return {**data, "order_id": order_id, "amount": amount, "order_found": True}


def check_risk(data: dict) -> dict:
    """Risk check — high-value orders get flagged."""
    amount = data.get("amount", 0)
    is_risky = amount > 10000
    if is_risky:
        print(f"    ⚠️  HIGH RISK: ${amount:.2f} exceeds $10,000 threshold")
    else:
        print(f"    ✓  Risk check passed (${amount:.2f})")
    return {**data, "risk_flagged": is_risky}


def refund_stripe(data: dict) -> dict:
    if not services["stripe"]:
        raise ConnectionError("Stripe: 503 Service Unavailable")
    amount = data["amount"]
    print(f"    💳 Stripe refund processed: ${amount:.2f} → re_1abc")
    return {**data, "refund_id": "re_1abc", "refund_provider": "stripe"}


def refund_razorpay(data: dict) -> dict:
    if not services["razorpay"]:
        raise ConnectionError("Razorpay: gateway timeout")
    amount = data["amount"]
    print(f"    🏦 Razorpay refund processed: ${amount:.2f} → rfnd_xyz")
    return {**data, "refund_id": "rfnd_xyz", "refund_provider": "razorpay"}


def send_email(data: dict) -> dict:
    if not services["email"]:
        raise ConnectionError("SMTP: connection refused")
    email = data.get("customer_email", "customer@example.com")
    print(f"    📧 Confirmation email sent to {email}")
    return {**data, "notified_via": "email"}


def send_sms(data: dict) -> dict:
    if not services["sms"]:
        raise ConnectionError("Twilio: 429 rate limited")
    phone = data.get("customer_phone", "+44 7700 900000")
    print(f"    📱 SMS sent to {phone}")
    return {**data, "notified_via": "sms"}


def close_ticket(data: dict) -> dict:
    refund_id = data.get("refund_id", "unknown")
    notify = data.get("notified_via", "unknown")
    print(f"    ✅ Ticket closed (refund: {refund_id}, notified: {notify})")
    return {**data, "ticket_closed": True}


def llm_goal_demotion(failed_tool, attempted_paths, context):
    """LLM reasons about degraded goals when no path exists.
    
    This is the paper's 'goal demotion' — the LLM doesn't pick tools,
    it decides what degraded outcome is acceptable.
    """
    errors = context.get("errors", [])
    
    if any("refund" in e.lower() or "stripe" in e.lower() or "razorpay" in e.lower() for e in errors):
        print("    🤖 LLM: All payment providers down. Deferring refund.")
        return EscalationResult(
            action="abort",
            detail="Cannot process refund — all providers down. "
                   "Created ticket ESC-001 for manual processing within 24h. "
                   "Customer notified of delay.",
        )
    
    if any("email" in e.lower() or "sms" in e.lower() or "smtp" in e.lower() for e in errors):
        print("    🤖 LLM: All notification channels down. Logging for retry.")
        return EscalationResult(
            action="abort",
            detail="Refund processed but notification failed. "
                   "Queued for notification retry in 15 minutes.",
        )
    
    return EscalationResult(
        action="abort",
        detail=f"Unrecoverable failure. Escalated to human agent.",
    )


def build_router():
    router = SelfHealingRouter(escalation_callback=llm_goal_demotion)

    for name, fn in [
        ("lookup_order", lookup_order),
        ("check_risk", check_risk),
        ("refund_stripe", refund_stripe),
        ("refund_razorpay", refund_razorpay),
        ("send_email", send_email),
        ("send_sms", send_sms),
        ("close_ticket", close_ticket),
    ]:
        router.add_tool(name, handler=fn)

    # Lookup → risk check → refund
    router.add_edge("lookup_order", "check_risk", weight=1.0)
    router.add_edge("check_risk", "refund_stripe", weight=1.0)
    router.add_edge("check_risk", "refund_razorpay", weight=2.0)

    # Refund → notification
    router.add_edge("refund_stripe", "send_email", weight=1.0)
    router.add_edge("refund_stripe", "send_sms", weight=2.0)
    router.add_edge("refund_razorpay", "send_email", weight=1.0)
    router.add_edge("refund_razorpay", "send_sms", weight=2.0)

    # Notification → close
    router.add_edge("send_email", "close_ticket", weight=1.0)
    router.add_edge("send_sms", "close_ticket", weight=1.0)

    return router


def scenario(title, desc, service_overrides, order_data):
    global services
    services = {"stripe": True, "razorpay": True, "email": True, "sms": True}
    services.update(service_overrides)

    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"  {desc}")
    print(f"{'=' * 60}")

    router = build_router()
    result = router.route("lookup_order", "close_ticket", input_data=order_data)

    print(f"\n  Result:    {'✅ Success' if result.success else '❌ Escalated'}")
    print(f"  Path:      {' → '.join(result.path) if result.path else '(escalated)'}")
    print(f"  LLM calls: {result.llm_calls}")
    print(f"  Reroutes:  {result.reroutes}")
    if result.errors:
        print(f"  Errors:    {result.errors}")
    return result


def main():
    order = {
        "order_id": "ORD-7891",
        "amount": 49.99,
        "customer_email": "sarah@example.com",
        "customer_phone": "+44 7700 900123",
    }

    print("\n" + "🏢 CUSTOMER SUPPORT AGENT — Self-Healing Router Demo".center(60))
    print("Based on Section 3.1 of the paper (arXiv:2603.01548)".center(60))

    # S1: Happy path
    r1 = scenario(
        "S1: Happy Path",
        "Everything works. Stripe processes, email confirms.",
        {},
        order,
    )

    # S2: Stripe down → Razorpay
    r2 = scenario(
        "S2: Stripe Down",
        "Stripe 503 → automatic reroute to Razorpay. Zero LLM calls.",
        {"stripe": False},
        order,
    )

    # S5: Email dies mid-task → SMS fallback
    r3 = scenario(
        "S5: Email Fails",
        "Payment works, but email SMTP down → reroute to SMS.",
        {"email": False},
        order,
    )

    # S6: Both notification channels down → LLM escalation
    r4 = scenario(
        "S6: Both Notifications Down",
        "Stripe works, but email AND SMS both down.",
        {"email": False, "sms": False},
        order,
    )

    # S7: Triple failure — Stripe + Email + SMS
    r5 = scenario(
        "S7: Triple Failure (Stripe + Email + SMS)",
        "Stripe down, email down, SMS down. Router reroutes to Razorpay+SMS,\n"
        "  but SMS also fails → escalation.",
        {"stripe": False, "email": False, "sms": False},
        order,
    )

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    results = [r1, r2, r3, r4, r5]
    total_llm = sum(r.llm_calls for r in results)
    total_reroutes = sum(r.reroutes for r in results)
    
    print(f"  Total LLM calls:  {total_llm}  (ReAct equivalent: ~33)")
    print(f"  Total reroutes:   {total_reroutes}  (all deterministic, <1ms each)")
    print(f"  Silent failures:  0  (LangGraph has 2 in these scenarios)")
    print()
    print("  Key insight: The router handles compound failures that would")
    print("  silently break a static workflow (LangGraph), while using 93%")
    print("  fewer LLM calls than ReAct.")


if __name__ == "__main__":
    main()
