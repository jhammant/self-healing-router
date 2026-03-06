"""Tests for composite weight function and rate limiting."""

import math

from self_healing_router import HealthMonitor, CircuitBreakerState


class TestCompositeWeight:
    """Composite weight with various scenarios."""

    def test_healthy_default(self):
        m = HealthMonitor("tool", latency_threshold_ms=1000.0)
        for _ in range(5):
            m.record_success(500.0)  # 50% of threshold
        w = m.composite_weight()
        assert 0.4 < w < 2.0  # near 1.0

    def test_high_latency_increases_weight(self):
        m = HealthMonitor("tool", latency_threshold_ms=100.0)
        for _ in range(5):
            m.record_success(500.0)  # 5x threshold
        w = m.composite_weight()
        assert w > 4.0

    def test_high_error_rate_increases_weight(self):
        m = HealthMonitor("tool", latency_threshold_ms=1000.0)
        for _ in range(5):
            m.record_success(100.0)
        for _ in range(5):
            m.record_failure(100.0)
        # 50% error rate -> reliability = 1/(1-0.5) = 2.0
        w = m.composite_weight()
        assert w > 1.5

    def test_open_circuit_returns_inf(self):
        m = HealthMonitor("tool")
        m.trip()
        assert m.composite_weight() == float("inf")

    def test_half_open_multiplies_by_10(self):
        m = HealthMonitor("tool", latency_threshold_ms=1000.0)
        for _ in range(5):
            m.record_success(500.0)
        w_normal = m.composite_weight()
        m._circuit_state = CircuitBreakerState.HALF_OPEN
        w_half = m.composite_weight()
        assert w_half == w_normal * 10.0

    def test_rate_limit_at_zero_returns_inf(self):
        m = HealthMonitor("tool")
        for _ in range(5):
            m.record_success(100.0)
        m.set_rate_limit(0, 100)
        assert m.composite_weight() == float("inf")

    def test_rate_limit_plenty_no_impact(self):
        m = HealthMonitor("tool", latency_threshold_ms=1000.0)
        for _ in range(5):
            m.record_success(500.0)
        w_no_rl = m.composite_weight()
        m.set_rate_limit(80, 100)
        w_with_rl = m.composite_weight()
        assert w_no_rl == w_with_rl  # ratio > 0.5, factor = 1.0

    def test_no_data_returns_base(self):
        m = HealthMonitor("tool")
        assert m.composite_weight(base_cost=5.0) == 5.0

    def test_base_cost_scales(self):
        m = HealthMonitor("tool", latency_threshold_ms=1000.0)
        for _ in range(5):
            m.record_success(500.0)
        w1 = m.composite_weight(base_cost=1.0)
        w2 = m.composite_weight(base_cost=3.0)
        assert abs(w2 - w1 * 3.0) < 0.001


class TestRateLimitFactor:
    """Rate limit factor at different quota levels."""

    def test_no_rate_limit_set(self):
        m = HealthMonitor("tool")
        assert m.rate_limit_factor() == 1.0

    def test_plenty_of_quota(self):
        m = HealthMonitor("tool")
        m.set_rate_limit(80, 100)
        assert m.rate_limit_factor() == 1.0

    def test_moderate_quota(self):
        m = HealthMonitor("tool")
        m.set_rate_limit(30, 100)
        assert m.rate_limit_factor() == 2.0

    def test_low_quota(self):
        m = HealthMonitor("tool")
        m.set_rate_limit(15, 100)
        assert m.rate_limit_factor() == 10.0

    def test_very_low_quota(self):
        m = HealthMonitor("tool")
        m.set_rate_limit(7, 100)
        assert m.rate_limit_factor() == 50.0

    def test_critical_quota(self):
        m = HealthMonitor("tool")
        m.set_rate_limit(3, 100)
        assert m.rate_limit_factor() == 100.0

    def test_zero_quota(self):
        m = HealthMonitor("tool")
        m.set_rate_limit(0, 100)
        assert m.rate_limit_factor() == float("inf")

    def test_zero_total(self):
        m = HealthMonitor("tool")
        m.set_rate_limit(0, 0)
        assert m.rate_limit_factor() == 1.0
