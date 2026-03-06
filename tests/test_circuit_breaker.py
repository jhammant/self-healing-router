"""Tests for circuit breaker pattern."""

import time
from unittest.mock import patch

from self_healing_router import HealthMonitor, CircuitBreakerState


class TestCircuitBreaker:
    """Circuit breaker state transitions and weight penalties."""

    def test_initial_state_closed(self):
        m = HealthMonitor("tool")
        assert m.circuit_state == CircuitBreakerState.CLOSED

    def test_closed_to_open_after_threshold_failures(self):
        m = HealthMonitor("tool")
        m._failure_threshold = 3
        m.record_failure()
        m.record_failure()
        assert m.circuit_state == CircuitBreakerState.CLOSED
        m.record_failure()
        assert m.circuit_state == CircuitBreakerState.OPEN

    def test_open_to_half_open_after_timeout(self):
        m = HealthMonitor("tool")
        m._failure_threshold = 1
        m._recovery_timeout = 0.01  # 10ms
        m.record_failure()
        assert m.circuit_state == CircuitBreakerState.OPEN
        time.sleep(0.02)
        assert m.circuit_state == CircuitBreakerState.HALF_OPEN

    def test_half_open_to_closed_on_success(self):
        m = HealthMonitor("tool")
        m._failure_threshold = 1
        m._recovery_timeout = 0.01
        m.record_failure()
        time.sleep(0.02)
        assert m.circuit_state == CircuitBreakerState.HALF_OPEN
        m.record_success(10.0)
        assert m.circuit_state == CircuitBreakerState.CLOSED

    def test_half_open_to_open_on_failure(self):
        m = HealthMonitor("tool")
        m._failure_threshold = 1
        m._recovery_timeout = 0.01
        m.record_failure()
        time.sleep(0.02)
        assert m.circuit_state == CircuitBreakerState.HALF_OPEN
        m.record_failure()
        assert m.circuit_state == CircuitBreakerState.OPEN

    def test_weight_penalty_closed(self):
        m = HealthMonitor("tool")
        for _ in range(5):
            m.record_success(100.0)
        assert m.weight_penalty() == 1.0

    def test_weight_penalty_open(self):
        m = HealthMonitor("tool")
        m._failure_threshold = 1
        m.record_failure()
        assert m.weight_penalty() == float("inf")

    def test_weight_penalty_half_open(self):
        m = HealthMonitor("tool")
        m._failure_threshold = 1
        m._recovery_timeout = 0.01
        m.record_failure()
        time.sleep(0.02)
        assert m.circuit_state == CircuitBreakerState.HALF_OPEN
        assert m.weight_penalty() == 10.0

    def test_trip_manual(self):
        m = HealthMonitor("tool")
        m.trip()
        assert m.circuit_state == CircuitBreakerState.OPEN
        assert m.weight_penalty() == float("inf")

    def test_reset_circuit_manual(self):
        m = HealthMonitor("tool")
        m.trip()
        m.reset_circuit()
        assert m.circuit_state == CircuitBreakerState.CLOSED

    def test_consecutive_failures_reset_on_success(self):
        m = HealthMonitor("tool")
        m._failure_threshold = 3
        m.record_failure()
        m.record_failure()
        m.record_success(10.0)
        m.record_failure()
        m.record_failure()
        # Should still be CLOSED — success reset the counter
        assert m.circuit_state == CircuitBreakerState.CLOSED
