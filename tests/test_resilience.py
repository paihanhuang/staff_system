"""Tests for resilience utilities."""

import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock

from src.utils.resilience import (
    RetryConfig,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    with_retry,
    calculate_delay,
    CircuitOpenError,
    MaxRetriesExceededError,
)


class TestRetryConfig:
    """Tests for RetryConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RetryConfig()
        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.jitter is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = RetryConfig(max_attempts=5, base_delay=0.5)
        assert config.max_attempts == 5
        assert config.base_delay == 0.5


class TestCalculateDelay:
    """Tests for calculate_delay function."""

    def test_exponential_backoff(self):
        """Test exponential backoff calculation."""
        config = RetryConfig(base_delay=1.0, exponential_base=2.0, jitter=False)
        
        delay_0 = calculate_delay(0, config)
        delay_1 = calculate_delay(1, config)
        delay_2 = calculate_delay(2, config)
        
        assert delay_0 == 1.0  # 1 * 2^0 = 1
        assert delay_1 == 2.0  # 1 * 2^1 = 2
        assert delay_2 == 4.0  # 1 * 2^2 = 4

    def test_max_delay_cap(self):
        """Test that delay is capped at max_delay."""
        config = RetryConfig(base_delay=1.0, max_delay=5.0, exponential_base=2.0, jitter=False)
        
        delay_10 = calculate_delay(10, config)  # Would be 1024 without cap
        assert delay_10 == 5.0

    def test_jitter_within_bounds(self):
        """Test that jitter stays within ±25%."""
        config = RetryConfig(base_delay=4.0, jitter=True)
        
        delays = [calculate_delay(0, config) for _ in range(100)]
        
        # All delays should be within ±25% of 4.0
        for delay in delays:
            assert 3.0 <= delay <= 5.0


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""

    def test_initial_state_closed(self):
        """Test circuit starts in closed state."""
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED
        assert cb.can_execute() is True

    def test_opens_after_threshold(self):
        """Test circuit opens after failure threshold."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker(config=config)
        
        for _ in range(3):
            cb.record_failure()
        
        assert cb.state == CircuitState.OPEN
        assert cb.can_execute() is False

    def test_success_resets_failure_count(self):
        """Test success resets failure count."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker(config=config)
        
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        
        assert cb.failure_count == 0
        assert cb.state == CircuitState.CLOSED


class TestWithRetry:
    """Tests for the with_retry decorator."""

    @pytest.mark.asyncio
    async def test_successful_call_no_retry(self):
        """Test successful call returns immediately."""
        call_count = 0
        
        @with_retry(RetryConfig(max_attempts=3))
        async def success_fn():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = await success_fn()
        
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """Test function is retried on failure."""
        call_count = 0
        
        @with_retry(RetryConfig(max_attempts=3, base_delay=0.01))
        async def fail_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("temporary error")
            return "success"
        
        result = await fail_twice()
        
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        """Test MaxRetriesExceededError when all attempts fail."""
        @with_retry(RetryConfig(max_attempts=2, base_delay=0.01))
        async def always_fail():
            raise ValueError("always fails")
        
        with pytest.raises(MaxRetriesExceededError):
            await always_fail()

    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self):
        """Test retry with circuit breaker."""
        config = CircuitBreakerConfig(failure_threshold=2)
        cb = CircuitBreaker(config=config)
        
        @with_retry(RetryConfig(max_attempts=1), cb)
        async def failing_fn():
            raise ValueError("fail")
        
        # First failure
        try:
            await failing_fn()
        except MaxRetriesExceededError:
            pass
        
        # Second failure opens circuit
        try:
            await failing_fn()
        except (MaxRetriesExceededError, CircuitOpenError):
            pass
        
        assert cb.state == CircuitState.OPEN
