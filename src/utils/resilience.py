"""Resilience utilities with retry logic and circuit breaker patterns."""

import asyncio
import functools
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Type, TypeVar

from src.utils.logger import get_logger

logger = get_logger()

T = TypeVar("T")


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple[Type[Exception], ...] = (Exception,)


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5  # Failures before opening
    recovery_timeout: float = 30.0  # Seconds before trying again
    half_open_max_calls: int = 3  # Test calls in half-open state


@dataclass
class CircuitBreaker:
    """Circuit breaker for detecting and handling service failures."""

    config: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure_time: float = 0.0
    half_open_calls: int = 0

    def can_execute(self) -> bool:
        """Check if execution is allowed based on circuit state."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if time.time() - self.last_failure_time >= self.config.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
                logger.info("Circuit breaker transitioning to HALF_OPEN")
                return True
            return False

        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_calls < self.config.half_open_max_calls:
                self.half_open_calls += 1
                return True
            return False

        return False

    def record_success(self) -> None:
        """Record a successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            logger.info("Circuit breaker transitioning to CLOSED")
        self.failure_count = 0

    def record_failure(self) -> None:
        """Record a failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            logger.warning("Circuit breaker transitioning to OPEN from HALF_OPEN")
        elif self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(
                f"Circuit breaker OPEN after {self.failure_count} failures"
            )


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open and rejecting requests."""

    pass


class MaxRetriesExceededError(Exception):
    """Raised when all retry attempts are exhausted."""

    def __init__(self, message: str, last_exception: Optional[Exception] = None):
        super().__init__(message)
        self.last_exception = last_exception


def calculate_delay(
    attempt: int,
    config: RetryConfig,
) -> float:
    """Calculate delay for a retry attempt with exponential backoff."""
    delay = config.base_delay * (config.exponential_base ** attempt)
    delay = min(delay, config.max_delay)

    if config.jitter:
        # Add random jitter (±25% of delay)
        jitter_range = delay * 0.25
        delay = delay + random.uniform(-jitter_range, jitter_range)

    return max(0, delay)


def with_retry(
    config: Optional[RetryConfig] = None,
    circuit_breaker: Optional[CircuitBreaker] = None,
) -> Callable:
    """Decorator for adding retry logic with optional circuit breaker.

    Args:
        config: Retry configuration. Defaults to RetryConfig().
        circuit_breaker: Optional circuit breaker instance.

    Returns:
        Decorated function with retry logic.

    Example:
        @with_retry(RetryConfig(max_attempts=3))
        async def call_api():
            ...
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            # Check circuit breaker
            if circuit_breaker and not circuit_breaker.can_execute():
                raise CircuitOpenError(
                    f"Circuit breaker is OPEN for {func.__name__}"
                )

            last_exception: Optional[Exception] = None

            for attempt in range(config.max_attempts):
                try:
                    result = await func(*args, **kwargs)

                    # Record success for circuit breaker
                    if circuit_breaker:
                        circuit_breaker.record_success()

                    return result

                except config.retryable_exceptions as e:
                    last_exception = e

                    # Record failure for circuit breaker
                    if circuit_breaker:
                        circuit_breaker.record_failure()
                        if circuit_breaker.state == CircuitState.OPEN:
                            raise CircuitOpenError(
                                f"Circuit breaker opened for {func.__name__}"
                            ) from e

                    if attempt < config.max_attempts - 1:
                        delay = calculate_delay(attempt, config)
                        logger.warning(
                            f"Retry {attempt + 1}/{config.max_attempts} for "
                            f"{func.__name__} after error: {e}. "
                            f"Waiting {delay:.2f}s"
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"All {config.max_attempts} attempts exhausted for "
                            f"{func.__name__}: {e}"
                        )

            raise MaxRetriesExceededError(
                f"Max retries ({config.max_attempts}) exceeded for {func.__name__}",
                last_exception,
            )

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Optional[Exception] = None

            for attempt in range(config.max_attempts):
                try:
                    result = func(*args, **kwargs)
                    if circuit_breaker:
                        circuit_breaker.record_success()
                    return result

                except config.retryable_exceptions as e:
                    last_exception = e
                    if circuit_breaker:
                        circuit_breaker.record_failure()

                    if attempt < config.max_attempts - 1:
                        delay = calculate_delay(attempt, config)
                        logger.warning(
                            f"Retry {attempt + 1}/{config.max_attempts} for "
                            f"{func.__name__} after error: {e}. "
                            f"Waiting {delay:.2f}s"
                        )
                        time.sleep(delay)

            raise MaxRetriesExceededError(
                f"Max retries ({config.max_attempts}) exceeded for {func.__name__}",
                last_exception,
            )

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# Default configurations for common use cases
DEFAULT_API_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    base_delay=1.0,
    max_delay=30.0,
    exponential_base=2.0,
    jitter=True,
)

AGGRESSIVE_RETRY_CONFIG = RetryConfig(
    max_attempts=5,
    base_delay=0.5,
    max_delay=60.0,
    exponential_base=2.0,
    jitter=True,
)
