"""Rate limiting utilities for API protection."""

import time
from dataclasses import dataclass, field
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger()


class RateLimitExceededError(Exception):
    """Raised when rate limit is exceeded."""

    def __init__(self, message: str, retry_after: float):
        super().__init__(message)
        self.retry_after = retry_after


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_size: int = 10  # Max burst allowed


@dataclass
class TokenBucket:
    """Token bucket rate limiter.

    Allows bursts up to bucket capacity while maintaining
    a sustainable average rate.
    """

    capacity: float
    fill_rate: float  # Tokens per second
    tokens: float = field(init=False)
    last_update: float = field(init=False)

    def __post_init__(self):
        """Initialize bucket as full."""
        self.tokens = self.capacity
        self.last_update = time.time()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_update
        self.tokens = min(self.capacity, self.tokens + elapsed * self.fill_rate)
        self.last_update = now

    def consume(self, tokens: float = 1.0) -> bool:
        """Try to consume tokens.

        Args:
            tokens: Number of tokens to consume.

        Returns:
            True if tokens were consumed, False if insufficient.
        """
        self._refill()

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def time_until_available(self, tokens: float = 1.0) -> float:
        """Calculate time until tokens are available.

        Args:
            tokens: Number of tokens needed.

        Returns:
            Seconds until tokens are available.
        """
        self._refill()

        if self.tokens >= tokens:
            return 0.0

        needed = tokens - self.tokens
        return needed / self.fill_rate


@dataclass
class SlidingWindowCounter:
    """Sliding window counter for more accurate rate limiting."""

    window_size: float  # Window size in seconds
    max_requests: int
    timestamps: list[float] = field(default_factory=list)

    def _clean_old(self) -> None:
        """Remove timestamps outside the window."""
        now = time.time()
        cutoff = now - self.window_size
        self.timestamps = [ts for ts in self.timestamps if ts > cutoff]

    def check_and_increment(self) -> bool:
        """Check if request is allowed and increment counter.

        Returns:
            True if request is allowed, False if rate limited.
        """
        self._clean_old()

        if len(self.timestamps) < self.max_requests:
            self.timestamps.append(time.time())
            return True
        return False

    def time_until_available(self) -> float:
        """Calculate time until a request is available.

        Returns:
            Seconds until the oldest request expires from window.
        """
        self._clean_old()

        if len(self.timestamps) < self.max_requests:
            return 0.0

        oldest = min(self.timestamps)
        return (oldest + self.window_size) - time.time()


class RateLimiter:
    """Rate limiter combining token bucket and sliding window."""

    def __init__(self, config: Optional[RateLimitConfig] = None):
        """Initialize rate limiter.

        Args:
            config: Rate limit configuration.
        """
        self.config = config or RateLimitConfig()

        # Token bucket for burst control
        self.bucket = TokenBucket(
            capacity=float(self.config.burst_size),
            fill_rate=self.config.requests_per_minute / 60.0,
        )

        # Sliding window for minute rate
        self.minute_window = SlidingWindowCounter(
            window_size=60.0,
            max_requests=self.config.requests_per_minute,
        )

        # Sliding window for hour rate
        self.hour_window = SlidingWindowCounter(
            window_size=3600.0,
            max_requests=self.config.requests_per_hour,
        )

    def check(self, raise_on_limit: bool = True) -> bool:
        """Check if request is allowed.

        Args:
            raise_on_limit: Whether to raise exception on limit.

        Returns:
            True if allowed.

        Raises:
            RateLimitExceededError: If rate limited and raise_on_limit is True.
        """
        # Check token bucket first (for bursts)
        if not self.bucket.consume():
            retry_after = self.bucket.time_until_available()
            if raise_on_limit:
                raise RateLimitExceededError(
                    "Burst rate limit exceeded",
                    retry_after,
                )
            return False

        # Check minute window
        if not self.minute_window.check_and_increment():
            retry_after = self.minute_window.time_until_available()
            if raise_on_limit:
                raise RateLimitExceededError(
                    f"Rate limit exceeded: {self.config.requests_per_minute}/minute",
                    retry_after,
                )
            return False

        # Check hour window
        if not self.hour_window.check_and_increment():
            retry_after = self.hour_window.time_until_available()
            if raise_on_limit:
                raise RateLimitExceededError(
                    f"Rate limit exceeded: {self.config.requests_per_hour}/hour",
                    retry_after,
                )
            return False

        return True


class SessionRateLimiter:
    """Rate limiter that tracks per-session and global limits."""

    def __init__(
        self,
        per_session_config: Optional[RateLimitConfig] = None,
        global_config: Optional[RateLimitConfig] = None,
    ):
        """Initialize session rate limiter.

        Args:
            per_session_config: Rate limits per session.
            global_config: Global rate limits across all sessions.
        """
        self.per_session_config = per_session_config or RateLimitConfig(
            requests_per_minute=30,
            requests_per_hour=500,
            burst_size=5,
        )
        self.global_config = global_config or RateLimitConfig(
            requests_per_minute=100,
            requests_per_hour=3000,
            burst_size=20,
        )

        self.session_limiters: dict[str, RateLimiter] = {}
        self.global_limiter = RateLimiter(self.global_config)

    def _get_session_limiter(self, session_id: str) -> RateLimiter:
        """Get or create rate limiter for a session."""
        if session_id not in self.session_limiters:
            self.session_limiters[session_id] = RateLimiter(self.per_session_config)
        return self.session_limiters[session_id]

    def check(self, session_id: str, raise_on_limit: bool = True) -> bool:
        """Check if request is allowed for a session.

        Args:
            session_id: The session ID.
            raise_on_limit: Whether to raise exception on limit.

        Returns:
            True if allowed.

        Raises:
            RateLimitExceededError: If rate limited and raise_on_limit is True.
        """
        # Check global limit first
        if not self.global_limiter.check(raise_on_limit=False):
            retry_after = max(
                self.global_limiter.bucket.time_until_available(),
                self.global_limiter.minute_window.time_until_available(),
            )
            if raise_on_limit:
                raise RateLimitExceededError(
                    "Global rate limit exceeded",
                    retry_after,
                )
            return False

        # Check session limit
        session_limiter = self._get_session_limiter(session_id)
        return session_limiter.check(raise_on_limit)

    def cleanup_session(self, session_id: str) -> None:
        """Remove a session's rate limiter.

        Args:
            session_id: The session ID to clean up.
        """
        if session_id in self.session_limiters:
            del self.session_limiters[session_id]
            logger.debug(f"Cleaned up rate limiter for session {session_id}")


# Global rate limiter instance
_session_rate_limiter: Optional[SessionRateLimiter] = None


def get_rate_limiter() -> SessionRateLimiter:
    """Get the global session rate limiter."""
    global _session_rate_limiter
    if _session_rate_limiter is None:
        _session_rate_limiter = SessionRateLimiter()
    return _session_rate_limiter
