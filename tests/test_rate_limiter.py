"""Tests for rate limiter utilities."""

import time
import pytest

from src.api.rate_limiter import (
    RateLimitConfig,
    TokenBucket,
    SlidingWindowCounter,
    RateLimiter,
    SessionRateLimiter,
    RateLimitExceededError,
)


class TestTokenBucket:
    """Tests for TokenBucket."""

    def test_initial_bucket_is_full(self):
        """Test bucket starts at capacity."""
        bucket = TokenBucket(capacity=10.0, fill_rate=1.0)
        
        assert bucket.tokens == 10.0

    def test_consume_reduces_tokens(self):
        """Test consuming tokens reduces count."""
        bucket = TokenBucket(capacity=10.0, fill_rate=1.0)
        
        result = bucket.consume(5.0)
        
        assert result is True
        assert bucket.tokens == 5.0

    def test_consume_fails_insufficient_tokens(self):
        """Test consuming more than available fails."""
        bucket = TokenBucket(capacity=5.0, fill_rate=1.0)
        
        result = bucket.consume(10.0)
        
        assert result is False

    def test_tokens_refill_over_time(self):
        """Test tokens refill based on time."""
        bucket = TokenBucket(capacity=10.0, fill_rate=100.0)  # Fast refill
        bucket.consume(5.0)
        
        time.sleep(0.1)  # Wait for refill
        bucket._refill()
        
        assert bucket.tokens > 5.0

    def test_time_until_available(self):
        """Test time calculation for token availability."""
        bucket = TokenBucket(capacity=10.0, fill_rate=1.0)
        bucket.tokens = 0.0
        bucket.last_update = time.time()
        
        wait_time = bucket.time_until_available(5.0)
        
        assert 4.9 <= wait_time <= 5.1


class TestSlidingWindowCounter:
    """Tests for SlidingWindowCounter."""

    def test_allows_requests_under_limit(self):
        """Test requests under limit are allowed."""
        counter = SlidingWindowCounter(window_size=60.0, max_requests=10)
        
        for _ in range(5):
            assert counter.check_and_increment() is True
        
        assert len(counter.timestamps) == 5

    def test_rejects_requests_over_limit(self):
        """Test requests over limit are rejected."""
        counter = SlidingWindowCounter(window_size=60.0, max_requests=2)
        
        counter.check_and_increment()
        counter.check_and_increment()
        
        assert counter.check_and_increment() is False

    def test_old_requests_expire(self):
        """Test old requests are cleaned from window."""
        counter = SlidingWindowCounter(window_size=0.1, max_requests=2)
        
        counter.check_and_increment()
        counter.check_and_increment()
        
        time.sleep(0.15)
        
        # Old requests should have expired
        assert counter.check_and_increment() is True


class TestRateLimiter:
    """Tests for RateLimiter."""

    def test_allows_requests_under_limit(self):
        """Test requests under limit pass."""
        config = RateLimitConfig(
            requests_per_minute=60,
            requests_per_hour=1000,
            burst_size=10,
        )
        limiter = RateLimiter(config)
        
        for _ in range(5):
            assert limiter.check(raise_on_limit=False) is True

    def test_raises_on_burst_exceeded(self):
        """Test RateLimitExceededError on burst limit."""
        config = RateLimitConfig(burst_size=2)
        limiter = RateLimiter(config)
        
        limiter.check()
        limiter.check()
        
        with pytest.raises(RateLimitExceededError):
            limiter.check()

    def test_rate_limit_error_has_retry_after(self):
        """Test RateLimitExceededError contains retry_after."""
        config = RateLimitConfig(burst_size=1)
        limiter = RateLimiter(config)
        
        limiter.check()
        
        with pytest.raises(RateLimitExceededError) as exc_info:
            limiter.check()
        
        assert exc_info.value.retry_after > 0


class TestSessionRateLimiter:
    """Tests for SessionRateLimiter."""

    def test_separate_limits_per_session(self):
        """Test each session has independent limits."""
        limiter = SessionRateLimiter(
            per_session_config=RateLimitConfig(burst_size=2),
        )
        
        limiter.check("session_1")
        limiter.check("session_1")
        limiter.check("session_2")  # Different session
        
        with pytest.raises(RateLimitExceededError):
            limiter.check("session_1")  # Session 1 exceeded
        
        # Session 2 still has capacity
        assert limiter.check("session_2", raise_on_limit=False) is True

    def test_global_limit_applies_to_all(self):
        """Test global limit affects all sessions."""
        limiter = SessionRateLimiter(
            per_session_config=RateLimitConfig(burst_size=10),
            global_config=RateLimitConfig(burst_size=2),
        )
        
        limiter.check("session_1")
        limiter.check("session_2")
        
        # Global limit reached
        with pytest.raises(RateLimitExceededError):
            limiter.check("session_3")

    def test_cleanup_removes_session(self):
        """Test cleanup_session removes rate limiter."""
        limiter = SessionRateLimiter()
        
        limiter.check("session_1")
        assert "session_1" in limiter.session_limiters
        
        limiter.cleanup_session("session_1")
        assert "session_1" not in limiter.session_limiters
