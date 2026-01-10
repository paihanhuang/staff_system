"""Utility modules for the Synapse Council."""

from src.utils.config import Settings, get_settings
from src.utils.logger import get_logger, setup_logging
from src.utils.resilience import (
    RetryConfig,
    CircuitBreaker,
    CircuitBreakerConfig,
    with_retry,
    CircuitOpenError,
    MaxRetriesExceededError,
    DEFAULT_API_RETRY_CONFIG,
)
from src.utils.sanitization import (
    sanitize_user_input,
    validate_question,
    sanitize_and_validate,
    SanitizationError,
    SanitizationResult,
)
from src.utils.metrics import (
    TokenUsage,
    UsageMetrics,
    CostEstimator,
    PhaseTimer,
    get_cost_estimator,
)

__all__ = [
    # Config
    "Settings",
    "get_settings",
    # Logging
    "get_logger",
    "setup_logging",
    # Resilience
    "RetryConfig",
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "with_retry",
    "CircuitOpenError",
    "MaxRetriesExceededError",
    "DEFAULT_API_RETRY_CONFIG",
    # Sanitization
    "sanitize_user_input",
    "validate_question",
    "sanitize_and_validate",
    "SanitizationError",
    "SanitizationResult",
    # Metrics
    "TokenUsage",
    "UsageMetrics",
    "CostEstimator",
    "PhaseTimer",
    "get_cost_estimator",
]
