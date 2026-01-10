"""Metrics tracking for token usage, costs, and timing."""

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class TokenUsage:
    """Token usage for a single API call."""

    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.input_tokens + self.output_tokens


@dataclass
class UsageMetrics:
    """Aggregated usage metrics for a session."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    api_calls: int = 0
    phase_timings: dict[str, float] = field(default_factory=dict)
    usage_by_model: dict[str, TokenUsage] = field(default_factory=dict)
    usage_by_phase: dict[str, TokenUsage] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """Total tokens used across all calls."""
        return self.total_input_tokens + self.total_output_tokens

    def add_usage(
        self,
        usage: TokenUsage,
        phase: Optional[str] = None,
    ) -> None:
        """Add usage from an API call.

        Args:
            usage: Token usage from the call.
            phase: Optional phase name for categorization.
        """
        self.total_input_tokens += usage.input_tokens
        self.total_output_tokens += usage.output_tokens
        self.api_calls += 1

        # Track by model
        if usage.model:
            if usage.model not in self.usage_by_model:
                self.usage_by_model[usage.model] = TokenUsage(model=usage.model)
            self.usage_by_model[usage.model].input_tokens += usage.input_tokens
            self.usage_by_model[usage.model].output_tokens += usage.output_tokens

        # Track by phase
        if phase:
            if phase not in self.usage_by_phase:
                self.usage_by_phase[phase] = TokenUsage()
            self.usage_by_phase[phase].input_tokens += usage.input_tokens
            self.usage_by_phase[phase].output_tokens += usage.output_tokens

    def record_phase_timing(self, phase: str, duration: float) -> None:
        """Record timing for a phase.

        Args:
            phase: Phase name.
            duration: Duration in seconds.
        """
        if phase in self.phase_timings:
            self.phase_timings[phase] += duration
        else:
            self.phase_timings[phase] = duration

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_cost": round(self.total_cost, 6),
            "api_calls": self.api_calls,
            "phase_timings": self.phase_timings,
            "usage_by_model": {
                model: {
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "total_tokens": usage.total_tokens,
                }
                for model, usage in self.usage_by_model.items()
            },
            "usage_by_phase": {
                phase: {
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "total_tokens": usage.total_tokens,
                }
                for phase, usage in self.usage_by_phase.items()
            },
        }


# Pricing per 1M tokens (as of 2026, approximate)
MODEL_PRICING = {
    # OpenAI
    "o3": {"input": 15.00, "output": 60.00},  # Premium reasoning model
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    # Anthropic
    "claude-opus-4-20250514": {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-opus": {"input": 15.00, "output": 75.00},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    # Google
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-3.0-pro": {"input": 1.25, "output": 5.00},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
}

# Fallback pricing for unknown models
DEFAULT_PRICING = {"input": 5.00, "output": 15.00}


class CostEstimator:
    """Estimates costs based on token usage."""

    def __init__(self, pricing: Optional[dict] = None):
        """Initialize with optional custom pricing.

        Args:
            pricing: Custom pricing dict mapping model names to
                     {"input": price_per_1m, "output": price_per_1m}.
        """
        self.pricing = {**MODEL_PRICING, **(pricing or {})}

    def get_pricing(self, model: str) -> dict[str, float]:
        """Get pricing for a model.

        Args:
            model: Model name.

        Returns:
            Dict with "input" and "output" prices per 1M tokens.
        """
        # Try exact match first
        if model in self.pricing:
            return self.pricing[model]

        # Try partial match (for versioned model names)
        for pricing_model, prices in self.pricing.items():
            if pricing_model in model or model in pricing_model:
                return prices

        return DEFAULT_PRICING

    def estimate_cost(self, usage: TokenUsage) -> float:
        """Estimate cost for token usage.

        Args:
            usage: Token usage to price.

        Returns:
            Estimated cost in USD.
        """
        pricing = self.get_pricing(usage.model)

        input_cost = (usage.input_tokens / 1_000_000) * pricing["input"]
        output_cost = (usage.output_tokens / 1_000_000) * pricing["output"]

        return input_cost + output_cost

    def estimate_total_cost(self, metrics: UsageMetrics) -> float:
        """Estimate total cost for all usage in metrics.

        Args:
            metrics: Usage metrics with per-model tracking.

        Returns:
            Total estimated cost in USD.
        """
        total = 0.0

        for model, usage in metrics.usage_by_model.items():
            pricing = self.get_pricing(model)
            total += (usage.input_tokens / 1_000_000) * pricing["input"]
            total += (usage.output_tokens / 1_000_000) * pricing["output"]

        return total


class PhaseTimer:
    """Context manager for timing phases."""

    def __init__(self, metrics: UsageMetrics, phase: str):
        """Initialize timer.

        Args:
            metrics: Metrics object to record timing.
            phase: Phase name.
        """
        self.metrics = metrics
        self.phase = phase
        self.start_time: Optional[float] = None

    def __enter__(self) -> "PhaseTimer":
        """Start timing."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Record timing."""
        if self.start_time:
            duration = time.time() - self.start_time
            self.metrics.record_phase_timing(self.phase, duration)


# Global cost estimator instance
_cost_estimator: Optional[CostEstimator] = None


def get_cost_estimator() -> CostEstimator:
    """Get the global cost estimator instance."""
    global _cost_estimator
    if _cost_estimator is None:
        _cost_estimator = CostEstimator()
    return _cost_estimator
