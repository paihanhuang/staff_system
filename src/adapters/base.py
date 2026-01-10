"""Base adapter for AI models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Type, TypeVar

from pydantic import BaseModel

from src.utils.logger import get_logger
from src.utils.metrics import TokenUsage
from src.utils.resilience import RetryConfig

T = TypeVar("T", bound=BaseModel)


@dataclass
class AdapterResponse:
    """Response from an adapter including content and usage metrics."""

    content: Any
    usage: TokenUsage
    model_used: str
    was_fallback: bool = False


class BaseAdapter(ABC):
    """Abstract base class for AI model adapters."""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        fallback_models: Optional[list[str]] = None,
        retry_config: Optional[RetryConfig] = None,
    ):
        """Initialize the adapter.

        Args:
            model_name: Name of the primary model to use.
            api_key: API key for authentication (if not using env var).
            fallback_models: List of fallback model names to try on failure.
            retry_config: Configuration for retry behavior.
        """
        self.model_name = model_name
        self.api_key = api_key
        self.fallback_models = fallback_models or []
        self.retry_config = retry_config
        self.logger = get_logger()

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AdapterResponse:
        """Generate a response from the model.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            AdapterResponse with content and usage metrics.
        """
        pass

    @abstractmethod
    async def generate_structured(
        self,
        prompt: str,
        response_model: Type[T],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> T:
        """Generate a structured response that conforms to a Pydantic model.

        Args:
            prompt: The user prompt.
            response_model: Pydantic model class for the response.
            system_prompt: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            An instance of the response_model.
        """
        pass

    @abstractmethod
    async def generate_structured_with_usage(
        self,
        prompt: str,
        response_model: Type[T],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> tuple[T, TokenUsage]:
        """Generate structured response with usage tracking.

        Args:
            prompt: The user prompt.
            response_model: Pydantic model class for the response.
            system_prompt: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            Tuple of (response_model instance, TokenUsage).
        """
        pass

    def get_model_chain(self) -> list[str]:
        """Get the full model chain including fallbacks.

        Returns:
            List of model names to try in order.
        """
        return [self.model_name] + self.fallback_models

    def _log_request(self, prompt: str, system_prompt: Optional[str] = None) -> None:
        """Log an API request."""
        self.logger.debug(
            f"[{self.__class__.__name__}] Request to {self.model_name}: "
            f"prompt_len={len(prompt)}, system_len={len(system_prompt or '')}"
        )

    def _log_response(
        self,
        response: Any,
        usage: Optional[TokenUsage] = None,
    ) -> None:
        """Log an API response with usage metrics."""
        response_len = len(str(response)) if response else 0
        usage_info = ""
        if usage:
            usage_info = (
                f", tokens_in={usage.input_tokens}, "
                f"tokens_out={usage.output_tokens}"
            )
        self.logger.debug(
            f"[{self.__class__.__name__}] Response from {self.model_name}: "
            f"response_len={response_len}{usage_info}"
        )

    def _log_fallback(self, from_model: str, to_model: str, reason: str) -> None:
        """Log a fallback to another model."""
        self.logger.warning(
            f"[{self.__class__.__name__}] Falling back from {from_model} to "
            f"{to_model}: {reason}"
        )

