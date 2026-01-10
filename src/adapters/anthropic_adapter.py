"""Anthropic adapter for Claude models."""

from typing import Optional, Type, TypeVar

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from src.adapters.base import AdapterResponse, BaseAdapter
from src.utils.config import get_settings
from src.utils.metrics import TokenUsage
from src.utils.resilience import (
    RetryConfig,
    with_retry,
    CircuitBreaker,
    CircuitBreakerConfig,
    MaxRetriesExceededError,
)

T = TypeVar("T", bound=BaseModel)


class AnthropicAdapter(BaseAdapter):
    """Adapter for Anthropic Claude models."""

    # Shared circuit breaker for Anthropic API
    _circuit_breaker = CircuitBreaker(CircuitBreakerConfig(
        failure_threshold=5,
        recovery_timeout=60.0,
    ))

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        fallback_models: Optional[list[str]] = None,
    ):
        """Initialize the Anthropic adapter.

        Args:
            model_name: Model name (defaults to settings).
            api_key: API key (defaults to settings).
            fallback_models: List of fallback model names.
        """
        settings = get_settings()

        retry_config = RetryConfig(
            max_attempts=settings.retry_max_attempts,
            base_delay=settings.retry_base_delay,
            max_delay=settings.retry_max_delay,
        )

        super().__init__(
            model_name=model_name or settings.engineer_model,
            api_key=api_key or settings.anthropic_api_key,
            fallback_models=fallback_models,
            retry_config=retry_config,
        )
        self._clients: dict[str, ChatAnthropic] = {}

    def _get_client(self, model: str, temperature: float, max_tokens: int) -> ChatAnthropic:
        """Get or create a ChatAnthropic client for a model."""
        key = f"{model}_{temperature}_{max_tokens}"
        if key not in self._clients:
            self._clients[key] = ChatAnthropic(
                model=model,
                api_key=self.api_key,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        return self._clients[key]

    def _extract_usage(self, response, model: str) -> TokenUsage:
        """Extract token usage from Anthropic response."""
        usage = TokenUsage(model=model)

        if hasattr(response, "response_metadata"):
            metadata = response.response_metadata
            if "usage" in metadata:
                token_usage = metadata["usage"]
                usage.input_tokens = token_usage.get("input_tokens", 0)
                usage.output_tokens = token_usage.get("output_tokens", 0)

        return usage

    async def _call_with_fallback(self, func, *args, **kwargs):
        """Call a function with fallback to alternative models."""
        models = self.get_model_chain()
        last_error = None

        for i, model in enumerate(models):
            is_fallback = i > 0
            if is_fallback:
                self._log_fallback(models[i - 1], model, str(last_error))

            try:
                return await func(model, *args, **kwargs), model, is_fallback
            except Exception as e:
                last_error = e
                if i == len(models) - 1:
                    raise MaxRetriesExceededError(
                        f"All models failed: {models}", last_error
                    )

        raise last_error

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AdapterResponse:
        """Generate a response from Claude with retry and fallback."""
        self._log_request(prompt, system_prompt)

        @with_retry(self.retry_config, self._circuit_breaker)
        async def _generate(model: str) -> tuple:
            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(HumanMessage(content=prompt))

            client = self._get_client(model, temperature, max_tokens)
            response = await client.ainvoke(messages)
            usage = self._extract_usage(response, model)
            return response.content, usage

        try:
            result, model_used, was_fallback = await self._call_with_fallback(_generate)
            content, usage = result
            usage.model = model_used

            self._log_response(content, usage)

            return AdapterResponse(
                content=content,
                usage=usage,
                model_used=model_used,
                was_fallback=was_fallback,
            )
        except Exception as e:
            self.logger.error(f"Anthropic generate failed: {e}")
            raise

    async def generate_structured(
        self,
        prompt: str,
        response_model: Type[T],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> T:
        """Generate a structured response that conforms to a Pydantic model."""
        result, _ = await self.generate_structured_with_usage(
            prompt, response_model, system_prompt, temperature, max_tokens
        )
        return result

    async def generate_structured_with_usage(
        self,
        prompt: str,
        response_model: Type[T],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> tuple[T, TokenUsage]:
        """Generate structured response with usage tracking."""
        self._log_request(prompt, system_prompt)

        @with_retry(self.retry_config, self._circuit_breaker)
        async def _generate_structured(model: str) -> tuple:
            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(HumanMessage(content=prompt))

            client = self._get_client(model, temperature, max_tokens)
            structured_client = client.with_structured_output(response_model)
            result = await structured_client.ainvoke(messages)

            usage = TokenUsage(
                model=model,
                input_tokens=len(prompt) // 4,
                output_tokens=len(str(result)) // 4,
            )
            return result, usage

        try:
            result_tuple, model_used, was_fallback = await self._call_with_fallback(
                _generate_structured
            )
            result, usage = result_tuple
            usage.model = model_used

            self._log_response(result, usage)
            return result, usage
        except Exception as e:
            self.logger.error(f"Anthropic generate_structured failed: {e}")
            raise


class EngineerAdapter(AnthropicAdapter):
    """Adapter specifically for The Engineer (Claude 4.5 Sonnet)."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Engineer adapter."""
        settings = get_settings()
        fallbacks = [
            m.strip() for m in settings.engineer_fallback_models.split(",")
            if m.strip()
        ]
        super().__init__(
            model_name=settings.engineer_model,
            api_key=api_key,
            fallback_models=fallbacks,
        )


class ArchitectAdapter(AnthropicAdapter):
    """Adapter for The Architect using Anthropic/Claude."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Architect adapter."""
        settings = get_settings()
        fallbacks = [
            m.strip() for m in settings.architect_fallback_models.split(",")
            if m.strip()
        ]
        super().__init__(
            model_name=settings.architect_model,
            api_key=api_key,
            fallback_models=fallbacks,
        )


class AuditorAdapter(AnthropicAdapter):
    """Adapter for The Auditor using Anthropic/Claude."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Auditor adapter."""
        settings = get_settings()
        fallbacks = [
            m.strip() for m in settings.auditor_fallback_models.split(",")
            if m.strip()
        ]
        super().__init__(
            model_name=settings.auditor_model,
            api_key=api_key,
            fallback_models=fallbacks,
        )


class SupervisorAdapter(AnthropicAdapter):
    """Adapter for The Supervisor using Anthropic/Claude."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Supervisor adapter."""
        settings = get_settings()
        super().__init__(
            model_name=settings.supervisor_model,
            api_key=api_key,
            fallback_models=["claude-sonnet-4-20250514", "claude-3-5-sonnet-20241022"],
        )

