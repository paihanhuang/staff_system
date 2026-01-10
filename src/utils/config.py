"""Configuration management using pydantic-settings."""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API Keys
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None

    # Model Configuration
    supervisor_model: str = "gpt-4o"
    architect_model: str = "gpt-4o"
    engineer_model: str = "claude-sonnet-4-20250514"
    auditor_model: str = "gemini-2.0-flash"

    # Application Settings
    max_rounds: int = 3
    debug: bool = False
    log_level: str = "INFO"

    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Timeouts (in seconds)
    model_timeout: int = 120
    request_timeout: int = 300

    # Retry Settings
    retry_max_attempts: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 30.0

    # Rate Limiting
    rate_limit_per_session_per_minute: int = 30
    rate_limit_per_session_per_hour: int = 500
    rate_limit_global_per_minute: int = 100
    rate_limit_global_per_hour: int = 3000

    # Fallback Models (comma-separated fallback chains)
    architect_fallback_models: str = "gpt-4o-mini,gpt-4-turbo"
    engineer_fallback_models: str = "claude-sonnet-4-20250514,claude-3-5-sonnet-20241022"
    auditor_fallback_models: str = "gemini-1.5-pro,gemini-1.5-flash"

    def validate_api_keys(self) -> dict[str, bool]:
        """Check which API keys are configured."""
        return {
            "openai": bool(self.openai_api_key),
            "anthropic": bool(self.anthropic_api_key),
            "google": bool(self.google_api_key),
        }


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
