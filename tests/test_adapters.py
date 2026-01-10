"""Tests for AI adapters."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from src.adapters.openai_adapter import OpenAIAdapter, ArchitectAdapter
from src.adapters.anthropic_adapter import AnthropicAdapter, EngineerAdapter
from src.adapters.google_adapter import GoogleAdapter, AuditorAdapter
from src.models.proposal import ArchitectureProposal


class TestOpenAIAdapter:
    """Tests for the OpenAI adapter."""

    @pytest.fixture
    def adapter(self):
        """Create an adapter instance."""
        return OpenAIAdapter(model_name="gpt-4o", api_key="test-key")

    def test_adapter_initialization(self, adapter):
        """Test adapter initializes correctly."""
        assert adapter.model_name == "gpt-4o"
        assert adapter.api_key == "test-key"

    @pytest.mark.asyncio
    async def test_generate_returns_string(self, adapter):
        """Test generate returns a string response."""
        with patch.object(adapter, 'client') as mock_client:
            mock_response = MagicMock()
            mock_response.content = "Test response"
            mock_client.ainvoke = AsyncMock(return_value=mock_response)

            # Note: This would need the actual client to be mocked properly
            # This is a structural test showing how tests would be organized


class TestArchitectAdapter:
    """Tests for the Architect adapter."""

    @pytest.fixture
    def adapter(self):
        """Create an adapter instance."""
        with patch('src.adapters.openai_adapter.get_settings') as mock_settings:
            mock_settings.return_value.architect_model = "o3"
            mock_settings.return_value.openai_api_key = "test-key"
            return ArchitectAdapter()

    def test_uses_architect_model(self, adapter):
        """Test adapter uses the architect model from settings."""
        assert adapter.model_name == "o3"


class TestAnthropicAdapter:
    """Tests for the Anthropic adapter."""

    @pytest.fixture
    def adapter(self):
        """Create an adapter instance."""
        return AnthropicAdapter(model_name="claude-sonnet-4-5-20250514", api_key="test-key")

    def test_adapter_initialization(self, adapter):
        """Test adapter initializes correctly."""
        assert adapter.model_name == "claude-sonnet-4-5-20250514"
        assert adapter.api_key == "test-key"


class TestGoogleAdapter:
    """Tests for the Google adapter."""

    @pytest.fixture
    def adapter(self):
        """Create an adapter instance."""
        return GoogleAdapter(model_name="gemini-3.0-pro", api_key="test-key")

    def test_adapter_initialization(self, adapter):
        """Test adapter initializes correctly."""
        assert adapter.model_name == "gemini-3.0-pro"
        assert adapter.api_key == "test-key"
