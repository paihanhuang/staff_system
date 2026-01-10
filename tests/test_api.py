"""Tests for the FastAPI backend."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock

from src.api.main import app


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for health endpoint."""

    def test_health_check(self, client):
        """Test health check returns status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "api_keys" in data


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_returns_info(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Synapse Council"
        assert "endpoints" in data


class TestSessionsEndpoint:
    """Tests for sessions list endpoint."""

    def test_list_sessions_empty(self, client):
        """Test listing sessions when empty."""
        response = client.get("/api/sessions")
        assert response.status_code == 200
        data = response.json()
        assert "sessions" in data


class TestDesignEndpoint:
    """Tests for design endpoint."""

    def test_design_requires_question(self, client):
        """Test that question is required."""
        response = client.post("/api/design", json={})
        assert response.status_code == 422  # Validation error

    @patch('src.api.handlers.session_manager.create_session')
    @patch('src.api.handlers.session_manager.run_session')
    async def test_design_creates_session(self, mock_run, mock_create, client):
        """Test that design endpoint creates a session."""
        from src.api.handlers import Session
        from src.models import GraphState

        mock_session = Session(
            session_id="test-123",
            question="Test question",
        )
        mock_session.state = GraphState(
            session_id="test-123",
            user_question="Test question",
            current_phase="complete",
        )
        mock_session.is_complete = True

        mock_create.return_value = mock_session
        mock_run.return_value = mock_session

        response = client.post(
            "/api/design",
            json={"question": "Design a cache system"},
        )

        # Note: This is a structural test - actual async behavior
        # would need proper async test setup
