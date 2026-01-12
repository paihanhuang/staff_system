"""Tests for conversation continuity (storage and follow-up)."""

import asyncio
import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models import GraphState, ArchitectureDecisionRecord, ConversationTurn
from src.models.proposal import ArchitectureProposal
from src.utils.storage import SessionData, SessionStorage


class TestSessionStorage:
    """Tests for file-based session storage."""

    @pytest.fixture
    def temp_storage_dir(self, tmp_path):
        """Create a temporary storage directory."""
        return tmp_path / "sessions"

    @pytest.fixture
    def storage(self, temp_storage_dir):
        """Create a storage instance with temp directory."""
        return SessionStorage(storage_dir=temp_storage_dir)

    @pytest.fixture
    def sample_session_data(self):
        """Create sample session data."""
        return SessionData(
            session_id="test-123",
            question="Design a cache system",
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            state_json={
                "session_id": "test-123",
                "user_question": "Design a cache system",
                "current_phase": "complete",
            },
            conversation_turns=[],
            is_complete=True,
        )

    @pytest.mark.asyncio
    async def test_save_and_load_session(self, storage, sample_session_data):
        """Test saving and loading a session."""
        # Save
        await storage.save_session(sample_session_data)
        
        # Verify file exists
        path = storage._session_path(sample_session_data.session_id)
        assert path.exists()
        
        # Load
        loaded = await storage.load_session(sample_session_data.session_id)
        assert loaded is not None
        assert loaded.session_id == sample_session_data.session_id
        assert loaded.question == sample_session_data.question
        assert loaded.is_complete == sample_session_data.is_complete

    @pytest.mark.asyncio
    async def test_load_nonexistent_session(self, storage):
        """Test loading a session that doesn't exist."""
        loaded = await storage.load_session("nonexistent")
        assert loaded is None

    @pytest.mark.asyncio
    async def test_delete_session(self, storage, sample_session_data):
        """Test deleting a session."""
        await storage.save_session(sample_session_data)
        
        # Delete
        result = await storage.delete_session(sample_session_data.session_id)
        assert result is True
        
        # Verify deleted
        loaded = await storage.load_session(sample_session_data.session_id)
        assert loaded is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_session(self, storage):
        """Test deleting a session that doesn't exist."""
        result = await storage.delete_session("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_list_sessions(self, storage, sample_session_data):
        """Test listing sessions."""
        # Start empty
        sessions = await storage.list_sessions()
        assert len(sessions) == 0
        
        # Add one
        await storage.save_session(sample_session_data)
        sessions = await storage.list_sessions()
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == sample_session_data.session_id


class TestConversationTurn:
    """Tests for ConversationTurn model."""

    @pytest.fixture
    def sample_adr(self):
        """Create a sample ADR."""
        return ArchitectureDecisionRecord(
            title="Cache System Design",
            decision="Use Redis for caching",
            rationale="Redis provides high performance",
            context="Need a fast cache",
            majority_opinion=ArchitectureProposal(
                title="Redis Cache",
                summary="Use Redis",
                confidence=0.9,
                approach="Deploy Redis cluster",
                components=[],
                trade_offs=[],
                risks=[],
            ),
            consensus_level=0.85,
            rounds_taken=1,
        )

    def test_conversation_turn_creation(self, sample_adr):
        """Test creating a conversation turn."""
        turn = ConversationTurn(
            turn_id=1,
            question="Design a cache system",
            adr=sample_adr,
        )
        assert turn.turn_id == 1
        assert turn.question == "Design a cache system"
        assert turn.adr.title == "Cache System Design"


class TestGraphStateFollowUp:
    """Tests for follow-up fields in GraphState."""

    def test_default_not_follow_up(self):
        """Test that new state is not a follow-up by default."""
        state = GraphState(
            session_id="test",
            user_question="Design something",
        )
        assert state.is_follow_up is False
        assert state.follow_up_context is None
        assert state.conversation_turns == []

    def test_follow_up_state(self):
        """Test state with follow-up context."""
        state = GraphState(
            session_id="test",
            user_question="How to scale?",
            is_follow_up=True,
            follow_up_context="Previous design: Cache System",
        )
        assert state.is_follow_up is True
        assert state.follow_up_context == "Previous design: Cache System"
