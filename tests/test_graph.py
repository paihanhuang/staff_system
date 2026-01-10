"""Tests for the LangGraph state machine."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, patch

from src.models import GraphState, SystemContext, ArchitectureProposal, Critique, AuditResult
from src.graph.edges import (
    route_after_ideation,
    route_after_audit,
    route_after_clarification,
    should_continue,
)


class TestGraphState:
    """Tests for GraphState model."""

    def test_create_initial_state(self):
        """Test creating an initial state."""
        state = GraphState(
            session_id="test-123",
            user_question="Design a distributed cache",
        )

        assert state.session_id == "test-123"
        assert state.user_question == "Design a distributed cache"
        assert state.current_phase == "start"
        assert state.round_number == 0
        assert state.max_rounds == 3
        assert not state.consensus_reached
        assert state.architect_proposal is None
        assert state.engineer_proposal is None

    def test_state_with_context(self):
        """Test creating state with system context."""
        context = SystemContext(
            company_name="Test Corp",
            domain="fintech",
            current_tech_stack=["Python", "PostgreSQL"],
        )

        state = GraphState(
            session_id="test-456",
            user_question="Design a payment system",
            system_context=context,
        )

        assert state.system_context.company_name == "Test Corp"
        assert state.system_context.domain == "fintech"


class TestRouting:
    """Tests for routing logic."""

    def test_route_after_ideation_no_interrupt(self):
        """Test routing after ideation without interrupt."""
        state = GraphState(
            session_id="test",
            user_question="test",
            interrupt=None,
        )

        result = route_after_ideation(state)
        assert result == "cross_critique"

    def test_route_after_audit_consensus(self):
        """Test routing when consensus is reached."""
        state = GraphState(
            session_id="test",
            user_question="test",
            consensus_reached=True,
            round_number=1,
            max_rounds=3,
        )

        result = route_after_audit(state)
        assert result == "convergence"

    def test_route_after_audit_max_rounds(self):
        """Test routing when max rounds exceeded."""
        state = GraphState(
            session_id="test",
            user_question="test",
            consensus_reached=False,
            round_number=3,
            max_rounds=3,
        )

        result = route_after_audit(state)
        assert result == "escalate"

    def test_route_after_clarification_from_ideation(self):
        """Test resuming from ideation phase."""
        state = GraphState(
            session_id="test",
            user_question="test",
            current_phase="ideation_complete",
        )

        result = route_after_clarification(state)
        assert result == "cross_critique"


class TestShouldContinue:
    """Tests for should_continue logic."""

    def test_continue_when_processing(self):
        """Test continuing when still processing."""
        state = GraphState(
            session_id="test",
            user_question="test",
        )

        assert should_continue(state) is True

    def test_stop_when_complete(self):
        """Test stopping when ADR is complete."""
        proposal = ArchitectureProposal(
            title="Test",
            summary="Test summary",
            approach="Test approach",
            confidence=0.8,
        )

        from src.models import ArchitectureDecisionRecord

        adr = ArchitectureDecisionRecord(
            title="Test ADR",
            decision="Test decision",
            rationale="Test rationale",
            context="Test context",
            majority_opinion=proposal,
            consensus_level=0.8,
            rounds_taken=2,
        )

        state = GraphState(
            session_id="test",
            user_question="test",
            final_adr=adr,
        )

        assert should_continue(state) is False

    def test_stop_on_error(self):
        """Test stopping on error."""
        state = GraphState(
            session_id="test",
            user_question="test",
            error="Something went wrong",
        )

        assert should_continue(state) is False
