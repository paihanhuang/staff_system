"""Session and state management for the API."""

import asyncio
from datetime import datetime
from typing import Optional
from uuid import uuid4

from src.models import GraphState, SystemContext
from src.graph.graph import run_graph_with_interrupt, resume_graph
from src.utils.logger import SessionLogger, get_logger

logger = get_logger()


class Session:
    """A user session for a Synapse Council run."""

    def __init__(
        self,
        session_id: str,
        question: str,
        system_context: Optional[SystemContext] = None,
    ):
        """Initialize a session.

        Args:
            session_id: Unique session identifier.
            question: The user's system design question.
            system_context: Optional system context.
        """
        self.session_id = session_id
        self.question = question
        self.system_context = system_context
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.state: Optional[GraphState] = None
        self.is_complete = False
        self.is_waiting_for_input = False
        self.logger = SessionLogger(session_id)

    def update_state(self, state: GraphState) -> None:
        """Update the session state."""
        self.state = state
        self.updated_at = datetime.now()
        self.is_waiting_for_input = state.interrupt is not None
        self.is_complete = state.final_adr is not None or state.error is not None


class SessionManager:
    """Manages active sessions."""

    def __init__(self):
        """Initialize the session manager."""
        self._sessions: dict[str, Session] = {}
        self._lock = asyncio.Lock()

    async def create_session(
        self,
        question: str,
        system_context: Optional[SystemContext] = None,
        session_id: Optional[str] = None,
    ) -> Session:
        """Create a new session.

        Args:
            question: The user's system design question.
            system_context: Optional system context.
            session_id: Optional session ID (generated if not provided).

        Returns:
            The new session.
        """
        session_id = session_id or str(uuid4())[:8]

        async with self._lock:
            if session_id in self._sessions:
                raise ValueError(f"Session {session_id} already exists")

            session = Session(
                session_id=session_id,
                question=question,
                system_context=system_context,
            )
            self._sessions[session_id] = session
            logger.info(f"Created session {session_id}")

        return session

    async def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID.

        Args:
            session_id: The session ID.

        Returns:
            The session, or None if not found.
        """
        return self._sessions.get(session_id)

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session.

        Args:
            session_id: The session ID.

        Returns:
            True if deleted, False if not found.
        """
        async with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.info(f"Deleted session {session_id}")
                return True
            return False

    async def run_session(self, session_id: str) -> Session:
        """Run the graph for a session.

        Args:
            session_id: The session ID.

        Returns:
            Updated session.

        Raises:
            ValueError: If session not found.
        """
        session = await self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        session.logger.log_event("session_started", message=session.question[:100])

        state, is_complete = await run_graph_with_interrupt(
            question=session.question,
            system_context=session.system_context,
            session_id=session_id,
        )

        session.update_state(state)
        session.is_complete = is_complete

        if state.interrupt:
            session.logger.log_interrupt(
                agent=state.interrupt.source,
                interrupt_type=state.interrupt.type.value,
                question=state.interrupt.question,
            )

        return session

    async def resume_session(self, session_id: str, user_response: str) -> Session:
        """Resume a paused session with user input.

        Args:
            session_id: The session ID.
            user_response: The user's response.

        Returns:
            Updated session.

        Raises:
            ValueError: If session not found or not waiting for input.
        """
        session = await self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        if not session.is_waiting_for_input:
            raise ValueError(f"Session {session_id} is not waiting for input")

        session.logger.log_user_response(user_response)

        state, is_complete = await resume_graph(
            state=session.state,
            user_response=user_response,
        )

        session.update_state(state)
        session.is_complete = is_complete

        if state.interrupt:
            session.logger.log_interrupt(
                agent=state.interrupt.source,
                interrupt_type=state.interrupt.type.value,
                question=state.interrupt.question,
            )
        elif is_complete:
            session.logger.log_consensus(
                reached=state.consensus_reached,
                rounds=state.round_number,
            )

        return session

    def list_sessions(self) -> list[dict]:
        """List all sessions.

        Returns:
            List of session info dicts.
        """
        return [
            {
                "session_id": s.session_id,
                "question": s.question[:100],
                "created_at": s.created_at.isoformat(),
                "is_complete": s.is_complete,
                "is_waiting_for_input": s.is_waiting_for_input,
            }
            for s in self._sessions.values()
        ]


# Global session manager instance
session_manager = SessionManager()
