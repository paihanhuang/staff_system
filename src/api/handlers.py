"""Session and state management for the API."""

import asyncio
from datetime import datetime
from typing import Optional
from uuid import uuid4

from src.models import GraphState, SystemContext, ConversationTurn
from src.utils.logger import SessionLogger, get_logger
from src.utils.storage import SessionData, SessionStorage, get_storage

logger = get_logger()


class Session:
    """A user session for a Synapse Council run."""

    def __init__(
        self,
        session_id: str,
        question: str,
        system_context: Optional[SystemContext] = None,
        architect_model: Optional[str] = None,
        engineer_model: Optional[str] = None,
        auditor_model: Optional[str] = None,
    ):
        """Initialize a session.

        Args:
            session_id: Unique session identifier.
            question: The user's system design question.
            system_context: Optional system context.
            architect_model: Override for architect model.
            engineer_model: Override for engineer model.
            auditor_model: Override for auditor model.
        """
        self.session_id = session_id
        self.question = question
        self.system_context = system_context
        self.architect_model = architect_model
        self.engineer_model = engineer_model
        self.auditor_model = auditor_model
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
    """Manages active sessions with persistence."""

    def __init__(self, storage: Optional[SessionStorage] = None):
        """Initialize the session manager.
        
        Args:
            storage: Optional storage backend (uses default if not provided).
        """
        self._sessions: dict[str, Session] = {}
        self._lock = asyncio.Lock()
        self._storage = storage or get_storage()

    async def _save_to_storage(self, session: Session) -> None:
        """Persist a session to storage."""
        if not session.state:
            return
        
        session_data = SessionData(
            session_id=session.session_id,
            question=session.question,
            created_at=session.created_at.isoformat(),
            updated_at=session.updated_at.isoformat(),
            state_json=session.state.model_dump(mode="json"),
            conversation_turns=[t.model_dump(mode="json") for t in session.state.conversation_turns],
            is_complete=session.is_complete,
        )
        await self._storage.save_session(session_data)

    async def _load_from_storage(self, session_id: str) -> Optional[Session]:
        """Load a session from storage."""
        data = await self._storage.load_session(session_id)
        if not data:
            return None
        
        session = Session(
            session_id=data.session_id,
            question=data.question,
        )
        session.created_at = datetime.fromisoformat(data.created_at)
        session.updated_at = datetime.fromisoformat(data.updated_at)
        session.is_complete = data.is_complete
        session.state = GraphState(**data.state_json)
        
        return session

    async def create_session(
        self,
        question: str,
        system_context: Optional[SystemContext] = None,
        session_id: Optional[str] = None,
        architect_model: Optional[str] = None,
        engineer_model: Optional[str] = None,
        auditor_model: Optional[str] = None,
    ) -> Session:
        """Create a new session.

        Args:
            question: The user's system design question.
            system_context: Optional system context.
            session_id: Optional session ID (generated if not provided).
            architect_model: Override for architect model.
            engineer_model: Override for engineer model.
            auditor_model: Override for auditor model.

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
                architect_model=architect_model,
                engineer_model=engineer_model,
                auditor_model=auditor_model,
            )
            self._sessions[session_id] = session
            logger.info(f"Created session {session_id}")

        return session

    async def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID (from memory or storage).

        Args:
            session_id: The session ID.

        Returns:
            The session, or None if not found.
        """
        # Try memory first
        if session_id in self._sessions:
            return self._sessions[session_id]
        
        # Try loading from storage
        session = await self._load_from_storage(session_id)
        if session:
            async with self._lock:
                self._sessions[session_id] = session
            logger.info(f"Loaded session {session_id} from storage")
        
        return session

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session from memory and storage.

        Args:
            session_id: The session ID.

        Returns:
            True if deleted, False if not found.
        """
        deleted = False
        async with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                deleted = True
        
        # Also remove from storage
        storage_deleted = await self._storage.delete_session(session_id)
        
        if deleted or storage_deleted:
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

        # Check if this is a follow-up (has previous turns)
        previous_turns = []
        follow_up_context = None
        if session.state and session.state.conversation_turns:
            previous_turns = session.state.conversation_turns
            last_turn = previous_turns[-1]
            follow_up_context = f"Previous design: {last_turn.adr.title}\nDecision: {last_turn.adr.decision[:500]}"

        # Initialize state to show we are starting
        initial_state = GraphState(
            session_id=session_id,
            user_question=session.question,
            system_context=session.system_context,
            current_phase="start",
            conversation_turns=previous_turns,
            follow_up_context=follow_up_context,
            is_follow_up=bool(previous_turns),
        )
        session.update_state(initial_state)

        # Run the graph with streaming updates
        from src.graph.graph import run_graph_stream
        
        final_state = initial_state
        async for state_update in run_graph_stream(
            question=session.question,
            system_context=session.system_context,
            session_id=session_id,
            architect_model=session.architect_model,
            engineer_model=session.engineer_model,
            auditor_model=session.auditor_model,
            follow_up_context=follow_up_context,
        ):
            session.update_state(state_update)
            final_state = state_update
            # We could save to storage incrementally here if needed for crash recovery
            # await self._save_to_storage(session) 
        
        # Preserve conversation history and add new turn
        if final_state.final_adr:
            new_turn = ConversationTurn(
                turn_id=len(previous_turns) + 1,
                question=session.question,
                adr=final_state.final_adr,
            )
            final_state.conversation_turns = previous_turns + [new_turn]

        session.update_state(final_state)
        session.is_complete = True

        if final_state.consensus_reached:
            session.logger.log_consensus(
                reached=True,
                rounds=final_state.round_number,
            )

        # Persist to storage
        await self._save_to_storage(session)

        return session

    async def continue_session(
        self,
        session_id: str,
        follow_up_question: str,
    ) -> Session:
        """Continue an existing session with a follow-up question.

        Args:
            session_id: The session ID to continue.
            follow_up_question: The follow-up question.

        Returns:
            Updated session with new result.

        Raises:
            ValueError: If session not found or not complete.
        """
        session = await self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        if not session.is_complete:
            raise ValueError(f"Session {session_id} is not complete yet")
        
        if not session.state or not session.state.final_adr:
            raise ValueError(f"Session {session_id} has no result to follow up on")

        logger.info(f"Continuing session {session_id} with follow-up: {follow_up_question[:50]}...")

        # Update session with new question
        session.question = follow_up_question
        session.is_complete = False
        session.updated_at = datetime.now()

        # Run the graph with context from previous turn
        return await self.run_session(session_id)

    def list_sessions(self) -> list[dict]:
        """List all sessions (in-memory only).

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
                "turns": len(s.state.conversation_turns) if s.state else 0,
            }
            for s in self._sessions.values()
        ]

    async def list_all_sessions(self) -> list[dict]:
        """List all sessions including those in storage.

        Returns:
            List of session info dicts.
        """
        # Get from memory
        memory_sessions = {s["session_id"]: s for s in self.list_sessions()}
        
        # Get from storage
        storage_sessions = await self._storage.list_sessions()
        
        # Merge (memory takes precedence)
        for stored in storage_sessions:
            if stored["session_id"] not in memory_sessions:
                memory_sessions[stored["session_id"]] = stored
        
        return list(memory_sessions.values())


# Global session manager instance
session_manager = SessionManager()
