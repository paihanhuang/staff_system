"""API routes for the Synapse Council."""

import asyncio
import json
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from src.models import SystemContext
from src.api.handlers import session_manager
from src.api.rate_limiter import (
    RateLimitExceededError,
    get_rate_limiter,
    RateLimitConfig,
    SessionRateLimiter,
)
from src.utils.logger import get_logger
from src.utils.sanitization import validate_question, SanitizationError
from src.utils.config import get_settings

logger = get_logger()

router = APIRouter(prefix="/api", tags=["synapse-council"])

# Rate limiter dependency
_rate_limiter: Optional[SessionRateLimiter] = None


def get_session_rate_limiter() -> SessionRateLimiter:
    """Get the session rate limiter."""
    global _rate_limiter
    if _rate_limiter is None:
        settings = get_settings()
        _rate_limiter = SessionRateLimiter(
            per_session_config=RateLimitConfig(
                requests_per_minute=settings.rate_limit_per_session_per_minute,
                requests_per_hour=settings.rate_limit_per_session_per_hour,
            ),
            global_config=RateLimitConfig(
                requests_per_minute=settings.rate_limit_global_per_minute,
                requests_per_hour=settings.rate_limit_global_per_hour,
            ),
        )
    return _rate_limiter


# Request/Response models
class DesignRequest(BaseModel):
    """Request to start a new design session."""

    question: str = Field(..., description="The system design question")
    system_context: Optional[SystemContext] = Field(
        default=None, description="Optional system context"
    )
    session_id: Optional[str] = Field(default=None, description="Optional session ID")
    architect_model: Optional[str] = Field(default=None, description="Override architect model")
    engineer_model: Optional[str] = Field(default=None, description="Override engineer model")
    auditor_model: Optional[str] = Field(default=None, description="Override auditor model")


class LogResponse(BaseModel):
    """Response containing server logs."""
    logs: list[str] = Field(..., description="List of log lines")


@router.get("/logs", response_model=LogResponse)
async def get_logs(lines: int = 100):
    """Get the latest server logs."""
    try:
        # Read from backend.log
        with open("backend.log", "r") as f:
            # Efficiently read last N lines using tail-like logic or just readlines for simplicity
            # For simplicity with small logs, readlines is fine. For huge logs, seek would be better.
            all_lines = f.readlines()
            return LogResponse(logs=all_lines[-lines:])
    except FileNotFoundError:
        return LogResponse(logs=["Log file not found."])
    except Exception as e:
        logger.error(f"Error reading logs: {e}")
        return LogResponse(logs=[f"Error reading logs: {str(e)}"])



class SessionResponse(BaseModel):
    """Response with session information."""

    session_id: str
    is_complete: bool
    is_waiting_for_input: bool
    current_phase: Optional[str] = None
    interrupt_question: Optional[str] = None
    interrupt_source: Optional[str] = None
    final_decision: Optional[str] = None
    error: Optional[str] = None
    usage_metrics: Optional[dict] = None


class AgentProposalSummary(BaseModel):
    """Summary of an agent's proposal."""

    title: Optional[str] = None
    summary: Optional[str] = None
    confidence: Optional[float] = None
    approach: Optional[str] = None


class DetailedSessionResponse(BaseModel):
    """Enhanced response with full progress details for real-time display."""

    session_id: str
    is_complete: bool
    is_waiting_for_input: bool
    is_running: bool = False
    current_phase: Optional[str] = None
    
    # Agent proposals
    architect_proposal: Optional[AgentProposalSummary] = None
    engineer_proposal: Optional[AgentProposalSummary] = None
    
    # Refined proposals
    architect_refined_proposal: Optional[AgentProposalSummary] = None
    engineer_refined_proposal: Optional[AgentProposalSummary] = None
    
    # Critique summaries (round 1)
    architect_critique_summary: Optional[str] = None
    engineer_critique_summary: Optional[str] = None
    
    # Critique summaries (round 2)
    architect_critique_2_summary: Optional[str] = None
    engineer_critique_2_summary: Optional[str] = None
    
    # Audit info
    audit_preferred: Optional[str] = None
    audit_consensus_possible: Optional[bool] = None
    
    # Interrupt info - Removed
    # interrupt_question: Optional[str] = None
    # interrupt_source: Optional[str] = None
    
    # Final result
    final_decision: Optional[str] = None
    consensus_level: Optional[float] = None
    
    # Metrics
    error: Optional[str] = None
    usage_metrics: Optional[dict] = None
    elapsed_time: Optional[float] = None



class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    timestamp: str
    version: str = "1.0.0"
    active_sessions: int
    api_keys_configured: dict[str, bool]


def _session_to_response(session) -> SessionResponse:
    """Convert a session to a response model."""
    state = session.state
    return SessionResponse(
        session_id=session.session_id,
        is_complete=session.is_complete,
        is_waiting_for_input=session.is_waiting_for_input,
        current_phase=state.current_phase if state else None,
        interrupt_question=state.interrupt.question if state and state.interrupt else None,
        interrupt_source=state.interrupt.source if state and state.interrupt else None,
        final_decision=state.final_adr.title if state and state.final_adr else None,
        error=state.error if state else None,
        usage_metrics=getattr(state, 'usage_metrics', None) if state else None,
    )


def _session_to_detailed_response(session, is_running: bool = False) -> DetailedSessionResponse:
    """Convert a session to a detailed response with agent proposals."""
    state = session.state
    
    # Extract architect proposal summary
    architect_proposal = None
    if state and state.architect_proposal:
        ap = state.architect_proposal
        architect_proposal = AgentProposalSummary(
            title=ap.title,
            summary=ap.summary,
            confidence=ap.confidence,
            approach=ap.approach,
        )
    
    # Extract engineer proposal summary
    engineer_proposal = None
    if state and state.engineer_proposal:
        ep = state.engineer_proposal
        engineer_proposal = AgentProposalSummary(
            title=ep.title,
            summary=ep.summary,
            confidence=ep.confidence,
            approach=ep.approach,
        )
    
    # Extract refined architect proposal summary
    architect_refined_proposal = None
    if state and state.architect_refined_proposal:
        arp = state.architect_refined_proposal
        architect_refined_proposal = AgentProposalSummary(
            title=arp.title,
            summary=arp.summary,
            confidence=arp.confidence,
            approach=arp.approach,
        )
    
    # Extract refined engineer proposal summary
    engineer_refined_proposal = None
    if state and state.engineer_refined_proposal:
        erp = state.engineer_refined_proposal
        engineer_refined_proposal = AgentProposalSummary(
            title=erp.title,
            summary=erp.summary,
            confidence=erp.confidence,
            approach=erp.approach,
        )
    
    # Extract critique summaries (round 1)
    architect_critique_summary = None
    engineer_critique_summary = None
    if state and state.architect_critique:
        ac = state.architect_critique
        architect_critique_summary = f"Agreement: {ac.agreement_level:.0%}. Concerns: {', '.join(ac.concerns[:2]) if ac.concerns else 'None'}"
    if state and state.engineer_critique:
        ec = state.engineer_critique
        engineer_critique_summary = f"Agreement: {ec.agreement_level:.0%}. Concerns: {', '.join(ec.concerns[:2]) if ec.concerns else 'None'}"
    
    # Extract critique summaries (round 2)
    architect_critique_2_summary = None
    engineer_critique_2_summary = None
    if state and state.architect_critique_2:
        ac2 = state.architect_critique_2
        architect_critique_2_summary = f"Agreement: {ac2.agreement_level:.0%}. Concerns: {', '.join(ac2.concerns[:2]) if ac2.concerns else 'None'}"
    if state and state.engineer_critique_2:
        ec2 = state.engineer_critique_2
        engineer_critique_2_summary = f"Agreement: {ec2.agreement_level:.0%}. Concerns: {', '.join(ec2.concerns[:2]) if ec2.concerns else 'None'}"
    
    # Extract audit info
    audit_preferred = None
    audit_consensus_possible = None
    if state and state.audit_result:
        audit_preferred = state.audit_result.preferred_approach
        audit_consensus_possible = state.audit_result.consensus_possible
    
    # Calculate elapsed time
    elapsed_time = None
    if hasattr(session, 'created_at'):
        elapsed_time = (datetime.now() - session.created_at).total_seconds()
    
    return DetailedSessionResponse(
        session_id=session.session_id,
        is_complete=session.is_complete,
        is_waiting_for_input=session.is_waiting_for_input,
        is_running=is_running,
        current_phase=state.current_phase if state else None,
        architect_proposal=architect_proposal,
        engineer_proposal=engineer_proposal,
        architect_refined_proposal=architect_refined_proposal,
        engineer_refined_proposal=engineer_refined_proposal,
        architect_critique_summary=architect_critique_summary,
        engineer_critique_summary=engineer_critique_summary,
        architect_critique_2_summary=architect_critique_2_summary,
        engineer_critique_2_summary=engineer_critique_2_summary,
        audit_preferred=audit_preferred,
        audit_consensus_possible=audit_consensus_possible,

        final_decision=state.final_adr.title if state and state.final_adr else None,
        consensus_level=state.final_adr.consensus_level if state and state.final_adr else None,
        error=state.error if state else None,
        usage_metrics=getattr(state, 'usage_metrics', None) if state else None,
        elapsed_time=elapsed_time,
    )


# Track running background tasks
_running_sessions: set[str] = set()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint for monitoring."""
    settings = get_settings()
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        active_sessions=len(session_manager.list_sessions()),
        api_keys_configured=settings.validate_api_keys(),
    )


@router.post("/design", response_model=SessionResponse)
async def start_design(
    request: DesignRequest,
    rate_limiter: SessionRateLimiter = Depends(get_session_rate_limiter),
) -> SessionResponse:
    """Start a new architecture design session.

    Creates a session and runs the Synapse Council graph until completion
    or an interrupt requiring user input.
    """
    # Validate input
    is_valid, error = validate_question(request.question)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error)

    # Generate session ID for rate limiting
    session_id = request.session_id or str(id(request))

    # Check rate limit
    try:
        rate_limiter.check(session_id)
    except RateLimitExceededError as e:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Retry after {e.retry_after:.1f} seconds.",
            headers={"Retry-After": str(int(e.retry_after))},
        )

    try:
        # Create session
        session = await session_manager.create_session(
            question=request.question,
            system_context=request.system_context,
            session_id=request.session_id,
        )

        # Run the graph
        session = await session_manager.run_session(session.session_id)

        return _session_to_response(session)

    except Exception as e:
        logger.error(f"Error starting design: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _run_session_background(session_id: str):
    """Background task to run the session."""
    try:
        _running_sessions.add(session_id)
        await session_manager.run_session(session_id)
    except Exception as e:
        logger.error(f"Background session {session_id} failed: {e}")
        # Store error in session if possible
        session = await session_manager.get_session(session_id)
        if session and session.state:
            session.state.error = str(e)
    finally:
        _running_sessions.discard(session_id)


@router.post("/design/start", response_model=DetailedSessionResponse)
async def start_design_async(
    request: DesignRequest,
    background_tasks: BackgroundTasks,
    rate_limiter: SessionRateLimiter = Depends(get_session_rate_limiter),
) -> DetailedSessionResponse:
    """Start a design session asynchronously (returns immediately).
    
    The session runs in the background. Poll /design/{session_id}/detailed 
    for progress updates.
    """
    # Validate input
    is_valid, error = validate_question(request.question)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error)

    # Generate session ID for rate limiting
    session_id = request.session_id or str(id(request))

    # Check rate limit
    try:
        rate_limiter.check(session_id)
    except RateLimitExceededError as e:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Retry after {e.retry_after:.1f} seconds.",
            headers={"Retry-After": str(int(e.retry_after))},
        )

    try:
        # Create session
        session = await session_manager.create_session(
            question=request.question,
            system_context=request.system_context,
            session_id=request.session_id,
            architect_model=request.architect_model,
            engineer_model=request.engineer_model,
            auditor_model=request.auditor_model,
        )

        # Start background task
        background_tasks.add_task(_run_session_background, session.session_id)
        
        return _session_to_detailed_response(session, is_running=True)

    except Exception as e:
        logger.error(f"Error starting async design: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/design/{session_id}/detailed", response_model=DetailedSessionResponse)
async def get_session_detailed(session_id: str) -> DetailedSessionResponse:
    """Get detailed status of a design session with agent proposals."""
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    is_running = session_id in _running_sessions
    return _session_to_detailed_response(session, is_running=is_running)




@router.get("/design/{session_id}", response_model=SessionResponse)
async def get_session_status(session_id: str) -> SessionResponse:
    """Get the status of a design session."""
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    return _session_to_response(session)


@router.get("/design/{session_id}/result")
async def get_session_result(session_id: str):
    """Get the full result of a completed design session."""
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    if not session.is_complete:
        raise HTTPException(
            status_code=400,
            detail="Session is not complete",
        )

    state = session.state
    if state.final_adr:
        return {
            "session_id": session_id,
            "adr": state.final_adr.model_dump(),
            "conversation_history": [m.model_dump() for m in state.conversation_history],
            "rounds_taken": state.round_number,
            "consensus_reached": state.consensus_reached,
        }
    else:
        return {
            "session_id": session_id,
            "escalated": True,
            "error": state.error,
            "architect_proposal": state.architect_proposal.model_dump() if state.architect_proposal else None,
            "engineer_proposal": state.engineer_proposal.model_dump() if state.engineer_proposal else None,
            "audit_result": state.audit_result.model_dump() if state.audit_result else None,
        }


@router.delete("/design/{session_id}")
async def delete_session(session_id: str):
    """Delete a design session."""
    deleted = await session_manager.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    return {"deleted": True, "session_id": session_id}


@router.get("/sessions")
async def list_sessions():
    """List all active sessions."""
    return {"sessions": session_manager.list_sessions()}


# WebSocket for real-time streaming
class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, session_id: str, websocket: WebSocket):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info(f"WebSocket connected for session {session_id}")

    def disconnect(self, session_id: str):
        """Disconnect a WebSocket."""
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            logger.info(f"WebSocket disconnected for session {session_id}")

    async def send_update(self, session_id: str, data: dict):
        """Send an update to a connected client."""
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_json(data)


ws_manager = ConnectionManager()


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time session updates.

    Clients can:
    - Start a session by sending {"action": "start", "question": "...", "context": {...}}
    - Respond to clarifications by sending {"action": "respond", "response": "..."}
    - Get status by sending {"action": "status"}
    """
    await ws_manager.connect(session_id, websocket)

    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action")

            if action == "start":
                # Start a new session
                question = data.get("question")
                if not question:
                    await websocket.send_json({"error": "Question is required"})
                    continue

                context_data = data.get("context")
                system_context = SystemContext(**context_data) if context_data else None

                try:
                    session = await session_manager.create_session(
                        question=question,
                        system_context=system_context,
                        session_id=session_id,
                    )

                    await websocket.send_json({
                        "event": "session_created",
                        "session_id": session_id,
                    })

                    # Run the graph
                    session = await session_manager.run_session(session_id)

                    # Send result
                    await websocket.send_json({
                        "event": "phase_complete",
                        "session_id": session_id,
                        "is_complete": session.is_complete,
                        "is_waiting_for_input": False,
                        "current_phase": session.state.current_phase if session.state else None,
                        "interrupt": None,
                    })

                except Exception as e:
                    await websocket.send_json({"error": str(e)})

            elif action == "status":
                session = await session_manager.get_session(session_id)
                if session:
                    await websocket.send_json({
                        "event": "status",
                        "session_id": session_id,
                        "is_complete": session.is_complete,
                        "is_waiting_for_input": session.is_waiting_for_input,
                    })
                else:
                    await websocket.send_json({"error": "Session not found"})

            else:
                await websocket.send_json({"error": f"Unknown action: {action}"})

    except WebSocketDisconnect:
        ws_manager.disconnect(session_id)
