"""Main LangGraph definition for the Synapse Council."""

import uuid
from typing import Any, Optional

from langgraph.graph import StateGraph, END

from src.models import GraphState, SystemContext
from src.graph.nodes import (
    ideation_node,
    cross_critique_node,
    audit_node,
    convergence_node,
    clarification_node,
)
from src.graph.edges import (
    route_after_ideation,
    route_after_cross_critique,
    route_after_audit,
    route_after_clarification,
)
from src.utils.logger import get_logger

logger = get_logger()


async def escalate_node(state: GraphState) -> dict:
    """Escalate to user when consensus cannot be reached.

    Presents all proposals with recommendations for the user to decide.
    """
    logger.info(f"[{state.session_id}] Escalating to user - no consensus after {state.round_number} rounds")

    # Build escalation message
    escalation_message = f"""
The Synapse Council could not reach consensus after {state.round_number} rounds of deliberation.

## Proposal A: The Architect
**{state.architect_proposal.title}**
{state.architect_proposal.summary}
Confidence: {state.architect_proposal.confidence:.0%}

## Proposal B: The Engineer
**{state.engineer_proposal.title}**
{state.engineer_proposal.summary}
Confidence: {state.engineer_proposal.confidence:.0%}

## Auditor's Assessment
Preferred approach: {state.audit_result.preferred_approach}
{state.audit_result.preference_rationale}

## Recommendation
{state.audit_result.synthesis_recommendation or "Please review both proposals and make a decision."}
"""

    return {
        "current_phase": "escalated",
        "error": escalation_message,
    }


def create_graph() -> StateGraph:
    """Create the LangGraph state machine for the Synapse Council.

    The workflow is:
    1. START -> ideation (parallel proposals from Architect and Engineer)
    2. ideation -> cross_critique (agents critique each other)
    3. cross_critique -> audit (Auditor evaluates both)
    4. audit -> convergence (if consensus) | escalate (if max rounds) | ideation (retry)
    5. Any node -> clarification (if interrupt) -> resume

    Returns:
        Compiled StateGraph ready for execution.
    """
    # Create the graph with our state schema
    workflow = StateGraph(GraphState)

    # Add all nodes
    workflow.add_node("ideation", ideation_node)
    workflow.add_node("cross_critique", cross_critique_node)
    workflow.add_node("audit", audit_node)
    workflow.add_node("convergence", convergence_node)
    workflow.add_node("clarification", clarification_node)
    workflow.add_node("escalate", escalate_node)

    # Set entry point
    workflow.set_entry_point("ideation")

    # Add conditional edges
    workflow.add_conditional_edges(
        "ideation",
        route_after_ideation,
        {
            "cross_critique": "cross_critique",
            "clarification": "clarification",
            "end": END,
        },
    )

    workflow.add_conditional_edges(
        "cross_critique",
        route_after_cross_critique,
        {
            "audit": "audit",
            "clarification": "clarification",
        },
    )

    workflow.add_conditional_edges(
        "audit",
        route_after_audit,
        {
            "convergence": "convergence",
            "clarification": "clarification",
            "escalate": "escalate",
            "ideation": "ideation",
        },
    )

    workflow.add_conditional_edges(
        "clarification",
        route_after_clarification,
        {
            "ideation": "ideation",
            "cross_critique": "cross_critique",
            "audit": "audit",
            "end": END,
        },
    )

    # Terminal nodes
    workflow.add_edge("convergence", END)
    workflow.add_edge("escalate", END)

    return workflow.compile()


async def run_graph(
    question: str,
    system_context: Optional[SystemContext] = None,
    session_id: Optional[str] = None,
    max_rounds: int = 3,
) -> GraphState:
    """Run the Synapse Council graph for a system design question.

    Args:
        question: The user's system design question.
        system_context: Optional context about the system.
        session_id: Optional session ID (generated if not provided).
        max_rounds: Maximum rounds before escalation.

    Returns:
        Final GraphState with the result.
    """
    # Create unique session ID
    session_id = session_id or str(uuid.uuid4())[:8]
    logger.info(f"[{session_id}] Starting Synapse Council for: {question[:100]}...")

    # Create initial state
    initial_state = GraphState(
        session_id=session_id,
        user_question=question,
        system_context=system_context,
        max_rounds=max_rounds,
        current_phase="start",
    )

    # Create and run the graph
    graph = create_graph()
    final_state = await graph.ainvoke(initial_state)

    logger.info(f"[{session_id}] Synapse Council complete - phase: {final_state.get('current_phase')}")

    return final_state


async def run_graph_with_interrupt(
    question: str,
    system_context: Optional[SystemContext] = None,
    session_id: Optional[str] = None,
    max_rounds: int = 3,
) -> tuple[GraphState, bool]:
    """Run the graph with interrupt support.

    This version yields when an interrupt is encountered, allowing
    the caller to provide user input before continuing.

    Args:
        question: The user's system design question.
        system_context: Optional context about the system.
        session_id: Optional session ID.
        max_rounds: Maximum rounds before escalation.

    Returns:
        Tuple of (current state, is_complete).
    """
    session_id = session_id or str(uuid.uuid4())[:8]

    initial_state = GraphState(
        session_id=session_id,
        user_question=question,
        system_context=system_context,
        max_rounds=max_rounds,
        current_phase="start",
    )

    graph = create_graph()

    # Use ainvoke to get the complete final state
    try:
        final_state_dict = await graph.ainvoke(initial_state)
        
        # Create GraphState from the result, merging with initial state for required fields
        state_data = {
            "session_id": session_id,
            "user_question": question,
            "system_context": system_context,
            "max_rounds": max_rounds,
        }
        state_data.update(final_state_dict)
        
        final_state = GraphState(**state_data)
        
        # Check if waiting for clarification
        if final_state.interrupt or final_state.current_phase == "waiting_for_clarification":
            return final_state, False
        
        # Check if complete
        is_complete = final_state.final_adr is not None or final_state.error is not None
        return final_state, is_complete
        
    except Exception as e:
        # Return error state
        error_state = GraphState(
            session_id=session_id,
            user_question=question,
            system_context=system_context,
            max_rounds=max_rounds,
            current_phase="error",
            error=str(e),
        )
        return error_state, True



async def resume_graph(
    state: GraphState,
    user_response: str,
) -> tuple[GraphState, bool]:
    """Resume a paused graph with user input.

    Args:
        state: The paused state.
        user_response: The user's response to the interrupt.

    Returns:
        Tuple of (new state, is_complete).
    """
    # Update state with user response
    state.user_response = user_response
    state.interrupt = None
    state.current_phase = "resuming"

    graph = create_graph()

    try:
        # Use ainvoke for complete state handling
        final_state_dict = await graph.ainvoke(state.model_dump())
        
        # Merge with original state for required fields
        state_data = state.model_dump()
        state_data.update(final_state_dict)
        
        final_state = GraphState(**state_data)
        
        # Check if waiting for another clarification
        if final_state.interrupt or final_state.current_phase == "waiting_for_clarification":
            return final_state, False
        
        is_complete = final_state.final_adr is not None or final_state.error is not None
        return final_state, is_complete
        
    except Exception as e:
        logger.error(f"Error resuming graph: {e}", exc_info=True)
        state.error = str(e)
        state.current_phase = "error"
        return state, True

