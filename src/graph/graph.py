"""Main LangGraph definition for the Synapse Council."""

import uuid
from typing import Any, Optional

from langgraph.graph import StateGraph, END

from src.models import GraphState, SystemContext
from src.graph.nodes import (
    ideation_node,
    cross_critique_node,
    refinement_node,
    cross_critique_2_node,
    audit_node,
    convergence_node,
)
from src.graph.edges import (
    route_after_ideation,
    route_after_cross_critique,
    route_after_refinement,
    route_after_cross_critique_2,
    route_after_audit,
)
from src.utils.logger import get_logger

logger = get_logger()


async def escalate_node(state: GraphState) -> dict:
    """Escalate to user when consensus cannot be reached.

    Presents all proposals with recommendations for the user to decide.
    """
    logger.info(f"[{state.session_id}] Escalating to user - no consensus after {state.round_number} rounds")

    # Use refined proposals if available, otherwise use original proposals
    architect_proposal = state.architect_refined_proposal or state.architect_proposal
    engineer_proposal = state.engineer_refined_proposal or state.engineer_proposal

    # Build escalation message
    escalation_message = f"""
The Synapse Council could not reach consensus after {state.round_number} rounds of deliberation.

## Proposal A: The Architect
**{architect_proposal.title}**
{architect_proposal.summary}
Confidence: {architect_proposal.confidence:.0%}

## Proposal B: The Engineer
**{engineer_proposal.title}**
{engineer_proposal.summary}
Confidence: {engineer_proposal.confidence:.0%}

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
    3. cross_critique -> refinement (agents improve based on critique)
    4. refinement -> cross_critique_2 (critique refined proposals)
    5. cross_critique_2 -> audit (Auditor evaluates refined proposals)
    6. audit -> convergence (if consensus) | escalate (if max rounds) | ideation (retry)

    Returns:
        Compiled StateGraph ready for execution.
    """
    # Create the graph with our state schema
    workflow = StateGraph(GraphState)

    # Add all nodes
    workflow.add_node("ideation", ideation_node)
    workflow.add_node("cross_critique", cross_critique_node)
    workflow.add_node("refinement", refinement_node)
    workflow.add_node("cross_critique_2", cross_critique_2_node)
    workflow.add_node("audit", audit_node)
    workflow.add_node("convergence", convergence_node)
    workflow.add_node("escalate", escalate_node)

    # Set entry point
    workflow.set_entry_point("ideation")

    # Add conditional edges
    workflow.add_conditional_edges(
        "ideation",
        route_after_ideation,
        {
            "cross_critique": "cross_critique",
            "end": END,
        },
    )

    workflow.add_conditional_edges(
        "cross_critique",
        route_after_cross_critique,
        {
            "refinement": "refinement",
        },
    )

    workflow.add_conditional_edges(
        "refinement",
        route_after_refinement,
        {
            "cross_critique_2": "cross_critique_2",
        },
    )

    workflow.add_conditional_edges(
        "cross_critique_2",
        route_after_cross_critique_2,
        {
            "audit": "audit",
        },
    )

    workflow.add_conditional_edges(
        "audit",
        route_after_audit,
        {
            "convergence": "convergence",
            "escalate": "escalate",
            "ideation": "ideation",
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
    architect_model: Optional[str] = None,
    engineer_model: Optional[str] = None,
    auditor_model: Optional[str] = None,
) -> GraphState:
    """Run the Synapse Council graph for a system design question.

    Args:
        question: The user's system design question.
        system_context: Optional context about the system.
        session_id: Optional session ID (generated if not provided).
        max_rounds: Maximum rounds before escalation.
        architect_model: Override model for Architect.
        engineer_model: Override model for Engineer.
        auditor_model: Override model for Auditor.

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
        architect_model=architect_model,
        engineer_model=engineer_model,
        auditor_model=auditor_model,
    )

    # Create and run the graph
    graph = create_graph()
    final_state_dict = await graph.ainvoke(initial_state)
    
    # Convert dict back to Pydantic model
    # We need to merge with initial state request data as langgraph might only return changed keys depending on configuration, 
    # though usually it returns full state for StateGraph(PydanticModel).
    # Being safe and ensuring valid model instantiation.
    state_data = {
        "session_id": session_id,
        "user_question": question,
        "system_context": system_context,
        "max_rounds": max_rounds,
        "architect_model": architect_model,
        "engineer_model": engineer_model,
        "auditor_model": auditor_model,
    }
    state_data.update(final_state_dict)
    
    final_state = GraphState(**state_data)

    logger.info(f"[{session_id}] Synapse Council complete - phase: {final_state.current_phase}")

    return final_state




