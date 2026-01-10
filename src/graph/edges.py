"""Conditional edge logic for the LangGraph state machine."""

from typing import Literal

from src.models import GraphState


def route_after_start(state: GraphState) -> Literal["ideation", "clarification"]:
    """Route from start node.

    Args:
        state: Current graph state.

    Returns:
        Next node to execute.
    """
    if state.interrupt:
        return "clarification"
    return "ideation"


def route_after_ideation(state: GraphState) -> Literal["cross_critique", "clarification", "end"]:
    """Route after ideation phase.

    Args:
        state: Current graph state.

    Returns:
        Next node to execute.
    """
    if state.error or state.current_phase == "error":
        print(f"DEBUG: Routing to end due to error: {state.error}")
        return "end"
    print("DEBUG: Routing to cross_critique")
    return "cross_critique"


def route_after_cross_critique(state: GraphState) -> Literal["audit", "clarification"]:
    """Route after cross-critique phase.

    Args:
        state: Current graph state.

    Returns:
        Next node to execute.
    """
    return "audit"


def route_after_audit(
    state: GraphState,
) -> Literal["convergence", "clarification", "escalate", "ideation"]:
    """Route after audit phase.

    Args:
        state: Current graph state.

    Returns:
        Next node to execute.
    """
    # Check for interrupts first - DISABLING
    # if state.interrupt:
    #    return "clarification"

    # Check if consensus is reached
    if state.consensus_reached:
        return "convergence"

    # Check if we've exceeded max rounds
    if state.round_number >= state.max_rounds:
        return "escalate"

    # Otherwise, go back for another round
    return "ideation"





def should_continue(state: GraphState) -> bool:
    """Check if the graph should continue execution.

    Args:
        state: Current graph state.

    Returns:
        True if execution should continue, False otherwise.
    """
    # Stop if we have a final ADR
    if state.final_adr:
        return False

    # Stop if there's an error
    if state.error:
        return False

    # Stop if we're waiting for user input
    if state.interrupt and not state.user_response:
        return False

    return True
