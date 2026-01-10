"""Node implementations for the LangGraph state machine."""

import asyncio
import re
import time
from datetime import datetime
from typing import Optional

from src.adapters.openai_adapter import ArchitectAdapter, SupervisorAdapter
from src.adapters.anthropic_adapter import EngineerAdapter
from src.adapters.google_adapter import AuditorAdapter
from src.models import (
    ArchitectureDecisionRecord,
    ArchitectureProposal,
    AuditResult,
    Critique,
    GraphState,
    Interrupt,
    InterruptType,
    Message,
    MessageRole,
    SystemContext,
)
from src.prompts.architect import ARCHITECT_IDEATION_PROMPT, ARCHITECT_SYSTEM_PROMPT, ARCHITECT_CRITIQUE_PROMPT
from src.prompts.engineer import ENGINEER_IDEATION_PROMPT, ENGINEER_SYSTEM_PROMPT, ENGINEER_CRITIQUE_PROMPT
from src.prompts.auditor import AUDITOR_PROMPT, AUDITOR_SYSTEM_PROMPT
from src.utils.logger import get_logger
from src.utils.sanitization import sanitize_user_input
from src.utils.metrics import UsageMetrics, get_cost_estimator

logger = get_logger()

# Pattern for detecting clarification requests
CLARIFICATION_PATTERN = re.compile(r"<<CLARIFICATION_NEEDED:\s*(.+?)>>", re.DOTALL)


def _check_for_interrupt(proposal: ArchitectureProposal, source: str) -> Optional[Interrupt]:
    """Check if a proposal contains a clarification request."""
    if proposal.clarification_needed:
        return Interrupt(
            type=InterruptType.CLARIFICATION_NEEDED,
            source=source,
            question=proposal.clarification_needed,
        )
    return None


def _format_components(components: list) -> str:
    """Format components list for prompts."""
    if not components:
        return "None specified"
    return "\n".join(
        f"- {c.name} ({c.type}): {c.technology} - {c.description}"
        for c in components
    )


def _format_trade_offs(trade_offs: list) -> str:
    """Format trade-offs list for prompts."""
    if not trade_offs:
        return "None specified"
    return "\n".join(
        f"- {t.aspect}: {t.choice} (Rationale: {t.rationale})"
        for t in trade_offs
    )


def _format_risks(risks: list) -> str:
    """Format risks list for prompts."""
    if not risks:
        return "None specified"
    return "\n".join(
        f"- [{r.severity.upper()}] {r.category}: {r.description} (Mitigation: {r.mitigation})"
        for r in risks
    )


async def ideation_node(state: GraphState) -> dict:
    """Node 1: Blind Ideation - Both agents propose solutions independently.

    The Architect (o3) and The Engineer (Claude) receive the same question
    but work independently to prevent groupthink.
    """
    start_time = time.time()
    logger.info(f"[{state.session_id}] Starting ideation phase")

    # Sanitize user input
    sanitization_result = sanitize_user_input(state.user_question)
    sanitized_question = sanitization_result.sanitized
    if sanitization_result.warnings:
        logger.warning(
            f"[{state.session_id}] Input sanitization warnings: "
            f"{sanitization_result.warnings}"
        )

    # Prepare context string
    context_str = (
        state.system_context.to_prompt_string()
        if state.system_context
        else "No additional context provided."
    )

    # Initialize adapters
    architect_adapter = ArchitectAdapter()
    engineer_adapter = EngineerAdapter()

    # Prepare prompts with sanitized input
    architect_prompt = ARCHITECT_IDEATION_PROMPT.format(
        user_question=sanitized_question,
        system_context=context_str,
    )
    engineer_prompt = ENGINEER_IDEATION_PROMPT.format(
        user_question=sanitized_question,
        system_context=context_str,
    )

    # Initialize metrics
    usage_metrics = UsageMetrics()
    cost_estimator = get_cost_estimator()

    try:
        # Run both agents in parallel with usage tracking
        architect_result = architect_adapter.generate_structured_with_usage(
            prompt=architect_prompt,
            response_model=ArchitectureProposal,
            system_prompt=ARCHITECT_SYSTEM_PROMPT,
            temperature=0.7,
        )
        engineer_result = engineer_adapter.generate_structured_with_usage(
            prompt=engineer_prompt,
            response_model=ArchitectureProposal,
            system_prompt=ENGINEER_SYSTEM_PROMPT,
            temperature=0.7,
        )

        results = await asyncio.gather(
            architect_result, engineer_result,
            return_exceptions=True
        )

        # Handle potential errors
        errors = [r for r in results if isinstance(r, Exception)]
        if len(errors) == 2:
            raise errors[0]  # Both failed

        # Extract results and usage
        if isinstance(results[0], Exception):
            logger.error(f"[{state.session_id}] Architect failed: {results[0]}")
            architect_proposal = None
            architect_usage = None
        else:
            architect_proposal, architect_usage = results[0]
            usage_metrics.add_usage(architect_usage, "ideation_architect")

        if isinstance(results[1], Exception):
            logger.error(f"[{state.session_id}] Engineer failed: {results[1]}")
            engineer_proposal = None
            engineer_usage = None
        else:
            engineer_proposal, engineer_usage = results[1]
            usage_metrics.add_usage(engineer_usage, "ideation_engineer")

        # At least one must succeed
        if not architect_proposal and not engineer_proposal:
            return {
                "current_phase": "error",
                "error": "Both agents failed during ideation",
            }

    except Exception as e:
        logger.error(f"[{state.session_id}] Ideation failed: {e}")
        return {
            "current_phase": "error",
            "error": f"Ideation phase failed: {str(e)}",
        }

    # Record phase timing
    phase_duration = time.time() - start_time
    usage_metrics.record_phase_timing("ideation", phase_duration)

    # Calculate cost estimate
    total_cost = cost_estimator.estimate_total_cost(usage_metrics)
    usage_metrics.total_cost = total_cost

    logger.info(
        f"[{state.session_id}] Ideation complete in {phase_duration:.2f}s - "
        f"Architect: {architect_proposal.title if architect_proposal else 'FAILED'}, "
        f"Engineer: {engineer_proposal.title if engineer_proposal else 'FAILED'}, "
        f"Est. cost: ${total_cost:.4f}"
    )

    # Check for interrupts - DISABLING for direct judgment
    interrupt = None

    # Create messages for history
    messages = []
    if architect_proposal:
        messages.append(Message(
            role=MessageRole.ARCHITECT,
            content=f"Proposal: {architect_proposal.title}\n{architect_proposal.summary}",
            metadata={"confidence": architect_proposal.confidence},
        ))
    if engineer_proposal:
        messages.append(Message(
            role=MessageRole.ENGINEER,
            content=f"Proposal: {engineer_proposal.title}\n{engineer_proposal.summary}",
            metadata={"confidence": engineer_proposal.confidence},
        ))

    result = {
        "architect_proposal": architect_proposal,
        "engineer_proposal": engineer_proposal,
        "current_phase": "ideation_complete",
        "interrupt": interrupt,
        "conversation_history": messages,
        "usage_metrics": usage_metrics.to_dict(),
    }
    logger.info(f"[{state.session_id}] Ideation returning keys: {list(result.keys())}")
    return result


async def cross_critique_node(state: GraphState) -> dict:
    """Node 1.5: Cross-Critique - Agents critique each other's proposals.

    The Architect critiques the Engineer's proposal and vice versa.
    This creates richer feedback before the final audit.
    """
    logger.info(f"[{state.session_id}] Starting cross-critique phase")

    if not state.architect_proposal or not state.engineer_proposal:
        raise ValueError("Both proposals must exist for cross-critique")

    # Prepare context
    context_str = (
        state.system_context.to_prompt_string()
        if state.system_context
        else "No additional context provided."
    )

    # Initialize adapters
    architect_adapter = ArchitectAdapter()
    engineer_adapter = EngineerAdapter()

    # Architect critiques Engineer's proposal
    architect_critique_prompt = ARCHITECT_CRITIQUE_PROMPT.format(
        user_question=state.user_question,
        system_context=context_str,
        engineer_title=state.engineer_proposal.title,
        engineer_summary=state.engineer_proposal.summary,
        engineer_approach=state.engineer_proposal.approach,
        engineer_components=_format_components(state.engineer_proposal.components),
        engineer_trade_offs=_format_trade_offs(state.engineer_proposal.trade_offs),
        engineer_risks=_format_risks(state.engineer_proposal.risks),
        engineer_diagram=state.engineer_proposal.mermaid_diagram or "Not provided",
    )

    # Engineer critiques Architect's proposal
    engineer_critique_prompt = ENGINEER_CRITIQUE_PROMPT.format(
        user_question=state.user_question,
        system_context=context_str,
        architect_title=state.architect_proposal.title,
        architect_summary=state.architect_proposal.summary,
        architect_approach=state.architect_proposal.approach,
        architect_components=_format_components(state.architect_proposal.components),
        architect_trade_offs=_format_trade_offs(state.architect_proposal.trade_offs),
        architect_risks=_format_risks(state.architect_proposal.risks),
        architect_diagram=state.architect_proposal.mermaid_diagram or "Not provided",
    )

    # Run both critiques in parallel
    architect_critique_task = architect_adapter.generate_structured(
        prompt=architect_critique_prompt,
        response_model=Critique,
        system_prompt=ARCHITECT_SYSTEM_PROMPT,
        temperature=0.5,
    )
    engineer_critique_task = engineer_adapter.generate_structured(
        prompt=engineer_critique_prompt,
        response_model=Critique,
        system_prompt=ENGINEER_SYSTEM_PROMPT,
        temperature=0.5,
    )

    architect_critique, engineer_critique = await asyncio.gather(
        architect_critique_task, engineer_critique_task
    )

    logger.info(
        f"[{state.session_id}] Cross-critique complete - "
        f"Architect agreement with Engineer: {architect_critique.agreement_level:.0%}, "
        f"Engineer agreement with Architect: {engineer_critique.agreement_level:.0%}"
    )

    # Create messages for history
    messages = [
        Message(
            role=MessageRole.ARCHITECT,
            content=f"Critique of Engineer's proposal: {', '.join(architect_critique.concerns[:3])}",
            metadata={"agreement_level": architect_critique.agreement_level},
        ),
        Message(
            role=MessageRole.ENGINEER,
            content=f"Critique of Architect's proposal: {', '.join(engineer_critique.concerns[:3])}",
            metadata={"agreement_level": engineer_critique.agreement_level},
        ),
    ]

    return {
        "architect_critique": architect_critique,
        "engineer_critique": engineer_critique,
        "current_phase": "cross_critique_complete",
        "conversation_history": messages,
    }


async def audit_node(state: GraphState) -> dict:
    """Node 2: Audit - The Auditor evaluates both proposals with full context.

    Gemini receives both proposals, the cross-critiques, and the system context
    to make a final assessment and determine if consensus is possible.
    """
    logger.info(f"[{state.session_id}] Starting audit phase")

    if not all([
        state.architect_proposal,
        state.engineer_proposal,
        state.architect_critique,
        state.engineer_critique,
    ]):
        raise ValueError("All proposals and critiques must exist for audit")

    # Prepare context
    context_str = (
        state.system_context.to_prompt_string()
        if state.system_context
        else "No additional context provided."
    )

    # Initialize auditor
    auditor_adapter = AuditorAdapter()

    # Format critique summaries
    architect_critique_str = (
        f"Strengths: {', '.join(state.architect_critique.strengths[:3])}\n"
        f"Weaknesses: {', '.join(state.architect_critique.weaknesses[:3])}\n"
        f"Concerns: {', '.join(state.architect_critique.concerns[:3])}\n"
        f"Agreement level: {state.architect_critique.agreement_level:.0%}"
    )
    engineer_critique_str = (
        f"Strengths: {', '.join(state.engineer_critique.strengths[:3])}\n"
        f"Weaknesses: {', '.join(state.engineer_critique.weaknesses[:3])}\n"
        f"Concerns: {', '.join(state.engineer_critique.concerns[:3])}\n"
        f"Agreement level: {state.engineer_critique.agreement_level:.0%}"
    )

    # Prepare audit prompt
    audit_prompt = AUDITOR_PROMPT.format(
        user_question=state.user_question,
        system_context=context_str,
        architect_title=state.architect_proposal.title,
        architect_summary=state.architect_proposal.summary,
        architect_approach=state.architect_proposal.approach,
        architect_confidence=f"{state.architect_proposal.confidence:.0%}",
        architect_components=_format_components(state.architect_proposal.components),
        architect_trade_offs=_format_trade_offs(state.architect_proposal.trade_offs),
        architect_risks=_format_risks(state.architect_proposal.risks),
        engineer_title=state.engineer_proposal.title,
        engineer_summary=state.engineer_proposal.summary,
        engineer_approach=state.engineer_proposal.approach,
        engineer_confidence=f"{state.engineer_proposal.confidence:.0%}",
        engineer_components=_format_components(state.engineer_proposal.components),
        engineer_trade_offs=_format_trade_offs(state.engineer_proposal.trade_offs),
        engineer_risks=_format_risks(state.engineer_proposal.risks),
        architect_critique=architect_critique_str,
        engineer_critique=engineer_critique_str,
    )

    audit_result = await auditor_adapter.generate_structured(
        prompt=audit_prompt,
        response_model=AuditResult,
        system_prompt=AUDITOR_SYSTEM_PROMPT,
        temperature=0.5,
    )

    logger.info(
        f"[{state.session_id}] Audit complete - "
        f"Preferred: {audit_result.preferred_approach}, "
        f"Consensus possible: {audit_result.consensus_possible}"
    )

    # Check for interrupt - DISABLING for direct judgment
    interrupt = None

    # Create message for history
    messages = [
        Message(
            role=MessageRole.AUDITOR,
            content=f"Audit complete. Preferred approach: {audit_result.preferred_approach}. "
                    f"Consensus possible: {audit_result.consensus_possible}. "
                    f"{audit_result.preference_rationale}",
            metadata={
                "consensus_possible": audit_result.consensus_possible,
                "security_issues_count": len(audit_result.security_issues),
            },
        ),
    ]

    return {
        "audit_result": audit_result,
        "consensus_reached": audit_result.consensus_possible,
        "current_phase": "audit_complete",
        "round_number": state.round_number + 1,
        "interrupt": interrupt,
        "conversation_history": messages,
    }


async def convergence_node(state: GraphState) -> dict:
    """Node 4: Convergence - Synthesize the final Architecture Decision Record.

    Creates the final ADR based on the audit results and proposals.
    """
    logger.info(f"[{state.session_id}] Starting convergence phase")

    if not all([state.architect_proposal, state.engineer_proposal, state.audit_result]):
        raise ValueError("Proposals and audit result must exist for convergence")

    # Determine the majority proposal
    if state.audit_result.preferred_approach == "architect":
        majority_proposal = state.architect_proposal
        minority_report = (
            f"The Engineer proposed: {state.engineer_proposal.title}\n"
            f"{state.engineer_proposal.summary}\n\n"
            f"Key differences: {', '.join(state.engineer_critique.concerns[:3])}"
        )
    elif state.audit_result.preferred_approach == "engineer":
        majority_proposal = state.engineer_proposal
        minority_report = (
            f"The Architect proposed: {state.architect_proposal.title}\n"
            f"{state.architect_proposal.summary}\n\n"
            f"Key differences: {', '.join(state.architect_critique.concerns[:3])}"
        )
    else:
        # Hybrid - prefer the one with higher confidence
        if state.architect_proposal.confidence >= state.engineer_proposal.confidence:
            majority_proposal = state.architect_proposal
            minority_report = f"The Engineer's proposal was partially incorporated. {state.audit_result.synthesis_recommendation}"
        else:
            majority_proposal = state.engineer_proposal
            minority_report = f"The Architect's proposal was partially incorporated. {state.audit_result.synthesis_recommendation}"

    # Calculate consensus level
    avg_agreement = (
        state.architect_critique.agreement_level + state.engineer_critique.agreement_level
    ) / 2
    consensus_level = avg_agreement if state.consensus_reached else avg_agreement * 0.5

    # Create the final ADR
    final_adr = ArchitectureDecisionRecord(
        title=f"ADR: {majority_proposal.title}",
        status="proposed",
        date=datetime.now(),
        decision=majority_proposal.summary,
        rationale=state.audit_result.preference_rationale,
        context=state.user_question,
        constraints=[c.description for c in (state.system_context.constraints if state.system_context else [])],
        majority_opinion=majority_proposal,
        minority_report=minority_report,
        alternatives_considered=[
            state.architect_proposal.approach,
            state.engineer_proposal.approach,
        ],
        positive_consequences=[t.rationale for t in majority_proposal.trade_offs[:3]],
        negative_consequences=state.audit_result.integration_concerns[:3],
        risks=[r.description for r in state.audit_result.risk_matrix[:5]],
        mermaid_diagram=majority_proposal.mermaid_diagram,
        consensus_level=consensus_level,
        rounds_taken=state.round_number,
    )

    logger.info(
        f"[{state.session_id}] Convergence complete - "
        f"Final ADR: {final_adr.title}, Consensus level: {consensus_level:.0%}"
    )

    # Create message for history
    messages = [
        Message(
            role=MessageRole.SYSTEM,
            content=f"Final decision: {final_adr.title}\n{final_adr.decision}",
            metadata={"consensus_level": consensus_level},
        ),
    ]

    return {
        "final_adr": final_adr,
        "current_phase": "complete",
        "conversation_history": messages,
    }


async def clarification_node(state: GraphState) -> dict:
    """Node 3: Clarification Loop - Handle user clarification responses.

    Processes the user's response to an interrupt and prepares to resume.
    If no user_response is provided yet, this indicates we're pausing for input.
    """
    logger.info(f"[{state.session_id}] Processing clarification")

    # If no user response yet, we're pausing to wait for input
    if not state.user_response:
        logger.info(f"[{state.session_id}] Waiting for user clarification response")
        # Return current state without changes - graph will pause here
        return {
            "current_phase": "waiting_for_clarification",
        }

    # Add user response to context if system_context exists
    additional_context = f"\n\nUser clarification: {state.user_response}"

    if state.system_context:
        current_additional = state.system_context.additional_context or ""
        state.system_context.additional_context = current_additional + additional_context
    else:
        state.system_context = SystemContext(additional_context=additional_context.strip())

    # Create message for history
    messages = [
        Message(
            role=MessageRole.USER,
            content=state.user_response,
        ),
    ]

    # Determine where to resume based on what phase we were in
    resume_phase = "ideation" if state.current_phase in ["start", "ideation_complete"] else "audit"

    logger.info(f"[{state.session_id}] Clarification processed, resuming at {resume_phase}")

    return {
        "interrupt": None,
        "user_response": None,
        "current_phase": resume_phase,
        "conversation_history": messages,
    }
