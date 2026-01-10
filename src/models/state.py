"""State models for the LangGraph state machine."""

from datetime import datetime
from enum import Enum
from typing import Annotated, Optional

from pydantic import BaseModel, Field

from src.models.context import SystemContext
from src.models.proposal import ArchitectureProposal, AuditResult, Critique


class MessageRole(str, Enum):
    """Role of a message in the conversation."""

    USER = "user"
    SUPERVISOR = "supervisor"
    ARCHITECT = "architect"
    ENGINEER = "engineer"
    AUDITOR = "auditor"
    SYSTEM = "system"


class Message(BaseModel):
    """A message in the conversation history."""

    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: dict = Field(default_factory=dict)


class InterruptType(str, Enum):
    """Types of interrupts that can pause the workflow."""

    CLARIFICATION_NEEDED = "clarification_needed"
    APPROVAL_REQUIRED = "approval_required"
    FEEDBACK_REQUESTED = "feedback_requested"


class Interrupt(BaseModel):
    """An interrupt that pauses the workflow for user input."""

    type: InterruptType
    source: str = Field(..., description="Which agent triggered the interrupt")
    question: str = Field(..., description="The question for the user")
    context: Optional[str] = Field(
        default=None, description="Additional context for the question"
    )
    options: list[str] = Field(
        default_factory=list, description="Suggested options for the user"
    )
    timestamp: datetime = Field(default_factory=datetime.now)


class ArchitectureDecisionRecord(BaseModel):
    """The final Architecture Decision Record (ADR)."""

    title: str = Field(..., description="Title of the ADR")
    status: str = Field(default="proposed", description="Status: proposed, accepted, deprecated")
    date: datetime = Field(default_factory=datetime.now)

    # Decision
    decision: str = Field(..., description="The architectural decision made")
    rationale: str = Field(..., description="Why this decision was made")

    # Context
    context: str = Field(..., description="The context and problem statement")
    constraints: list[str] = Field(default_factory=list, description="Constraints considered")

    # Options considered
    majority_opinion: ArchitectureProposal = Field(
        ..., description="The proposal that received majority support"
    )
    minority_report: Optional[str] = Field(
        default=None, description="Dissenting opinions and their rationale"
    )
    alternatives_considered: list[str] = Field(
        default_factory=list, description="Alternative approaches considered"
    )

    # Consequences
    positive_consequences: list[str] = Field(default_factory=list)
    negative_consequences: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)

    # Visualization
    mermaid_diagram: Optional[str] = Field(default=None, description="Architecture diagram")

    # Metadata
    consensus_level: float = Field(
        ..., ge=0.0, le=1.0, description="Level of consensus achieved (0-1)"
    )
    rounds_taken: int = Field(..., description="Number of rounds to reach decision")


def merge_messages(
    existing: list[Message], new: list[Message]
) -> list[Message]:
    """Merge message lists, appending new messages."""
    return existing + new


class GraphState(BaseModel):
    """The state object for the LangGraph state machine."""

    # Session info
    session_id: str = Field(..., description="Unique session identifier")

    # Input
    user_question: str = Field(..., description="The user's system design question")
    system_context: Optional[SystemContext] = Field(
        default=None, description="Optional system context"
    )

    # Proposals from ideation phase
    architect_proposal: Optional[ArchitectureProposal] = Field(
        default=None, description="Proposal from The Architect (o3)"
    )
    engineer_proposal: Optional[ArchitectureProposal] = Field(
        default=None, description="Proposal from The Engineer (Claude)"
    )

    # Critiques from cross-critique phase
    architect_critique: Optional[Critique] = Field(
        default=None, description="Architect's critique of Engineer's proposal"
    )
    engineer_critique: Optional[Critique] = Field(
        default=None, description="Engineer's critique of Architect's proposal"
    )

    # Audit from auditor phase
    audit_result: Optional[AuditResult] = Field(
        default=None, description="Audit result from The Auditor (Gemini)"
    )

    # Consensus tracking
    consensus_reached: bool = Field(default=False, description="Whether consensus was reached")
    round_number: int = Field(default=0, description="Current round number")
    max_rounds: int = Field(default=3, description="Maximum rounds before escalation")

    # Model overrides (user-selectable)
    architect_model: Optional[str] = Field(default=None, description="Override model for Architect")
    engineer_model: Optional[str] = Field(default=None, description="Override model for Engineer")
    auditor_model: Optional[str] = Field(default=None, description="Override model for Auditor")

    # Interrupt handling
    interrupt: Optional[Interrupt] = Field(
        default=None, description="Current interrupt waiting for user input"
    )
    user_response: Optional[str] = Field(
        default=None, description="User's response to an interrupt"
    )

    # Final output
    final_adr: Optional[ArchitectureDecisionRecord] = Field(
        default=None, description="The final Architecture Decision Record"
    )

    # Observability
    conversation_history: Annotated[list[Message], merge_messages] = Field(
        default_factory=list, description="Full conversation history"
    )

    # Current phase tracking
    current_phase: str = Field(
        default="start", description="Current phase of the workflow"
    )
    error: Optional[str] = Field(default=None, description="Error message if any")

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True
