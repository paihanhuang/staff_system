"""Architecture proposal and audit models."""

from typing import Optional

from pydantic import BaseModel, Field


class Component(BaseModel):
    """A component in the proposed architecture."""

    name: str = Field(..., description="Name of the component")
    type: str = Field(
        ..., description="Type of component (e.g., 'service', 'database', 'queue', 'cache')"
    )
    technology: str = Field(..., description="Technology or framework to use")
    description: str = Field(..., description="Description of the component's purpose")
    dependencies: list[str] = Field(
        default_factory=list, description="Names of other components this depends on"
    )


class TradeOff(BaseModel):
    """A trade-off decision in the architecture."""

    aspect: str = Field(..., description="The aspect being traded off (e.g., 'consistency')")
    choice: str = Field(..., description="The choice made")
    rationale: str = Field(..., description="Why this choice was made")
    alternatives: list[str] = Field(default_factory=list, description="Alternatives considered")


class Risk(BaseModel):
    """A risk identified in the architecture."""

    category: str = Field(
        ..., description="Category of risk (e.g., 'security', 'scalability', 'cost', 'complexity')"
    )
    severity: str = Field(..., description="Severity level (low, medium, high, critical)")
    description: str = Field(..., description="Description of the risk")
    mitigation: str = Field(..., description="Proposed mitigation strategy")
    likelihood: str = Field(default="medium", description="Likelihood of occurrence")


class ArchitectureProposal(BaseModel):
    """A complete architecture proposal from an AI agent."""

    title: str = Field(..., description="Short title for the proposal")
    summary: str = Field(..., description="Executive summary of the proposal")
    approach: str = Field(..., description="The overall architectural approach")
    components: list[Component] = Field(default_factory=list, description="List of components")
    trade_offs: list[TradeOff] = Field(default_factory=list, description="Trade-off decisions")
    risks: list[Risk] = Field(default_factory=list, description="Identified risks")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score from 0.0 to 1.0"
    )
    uncertainties: list[str] = Field(
        default_factory=list, description="Areas of uncertainty that may need clarification"
    )
    mermaid_diagram: Optional[str] = Field(
        default=None, description="Mermaid.js diagram code for visualization"
    )
    clarification_needed: Optional[str] = Field(
        default=None, description="Question for the user if clarification is needed"
    )


class Critique(BaseModel):
    """A critique of another agent's proposal."""

    target_proposal: str = Field(..., description="Title of the proposal being critiqued")
    strengths: list[str] = Field(default_factory=list, description="Strengths of the proposal")
    weaknesses: list[str] = Field(default_factory=list, description="Weaknesses identified")
    concerns: list[str] = Field(default_factory=list, description="Specific concerns")
    suggestions: list[str] = Field(default_factory=list, description="Improvement suggestions")
    agreement_level: float = Field(
        ..., ge=0.0, le=1.0, description="Level of agreement with the proposal (0-1)"
    )


class AuditResult(BaseModel):
    """Result of the auditor's analysis of both proposals."""

    preferred_approach: str = Field(
        ..., description="Which approach is preferred ('architect', 'engineer', or 'hybrid')"
    )
    preference_rationale: str = Field(..., description="Why this approach is preferred")
    risk_matrix: list[Risk] = Field(
        default_factory=list, description="Combined risk analysis"
    )
    integration_concerns: list[str] = Field(
        default_factory=list, description="Concerns about integration with existing systems"
    )
    security_issues: list[str] = Field(
        default_factory=list, description="Security issues identified"
    )
    scalability_assessment: str = Field(..., description="Assessment of scalability")
    consensus_possible: bool = Field(
        ..., description="Whether consensus between proposals is possible"
    )
    synthesis_recommendation: Optional[str] = Field(
        default=None, description="Recommendation for synthesizing the proposals"
    )
    clarification_needed: Optional[str] = Field(
        default=None, description="Question for the user if clarification is needed"
    )
