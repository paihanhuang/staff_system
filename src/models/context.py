"""System context models for providing background to agents."""

from typing import Optional

from pydantic import BaseModel, Field


class Constraint(BaseModel):
    """A constraint on the system design."""

    type: str = Field(
        ...,
        description="Type of constraint (e.g., 'budget', 'timeline', 'team_size', 'technology')",
    )
    description: str = Field(..., description="Description of the constraint")
    severity: str = Field(
        default="hard", description="Severity: 'hard' (must meet) or 'soft' (prefer to meet)"
    )
    value: Optional[str] = Field(default=None, description="Specific value if applicable")


class PerformanceSLA(BaseModel):
    """Performance SLA requirements."""

    metric: str = Field(..., description="The metric (e.g., 'latency', 'throughput', 'uptime')")
    target: str = Field(..., description="Target value (e.g., '<100ms', '99.99%')")
    priority: str = Field(default="high", description="Priority: 'critical', 'high', 'medium', 'low'")


class ExistingSystem(BaseModel):
    """Information about an existing system that must be integrated."""

    name: str = Field(..., description="Name of the existing system")
    type: str = Field(..., description="Type (e.g., 'database', 'api', 'service')")
    technology: str = Field(..., description="Technology stack")
    api_contracts: Optional[str] = Field(
        default=None, description="API contracts or interface specifications"
    )
    notes: Optional[str] = Field(default=None, description="Additional notes about integration")


class TeamInfo(BaseModel):
    """Information about the team that will build/maintain the system."""

    size: int = Field(..., description="Team size")
    expertise: list[str] = Field(
        default_factory=list, description="Areas of expertise (e.g., 'Python', 'Kubernetes', 'ML')"
    )
    experience_level: str = Field(
        default="senior", description="Overall experience level: 'junior', 'mid', 'senior', 'staff'"
    )


class SystemContext(BaseModel):
    """Complete context about the system being designed."""

    company_name: Optional[str] = Field(default=None, description="Company or project name")
    domain: Optional[str] = Field(
        default=None, description="Business domain (e.g., 'fintech', 'healthcare', 'e-commerce')"
    )
    current_tech_stack: list[str] = Field(
        default_factory=list, description="Current technology stack in use"
    )
    existing_systems: list[ExistingSystem] = Field(
        default_factory=list, description="Existing systems to integrate with"
    )
    constraints: list[Constraint] = Field(
        default_factory=list, description="Constraints on the design"
    )
    performance_slas: list[PerformanceSLA] = Field(
        default_factory=list, description="Performance SLA requirements"
    )
    team: Optional[TeamInfo] = Field(default=None, description="Team information")
    additional_context: Optional[str] = Field(
        default=None, description="Any additional context or requirements"
    )

    def to_prompt_string(self) -> str:
        """Convert context to a string suitable for inclusion in prompts."""
        parts = []

        if self.company_name:
            parts.append(f"Company/Project: {self.company_name}")

        if self.domain:
            parts.append(f"Domain: {self.domain}")

        if self.current_tech_stack:
            parts.append(f"Current Tech Stack: {', '.join(self.current_tech_stack)}")

        if self.existing_systems:
            systems_str = "\n".join(
                f"  - {s.name} ({s.type}): {s.technology}" for s in self.existing_systems
            )
            parts.append(f"Existing Systems:\n{systems_str}")

        if self.constraints:
            constraints_str = "\n".join(
                f"  - [{c.severity.upper()}] {c.type}: {c.description}" for c in self.constraints
            )
            parts.append(f"Constraints:\n{constraints_str}")

        if self.performance_slas:
            slas_str = "\n".join(
                f"  - {s.metric}: {s.target} (Priority: {s.priority})"
                for s in self.performance_slas
            )
            parts.append(f"Performance SLAs:\n{slas_str}")

        if self.team:
            parts.append(
                f"Team: {self.team.size} members, {self.team.experience_level} level, "
                f"expertise in {', '.join(self.team.expertise)}"
            )

        if self.additional_context:
            parts.append(f"Additional Context: {self.additional_context}")

        return "\n\n".join(parts) if parts else "No additional context provided."
