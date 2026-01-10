"""System prompts for all agents."""

from src.prompts.supervisor import SUPERVISOR_SYSTEM_PROMPT
from src.prompts.architect import ARCHITECT_SYSTEM_PROMPT
from src.prompts.engineer import ENGINEER_SYSTEM_PROMPT
from src.prompts.auditor import AUDITOR_SYSTEM_PROMPT

__all__ = [
    "SUPERVISOR_SYSTEM_PROMPT",
    "ARCHITECT_SYSTEM_PROMPT",
    "ENGINEER_SYSTEM_PROMPT",
    "AUDITOR_SYSTEM_PROMPT",
]
