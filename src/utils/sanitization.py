"""Input sanitization utilities for protection against prompt injection."""

import re
from dataclasses import dataclass
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger()


@dataclass
class SanitizationResult:
    """Result of sanitization with metadata."""

    original: str
    sanitized: str
    was_modified: bool
    warnings: list[str]


# Patterns that may indicate prompt injection attempts
DANGEROUS_PATTERNS = [
    # Direct instruction override attempts
    (r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)", "instruction_override"),
    (r"disregard\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)", "instruction_override"),
    (r"forget\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)", "instruction_override"),
    
    # System prompt extraction attempts
    (r"(what|show|reveal|display|print|output)\s+(is\s+)?(your|the)\s+(system\s+)?prompt", "prompt_extraction"),
    (r"(repeat|echo|recite)\s+(your\s+)?(system\s+)?(prompt|instructions)", "prompt_extraction"),
    
    # Role manipulation
    (r"you\s+are\s+now\s+(a|an)\s+", "role_manipulation"),
    (r"pretend\s+(you\'?re?|to\s+be)\s+", "role_manipulation"),
    (r"act\s+as\s+if\s+you\s+are", "role_manipulation"),
    
    # Delimiter injection (trying to break out of user input)
    (r"```\s*(system|assistant|user)\s*[\n:]", "delimiter_injection"),
    (r"<\|?(system|assistant|user|im_start|im_end)\|?>", "delimiter_injection"),
    
    # Jailbreak keywords
    (r"DAN\s+mode", "jailbreak_attempt"),
    (r"developer\s+mode\s+(enabled|on|active)", "jailbreak_attempt"),
]

# Maximum allowed question length
MAX_QUESTION_LENGTH = 10000  # characters
MIN_QUESTION_LENGTH = 10  # characters


class SanitizationError(Exception):
    """Raised when input cannot be safely sanitized."""

    pass


def detect_dangerous_patterns(text: str) -> list[tuple[str, str]]:
    """Detect potentially dangerous patterns in text.

    Args:
        text: The text to analyze.

    Returns:
        List of (matched_text, pattern_type) tuples.
    """
    findings = []
    text_lower = text.lower()

    for pattern, pattern_type in DANGEROUS_PATTERNS:
        matches = re.finditer(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            findings.append((match.group(), pattern_type))

    return findings


def sanitize_user_input(
    text: str,
    max_length: int = MAX_QUESTION_LENGTH,
    strip_dangerous: bool = True,
    log_warnings: bool = True,
) -> SanitizationResult:
    """Sanitize user input for safe use in prompts.

    Args:
        text: The raw user input.
        max_length: Maximum allowed length.
        strip_dangerous: Whether to remove dangerous patterns.
        log_warnings: Whether to log warnings for dangerous patterns.

    Returns:
        SanitizationResult with sanitized text and metadata.
    """
    original = text
    warnings: list[str] = []
    was_modified = False

    # Strip whitespace
    text = text.strip()
    if text != original.strip():
        was_modified = True

    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length]
        warnings.append(f"Input truncated from {len(original)} to {max_length} characters")
        was_modified = True
        if log_warnings:
            logger.warning(warnings[-1])

    # Detect dangerous patterns
    dangerous_findings = detect_dangerous_patterns(text)

    if dangerous_findings:
        pattern_types = set(ptype for _, ptype in dangerous_findings)
        warnings.append(f"Detected potentially dangerous patterns: {pattern_types}")
        if log_warnings:
            logger.warning(f"Potential prompt injection attempt detected: {pattern_types}")

        if strip_dangerous:
            # Replace dangerous patterns with sanitized versions
            for matched_text, pattern_type in dangerous_findings:
                text = text.replace(matched_text, f"[REDACTED:{pattern_type}]")
                was_modified = True

    # Normalize whitespace (collapse multiple spaces/newlines)
    text = re.sub(r'\s+', ' ', text)
    if text != original:
        was_modified = True

    # Remove null bytes and other control characters (except newlines)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    return SanitizationResult(
        original=original,
        sanitized=text,
        was_modified=was_modified,
        warnings=warnings,
    )


def validate_question(
    text: str,
    min_length: int = MIN_QUESTION_LENGTH,
    max_length: int = MAX_QUESTION_LENGTH,
    require_question_mark: bool = False,
) -> tuple[bool, Optional[str]]:
    """Validate a user question meets requirements.

    Args:
        text: The question text.
        min_length: Minimum allowed length.
        max_length: Maximum allowed length.
        require_question_mark: Whether to require a question mark.

    Returns:
        Tuple of (is_valid, error_message).
    """
    if not text or not text.strip():
        return False, "Question cannot be empty"

    text = text.strip()

    if len(text) < min_length:
        return False, f"Question must be at least {min_length} characters"

    if len(text) > max_length:
        return False, f"Question must not exceed {max_length} characters"

    if require_question_mark and "?" not in text:
        return False, "Question must contain a question mark"

    # Check if it's just whitespace or punctuation
    if not re.search(r'[a-zA-Z0-9]', text):
        return False, "Question must contain alphanumeric characters"

    return True, None


def escape_for_prompt(text: str) -> str:
    """Escape text for safe inclusion in prompts.

    Adds markers to clearly delineate user input from prompt structure.

    Args:
        text: The text to escape.

    Returns:
        Escaped text with clear boundaries.
    """
    # Wrap in clear delimiters
    return f"<user_input>\n{text}\n</user_input>"


def sanitize_and_validate(
    text: str,
    min_length: int = MIN_QUESTION_LENGTH,
    max_length: int = MAX_QUESTION_LENGTH,
) -> str:
    """Sanitize and validate user input, raising on invalid input.

    Args:
        text: The raw user input.
        min_length: Minimum allowed length.
        max_length: Maximum allowed length.

    Returns:
        Sanitized text.

    Raises:
        SanitizationError: If input is invalid.
    """
    # First validate
    is_valid, error = validate_question(text, min_length, max_length)
    if not is_valid:
        raise SanitizationError(error)

    # Then sanitize
    result = sanitize_user_input(text, max_length)

    return result.sanitized
