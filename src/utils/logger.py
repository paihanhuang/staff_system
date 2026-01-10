"""Logging utilities for observability."""

import logging
import sys
from datetime import datetime
from typing import Optional

# Global logger instance
_logger: Optional[logging.Logger] = None


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with colors."""
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Set up logging configuration."""
    global _logger

    # Create logger
    logger = logging.getLogger("synapse_council")
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_format = ColoredFormatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Log everything to file
        file_format = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    _logger = logger
    return logger


def get_logger() -> logging.Logger:
    """Get the global logger instance."""
    global _logger
    if _logger is None:
        _logger = setup_logging()
    return _logger


class SessionLogger:
    """Logger for tracking a specific session's activity."""

    def __init__(self, session_id: str):
        """Initialize session logger."""
        self.session_id = session_id
        self.logger = get_logger()
        self.events: list[dict] = []

    def log_event(
        self,
        event_type: str,
        agent: Optional[str] = None,
        message: str = "",
        metadata: Optional[dict] = None,
    ) -> None:
        """Log a session event."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "event_type": event_type,
            "agent": agent,
            "message": message,
            "metadata": metadata or {},
        }
        self.events.append(event)
        self.logger.info(f"[{self.session_id}] {event_type}: {message}")

    def log_proposal(self, agent: str, proposal_title: str, confidence: float) -> None:
        """Log a proposal submission."""
        self.log_event(
            "proposal_submitted",
            agent=agent,
            message=f"Proposal '{proposal_title}' with confidence {confidence:.2%}",
        )

    def log_critique(self, agent: str, target_proposal: str, agreement: float) -> None:
        """Log a critique submission."""
        self.log_event(
            "critique_submitted",
            agent=agent,
            message=f"Critique of '{target_proposal}' with {agreement:.2%} agreement",
        )

    def log_interrupt(self, agent: str, interrupt_type: str, question: str) -> None:
        """Log an interrupt request."""
        self.log_event(
            "interrupt_triggered",
            agent=agent,
            message=f"[{interrupt_type}] {question}",
        )

    def log_user_response(self, response: str) -> None:
        """Log a user's response to an interrupt."""
        self.log_event(
            "user_response",
            message=response[:100] + "..." if len(response) > 100 else response,
        )

    def log_consensus(self, reached: bool, rounds: int) -> None:
        """Log consensus status."""
        status = "reached" if reached else "not reached"
        self.log_event(
            "consensus_status",
            message=f"Consensus {status} after {rounds} rounds",
        )

    def get_events(self) -> list[dict]:
        """Get all logged events."""
        return self.events.copy()
