"""FastAPI backend for the Synapse Council."""

from src.api.main import app
from src.api.routes import router
from src.api.handlers import SessionManager

__all__ = [
    "app",
    "router",
    "SessionManager",
]
