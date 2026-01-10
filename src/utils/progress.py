"""Real-time progress tracking for session state updates."""

from typing import Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
import threading

# Global progress store - thread-safe tracking of session progress
_progress_store: dict[str, dict] = {}
_progress_lock = threading.Lock()


def update_session_progress(session_id: str, **updates):
    """Update progress for a session in real-time.
    
    This allows nodes to update their progress status immediately,
    without waiting for the full node to complete.
    
    Args:
        session_id: The session ID to update.
        **updates: Key-value pairs to update (e.g., current_phase="ideation").
    """
    with _progress_lock:
        if session_id not in _progress_store:
            _progress_store[session_id] = {}
        _progress_store[session_id].update(updates)
        _progress_store[session_id]["last_updated"] = datetime.now().isoformat()


def get_session_progress(session_id: str) -> dict:
    """Get the current progress for a session.
    
    Args:
        session_id: The session ID.
        
    Returns:
        Dictionary of progress data.
    """
    with _progress_lock:
        return _progress_store.get(session_id, {}).copy()


def clear_session_progress(session_id: str):
    """Clear progress data for a session.
    
    Args:
        session_id: The session ID.
    """
    with _progress_lock:
        if session_id in _progress_store:
            del _progress_store[session_id]
