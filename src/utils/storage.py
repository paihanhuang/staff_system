"""Session storage for persistence across server restarts."""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Any

from pydantic import BaseModel

from src.utils.logger import get_logger

logger = get_logger()

# Default storage directory
DEFAULT_STORAGE_DIR = Path("data/sessions")
DEFAULT_TTL_DAYS = 7


class SessionData(BaseModel):
    """Serializable session data for storage."""
    
    session_id: str
    question: str
    created_at: str
    updated_at: str
    state_json: dict
    conversation_turns: list[dict] = []
    is_complete: bool = False
    

class SessionStorage:
    """File-based session storage."""
    
    def __init__(self, storage_dir: Optional[Path] = None, ttl_days: int = DEFAULT_TTL_DAYS):
        """Initialize storage.
        
        Args:
            storage_dir: Directory to store session files.
            ttl_days: Days to keep sessions before cleanup.
        """
        self.storage_dir = storage_dir or DEFAULT_STORAGE_DIR
        self.ttl_days = ttl_days
        self._ensure_storage_dir()
    
    def _ensure_storage_dir(self) -> None:
        """Create storage directory if it doesn't exist."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Session storage initialized at {self.storage_dir}")
    
    def _session_path(self, session_id: str) -> Path:
        """Get the path for a session file."""
        return self.storage_dir / f"{session_id}.json"
    
    async def save_session(self, session_data: SessionData) -> None:
        """Save a session to storage.
        
        Args:
            session_data: Session data to save.
        """
        path = self._session_path(session_data.session_id)
        try:
            with open(path, "w") as f:
                json.dump(session_data.model_dump(), f, indent=2, default=str)
            logger.info(f"Saved session {session_data.session_id} to storage")
        except Exception as e:
            logger.error(f"Failed to save session {session_data.session_id}: {e}")
            raise
    
    async def load_session(self, session_id: str) -> Optional[SessionData]:
        """Load a session from storage.
        
        Args:
            session_id: Session ID to load.
            
        Returns:
            Session data if found, None otherwise.
        """
        path = self._session_path(session_id)
        if not path.exists():
            return None
        
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return SessionData(**data)
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session from storage.
        
        Args:
            session_id: Session ID to delete.
            
        Returns:
            True if deleted, False if not found.
        """
        path = self._session_path(session_id)
        if not path.exists():
            return False
        
        try:
            path.unlink()
            logger.info(f"Deleted session {session_id} from storage")
            return True
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False
    
    async def list_sessions(self) -> list[dict]:
        """List all stored sessions.
        
        Returns:
            List of session metadata dicts.
        """
        sessions = []
        for path in self.storage_dir.glob("*.json"):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                sessions.append({
                    "session_id": data.get("session_id"),
                    "question": data.get("question", "")[:100],
                    "created_at": data.get("created_at"),
                    "updated_at": data.get("updated_at"),
                    "is_complete": data.get("is_complete", False),
                    "turns": len(data.get("conversation_turns", [])),
                })
            except Exception as e:
                logger.warning(f"Failed to read session file {path}: {e}")
        
        return sorted(sessions, key=lambda x: x.get("updated_at", ""), reverse=True)
    
    async def cleanup_expired(self) -> int:
        """Clean up sessions older than TTL.
        
        Returns:
            Number of sessions cleaned up.
        """
        cutoff = datetime.now() - timedelta(days=self.ttl_days)
        cleaned = 0
        
        for path in self.storage_dir.glob("*.json"):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                
                updated_at = datetime.fromisoformat(data.get("updated_at", ""))
                if updated_at < cutoff:
                    path.unlink()
                    logger.info(f"Cleaned up expired session: {path.stem}")
                    cleaned += 1
            except Exception as e:
                logger.warning(f"Error during cleanup of {path}: {e}")
        
        return cleaned


# Global storage instance
_storage: Optional[SessionStorage] = None


def get_storage() -> SessionStorage:
    """Get the global storage instance."""
    global _storage
    if _storage is None:
        _storage = SessionStorage()
    return _storage
