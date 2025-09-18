"""High-level workflow interface for analysis sessions."""

from typing import Dict, Any, Optional

from .graph import SessionManager


class AnalysisWorkflow:
    """Facade for orchestrating analysis sessions."""

    def __init__(self, session_manager: Optional[SessionManager] = None) -> None:
        """Create workflow with optional custom session manager."""
        self._session_manager = session_manager or SessionManager()

    def start_session(self, session_id: str | None = None) -> str:
        """Start a new analysis session."""
        return self._session_manager.start_session(session_id)

    def analyze_query(self, user_query: str, session_id: str) -> Dict[str, Any]:
        """Run analysis for the given query within a session."""
        return self._session_manager.analyze_query(user_query, session_id)

    def get_session_history(self, session_id: str) -> Dict[str, Any]:
        """Retrieve conversation history for a session."""
        return self._session_manager.get_session_history(session_id)

    def list_sessions(self) -> list[Dict[str, Any]]:
        """List active analysis sessions."""
        return self._session_manager.list_sessions()

    def delete_session(self, session_id: str) -> bool:
        """Remove a session and its history."""
        return self._session_manager.delete_session(session_id)
