"""Controller coordinating analysis requests between interfaces and workflow."""
from typing import Dict, Any

from workflow import AnalysisWorkflow


class AnalysisController:
    """Application-level API for analysis operations."""

    def __init__(self, workflow: AnalysisWorkflow | None = None) -> None:
        self._workflow = workflow or AnalysisWorkflow()

    def start_session(self) -> str:
        """Create a new analysis session."""
        return self._workflow.start_session()

    def analyze_query(self, user_query: str, session_id: str) -> Dict[str, Any]:
        """Run the analysis workflow for a query."""
        return self._workflow.analyze_query(user_query, session_id)

    def get_session_history(self, session_id: str) -> Dict[str, Any]:
        """Return conversation history for a session."""
        return self._workflow.get_session_history(session_id)

    def list_sessions(self) -> list[Dict[str, Any]]:
        """List all active sessions."""
        return self._workflow.list_sessions()

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and its history."""
        return self._workflow.delete_session(session_id)
