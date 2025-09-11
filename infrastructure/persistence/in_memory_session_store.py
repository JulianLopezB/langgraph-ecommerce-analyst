from typing import Dict, List, Optional

from domain.entities import AnalysisSession
from domain.services import SessionStore


class InMemorySessionStore(SessionStore):
    """Simple in-memory session storage used as a legacy shim."""

    def __init__(self) -> None:
        self._sessions: Dict[str, AnalysisSession] = {}

    def save_session(self, session: AnalysisSession) -> None:
        self._sessions[session.session_id] = session

    def get_session(self, session_id: str) -> Optional[AnalysisSession]:
        return self._sessions.get(session_id)

    def list_sessions(self) -> List[AnalysisSession]:
        return list(self._sessions.values())

    def delete_session(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None
