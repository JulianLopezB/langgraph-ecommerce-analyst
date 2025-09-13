from abc import ABC, abstractmethod
from typing import List, Optional

from domain.entities import AnalysisSession


class SessionStore(ABC):
    """Abstract persistence interface for analysis sessions."""

    @abstractmethod
    def save_session(self, session: AnalysisSession) -> None:
        """Persist a session instance, including any cached artifacts."""
        raise NotImplementedError

    @abstractmethod
    def get_session(self, session_id: str) -> Optional[AnalysisSession]:
        """Retrieve a session by its identifier with all artifacts."""
        raise NotImplementedError

    @abstractmethod
    def list_sessions(self) -> List[AnalysisSession]:
        """Return all stored sessions."""
        raise NotImplementedError

    @abstractmethod
    def delete_session(self, session_id: str) -> bool:
        """Remove a session from storage."""
        raise NotImplementedError
