"""Session state container for CLI interactions."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

try:
    from langchain_core.messages import BaseMessage
except Exception:  # pragma: no cover - fallback if langchain_core not installed
    BaseMessage = Dict[str, str]  # type: ignore

Message = Union[Dict[str, str], BaseMessage]


@dataclass
class SessionState:
    """Holds conversation messages and the latest result for a session."""

    messages: List[Message] = field(default_factory=list)
    last_result: Any | None = None

    def append_user(self, message: str) -> None:
        """Record a user message."""
        self.messages.append({"role": "user", "content": message})

    def append_agent(self, message: str) -> None:
        """Record a message from the agent/assistant."""
        self.messages.append({"role": "assistant", "content": message})

    def update_result(self, result: Any | None) -> None:
        """Store the latest analysis result."""
        self.last_result = result


def create_session_state() -> SessionState:
    """Factory for initializing a new :class:`SessionState`."""
    return SessionState()
