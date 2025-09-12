"""Command-line utilities for interactive analysis sessions.

This module provides a very small interactive loop that can be used for
manual testing or demonstrations.  It keeps track of the conversation in a
``SessionState`` object so that each user message and assistant response are
remembered locally.  The heavy lifting of the analysis is delegated to the
application's analysis controller through :func:`run_analysis`.

Example
-------
>>> from interface.cli import start_cli
>>> start_cli()  # doctest: +SKIP
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app_factory import create_analysis_controller


@dataclass
class Message:
    """A single conversational message."""

    role: str
    content: str


@dataclass
class SessionState:
    """Hold conversation history and last workflow result."""

    messages: List[Message] = field(default_factory=list)
    last_result: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None


_controller = None


def _get_controller():
    """Lazily instantiate the analysis controller."""
    global _controller
    if _controller is None:
        _controller = create_analysis_controller()
    return _controller


def run_analysis(user_input: str, state: SessionState) -> str:
    """Run analysis for ``user_input`` and update ``state`` with results.

    The function ensures that a session is started and stores the most recent
    analysis result so it is available for future calls.
    """

    controller = _get_controller()

    if state.session_id is None:
        state.session_id = controller.start_session()

    result = controller.analyze_query(user_input, state.session_id)
    state.last_result = result

    # Return the assistant message (insights) if available, otherwise a default
    return result.get("insights", "")


def start_cli() -> None:
    """Start a simple interactive CLI session.

    The conversation history and last result are preserved in ``SessionState``
    until the user issues the ``reset`` command.
    """

    state = SessionState()
    print("Type 'reset' to start over or 'exit' to quit.")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        command = user_input.lower()
        if command in {"exit", "quit"}:
            break
        if command == "reset":
            state = SessionState()
            print("🔄 Conversation reset. How can I help?")
            continue

        state.messages.append(Message(role="user", content=user_input))
        assistant_message = run_analysis(user_input, state)
        state.messages.append(Message(role="assistant", content=assistant_message))
        print(f"Assistant: {assistant_message}")


__all__ = ["SessionState", "run_analysis", "start_cli", "Message"]

