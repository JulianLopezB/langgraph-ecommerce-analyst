import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from workflow.graph import SessionManager
from infrastructure.persistence.in_memory_session_store import InMemorySessionStore


class FakeAgent:
    def analyze(self, user_query: str, session_id: str):
        return {
            "session_id": session_id,
            "conversation_history": [
                {
                    "timestamp": datetime.now().isoformat(),
                    "role": "assistant",
                    "content": "hi",
                    "type": "text",
                }
            ],
        }


def test_session_manager_stores_history_and_counts():
    store = InMemorySessionStore()
    manager = SessionManager(session_store=store, agent=FakeAgent())

    session_id = manager.start_session()
    result = manager.analyze_query("hello", session_id)

    assert result["session_id"] == session_id
    session = store.get_session(session_id)
    assert session is not None
    assert session.analysis_count == 1
    assert len(session.conversation_history) == 1


def test_session_manager_delete_session():
    store = InMemorySessionStore()
    manager = SessionManager(session_store=store, agent=FakeAgent())
    session_id = manager.start_session()
    assert manager.delete_session(session_id) is True
    assert store.get_session(session_id) is None
