import os
import sys
from datetime import datetime
from unittest.mock import patch

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from workflow.graph import SessionManager, DataAnalysisAgent
from infrastructure.persistence.in_memory_session_store import InMemorySessionStore


@pytest.fixture
def manager_and_store():
    """Provide a session manager with an in-memory store."""
    store = InMemorySessionStore()
    agent = DataAnalysisAgent()
    manager = SessionManager(session_store=store, agent=agent)
    return manager, store


def fake_analyze(user_query: str, session_id: str):
    """Deterministic analysis result for testing."""
    return {
        "session_id": session_id,
        "conversation_history": [
            {
                "timestamp": datetime(2024, 1, 1).isoformat(),
                "role": "assistant",
                "content": f"response to {user_query}",
                "type": "text",
            }
        ],
    }


def test_start_session_and_list_sessions(manager_and_store):
    manager, store = manager_and_store
    session_id1 = manager.start_session()
    session_id2 = manager.start_session()

    assert store.get_session(session_id1) is not None
    assert store.get_session(session_id2) is not None

    sessions = manager.list_sessions()
    ids = {s["session_id"] for s in sessions}
    assert {session_id1, session_id2}.issubset(ids)


def test_analyze_query_and_unknown_session(manager_and_store):
    manager, store = manager_and_store

    with patch.object(manager.agent, "analyze", side_effect=fake_analyze):
        session_id = manager.start_session()
        result = manager.analyze_query("hello", session_id)
        assert result["session_id"] == session_id

        session = store.get_session(session_id)
        assert session.analysis_count == 1
        assert len(session.conversation_history) == 1

        new_id = "missing-session"
        result2 = manager.analyze_query("hi", new_id)
        assert result2["session_id"] == new_id
        assert store.get_session(new_id) is not None


def test_get_session_history_and_unknown(manager_and_store):
    manager, store = manager_and_store

    with patch.object(manager.agent, "analyze", side_effect=fake_analyze):
        session_id = manager.start_session()
        manager.analyze_query("question", session_id)

    history = manager.get_session_history(session_id)
    assert history["session_id"] == session_id
    assert len(history["conversation_history"]) == 1

    assert manager.get_session_history("unknown") == {"error": "Session not found"}


def test_delete_session_and_unknown(manager_and_store):
    manager, _ = manager_and_store
    session_id = manager.start_session()
    assert manager.delete_session(session_id) is True
    assert manager.delete_session("does-not-exist") is False


def test_route_after_methods_default_to_handle_error():
    agent = DataAnalysisAgent()
    route_methods = [
        agent._route_after_understanding,
        agent._route_after_sql_generation,
        agent._route_after_sql_execution,
        agent._route_after_code_generation,
        agent._route_after_validation,
        agent._route_after_execution,
    ]

    for method in route_methods:
        assert method({"next_step": "continue"}) == "continue"
        assert method({}) == "handle_error"

