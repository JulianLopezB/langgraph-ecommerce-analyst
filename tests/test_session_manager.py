import os
import sys
from datetime import datetime
from unittest.mock import MagicMock
import pandas as pd


sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from workflow.graph import SessionManager, DataAnalysisAgent
from infrastructure.persistence.in_memory_session_store import InMemorySessionStore
from infrastructure.persistence import FilesystemArtifactStore


def create_manager(tmp_path=None):
    store = InMemorySessionStore()
    agent = MagicMock(spec=DataAnalysisAgent)
    artifact_store = FilesystemArtifactStore(base_path=str(tmp_path)) if tmp_path else FilesystemArtifactStore()

    def fake_analysis(
        user_query: str,
        session_id: str,
        conversation_history: list | None = None,
        artifacts: dict | None = None,
    ):

        return {
            "session_id": session_id,
            "conversation_history": [
                {
                    "timestamp": "2024-01-01T00:00:00",
                    "role": "assistant",
                    "content": f"Echo: {user_query}",
                    "type": "text",
                }
            ],
        }

    agent.analyze.side_effect = fake_analysis
    manager = SessionManager(session_store=store, agent=agent, artifact_store=artifact_store)
    return manager, store, agent, artifact_store


def test_start_session_initializes_metadata(tmp_path):
    manager, store, _, _ = create_manager(tmp_path)
    session_id = manager.start_session()
    session = store.get_session(session_id)

    assert session is not None
    assert session.session_id == session_id
    assert isinstance(session.created_at, datetime)
    assert session.analysis_count == 0
    assert session.conversation_history == []
    assert session.artifacts == {}


def test_analyze_query_updates_history_and_count(tmp_path):
    manager, store, agent, _ = create_manager(tmp_path)
    session_id = manager.start_session()

    result = manager.analyze_query("hello", session_id)
    agent.analyze.assert_called_once_with("hello", session_id, [], {})

    session = store.get_session(session_id)
    assert session.analysis_count == 1
    assert len(session.conversation_history) == 1
    assert result["conversation_history"][0]["content"] == "Echo: hello"


def test_analyze_query_merges_artifacts(tmp_path):
    manager, store, agent, artifact_store = create_manager(tmp_path)
    session_id = manager.start_session()

    df = pd.DataFrame({"a": [1, 2]})

    def analysis_with_artifacts(user_query: str, session_id: str, conversation_history=None, artifacts=None):
        return {
            "session_id": session_id,
            "conversation_history": [],
            "analysis_outputs": {"df": df},
        }

    agent.analyze.side_effect = analysis_with_artifacts
    manager.analyze_query("hi", session_id)

    session = store.get_session(session_id)
    meta = session.artifacts["df"]
    loaded = artifact_store.load_dataframe(meta["path"])
    pd.testing.assert_frame_equal(loaded, df)


def test_analyze_query_rehydrates_saved_artifacts(tmp_path):
    manager, store, agent, artifact_store = create_manager(tmp_path)
    session_id = manager.start_session()

    df = pd.DataFrame({"a": [1, 2]})
    meta = artifact_store.save_dataframe(df, "df")
    session = store.get_session(session_id)
    session.artifacts["df"] = meta
    store.save_session(session)

    def analysis_using_artifacts(user_query, session_id, conversation_history=None, artifacts=None):
        assert isinstance(artifacts["df"], pd.DataFrame)
        pd.testing.assert_frame_equal(artifacts["df"], df)
        return {"session_id": session_id, "conversation_history": []}

    agent.analyze.side_effect = analysis_using_artifacts
    manager.analyze_query("reuse", session_id)


def test_get_session_history_returns_expected_structure(tmp_path):
    manager, _, _, _ = create_manager(tmp_path)

    session_id = manager.start_session()
    manager.analyze_query("hello", session_id)

    history = manager.get_session_history(session_id)
    assert history["session_id"] == session_id
    assert history["analysis_count"] == 1
    assert len(history["conversation_history"]) == 1
    assert "created_at" in history


def test_list_sessions_returns_expected_structure(tmp_path):
    manager, _, _, _ = create_manager(tmp_path)

    session_id = manager.start_session()
    manager.analyze_query("hello", session_id)

    sessions = manager.list_sessions()
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == session_id
    assert sessions[0]["analysis_count"] == 1
    assert "created_at" in sessions[0]


def test_delete_session_removes_and_handles_unknown(tmp_path):
    manager, store, _, _ = create_manager(tmp_path)

    session_id = manager.start_session()

    assert manager.delete_session(session_id) is True
    assert store.get_session(session_id) is None
    assert manager.delete_session("unknown") is False
