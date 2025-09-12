import pytest
from unittest.mock import MagicMock

from interface.cli.session import (
    start_session,
    analyze_query_with_progress,
    get_session_history,
)


def test_start_session_delegates_to_controller():
    controller = MagicMock()
    controller.start_session.return_value = "sess"

    result = start_session(controller)

    assert result == "sess"
    controller.start_session.assert_called_once_with()


def test_get_session_history_delegates_to_controller():
    controller = MagicMock()
    controller.get_session_history.return_value = {"items": []}

    result = get_session_history("sess", controller)

    assert result == {"items": []}
    controller.get_session_history.assert_called_once_with("sess")


class DummyProgress:
    def __init__(self, *args, **kwargs):
        self.records = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def add_task(self, description, total=None):
        self.records.append(description)
        return 1

    def update(self, task, description):
        self.records.append(description)


class DummyThread:
    def __init__(self, target, daemon=None):
        self._target = target

    def start(self):
        self._target()


def test_analyze_query_with_progress(monkeypatch):
    controller = MagicMock()
    controller.analyze_query.return_value = {"res": 1}

    progress = DummyProgress()
    monkeypatch.setattr("interface.cli.session.Progress", lambda *args, **kwargs: progress)
    monkeypatch.setattr("interface.cli.session.time.sleep", lambda x: None)
    monkeypatch.setattr("interface.cli.session.threading.Thread", DummyThread)

    console = MagicMock()
    result = analyze_query_with_progress(console, "Q?", "sess", controller)

    assert result == {"res": 1}
    controller.analyze_query.assert_called_once_with("Q?", "sess")

    expected_steps = [
        "🧠 Understanding your question...",
        "🎯 Determining analysis approach...",
        "🔍 Analyzing data schema...",
        "⚡ Creating database query...",
        "📊 Executing query...",
        "🤖 Generating analysis code...",
        "🔬 Running calculations...",
        "✨ Preparing insights...",
        "✅ Analysis complete!",
    ]

    assert progress.records[1:] == expected_steps
