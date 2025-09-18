import os
import sys
from unittest.mock import Mock

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from interface.cli.interface import DataAnalysisCLI


def test_cli_typical_session(monkeypatch, capsys):
    controller = Mock()
    controller.start_session.return_value = "sess1"
    controller.analyze_query.return_value = {"insights": "great"}
    controller.get_session_history.return_value = {}

    inputs = iter(["What are sales?", "exit"])
    monkeypatch.setattr(
        "interface.cli.interface.prompt", lambda *args, **kwargs: next(inputs)
    )
    monkeypatch.setattr("interface.cli.session.start_session", lambda ctrl: "sess1")
    monkeypatch.setattr(
        "interface.cli.session.analyze_query_with_progress",
        lambda console, user_query, session_id, ctrl: {"insights": "great"},
    )
    monkeypatch.setattr(
        "interface.cli.display.display_results",
        lambda console, results: console.print(f"Insights: {results['insights']}"),
    )

    cli = DataAnalysisCLI(controller)
    cli.start_interactive_session()

    captured = capsys.readouterr()
    assert "Insights: great" in captured.out
    assert "Thanks for using the Data Analysis Agent" in captured.out
