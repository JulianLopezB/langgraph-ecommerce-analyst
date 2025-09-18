import os
import sys
from unittest.mock import Mock

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from application.controllers import AnalysisController
from workflow import AnalysisWorkflow


@pytest.fixture
def controller_workflow():
    workflow = Mock(spec=AnalysisWorkflow)
    controller = AnalysisController(workflow=workflow)
    return controller, workflow


def test_start_session(controller_workflow):
    controller, workflow = controller_workflow
    workflow.start_session.return_value = "session-id"

    result = controller.start_session()

    workflow.start_session.assert_called_once_with()
    assert result == "session-id"


def test_analyze_query(controller_workflow):
    controller, workflow = controller_workflow
    workflow.analyze_query.return_value = {"answer": 42}

    result = controller.analyze_query("query", "sess")

    workflow.analyze_query.assert_called_once_with("query", "sess")
    assert result == {"answer": 42}


def test_get_session_history(controller_workflow):
    controller, workflow = controller_workflow
    workflow.get_session_history.return_value = {"history": []}

    result = controller.get_session_history("sess")

    workflow.get_session_history.assert_called_once_with("sess")
    assert result == {"history": []}


def test_list_sessions(controller_workflow):
    controller, workflow = controller_workflow
    workflow.list_sessions.return_value = [1]

    result = controller.list_sessions()

    workflow.list_sessions.assert_called_once_with()
    assert result == [1]


def test_delete_session(controller_workflow):
    controller, workflow = controller_workflow
    workflow.delete_session.return_value = True

    result = controller.delete_session("sess")

    workflow.delete_session.assert_called_once_with("sess")
    assert result is True
