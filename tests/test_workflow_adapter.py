import os
import sys
from unittest.mock import Mock

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from application.orchestrators.analysis_workflow import (
    AnalysisWorkflow,
    create_workflow_adapter,
)


def test_workflow_adapter_sets_insights_key():
    workflow = Mock(spec=AnalysisWorkflow)
    workflow.run.return_value = "insight"
    adapter = create_workflow_adapter(workflow)
    state = {"user_query": "what is the revenue?"}
    result = adapter(state)
    workflow.run.assert_called_once_with("what is the revenue?")
    assert result["insights"] == "insight"
