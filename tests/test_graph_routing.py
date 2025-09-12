import os
import sys
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from workflow.graph import DataAnalysisAgent


@pytest.mark.parametrize(
    "method_name, step",
    [
        ("_route_after_understanding", "generate_sql"),
        ("_route_after_sql_generation", "execute_sql"),
        ("_route_after_sql_execution", "generate_python_code"),
        ("_route_after_code_generation", "validate_code"),
        ("_route_after_validation", "execute_code"),
        ("_route_after_execution", "synthesize_results"),
    ],
)
def test_route_after_methods(method_name, step):
    method = getattr(DataAnalysisAgent, method_name)

    state_with_next = {"next_step": step}
    assert method(None, state_with_next) == step

    state_without_next = {}
    assert method(None, state_without_next) == "handle_error"
