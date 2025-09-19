import os
import sys

import pandas as pd
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from workflow.nodes import execution, query_understanding
from workflow.state import create_initial_state


def test_artifact_reference_reuses_dataframe(monkeypatch):
    # Simulate first query producing a DataFrame artifact
    df = pd.DataFrame({"a": [1, 2]})

    state = create_initial_state("Make a churn analysis", "s1")
    state["sql_query"] = "SELECT * FROM table"

    class DummyRepo:
        def execute_query(self, q):
            return df

    # Mock the persistence.data_repository instead of a non-existent data_repo attribute
    monkeypatch.setattr("infrastructure.persistence.data_repository", DummyRepo())
    execution.execute_sql(state)

    assert "result_1" in state["analysis_outputs"]

    # Second query referencing "the result"
    new_state = create_initial_state(
        "Create a plot on the result", "s1", artifacts=state["analysis_outputs"]
    )

    query_understanding._resolve_context(new_state)

    assert new_state["user_query"] == "Create a plot on result_1"
    assert new_state["active_dataframe"] == "result_1"
    assert new_state["raw_dataset"].equals(df)
