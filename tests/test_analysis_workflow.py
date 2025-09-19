import os
import sys
from unittest.mock import Mock

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from application.orchestrators.analysis_workflow import AnalysisWorkflow
from domain.entities import ProcessType


def make_workflow(classification_output):
    schema_analysis = Mock()
    schema_analysis.analyze.return_value = {"schema": "info"}
    process_classification = Mock()
    process_classification.classify.return_value = classification_output
    sql_generation = Mock()
    sql_generation.generate.return_value = "SELECT 1"
    python_generation = Mock()
    python_generation.generate.return_value = "print('hi')"
    validation = Mock()
    validation.validate.return_value = True
    execution = Mock()
    execution.run_query.return_value = [1]
    execution.execute_code.return_value = [2]
    synthesis = Mock()
    synthesis.synthesize.return_value = "insight"

    # Mock pipeline components
    llm_client = Mock()
    validator = Mock()
    executor = Mock()

    workflow = AnalysisWorkflow(
        schema_analysis,
        process_classification,
        sql_generation,
        python_generation,
        validation,
        execution,
        synthesis,
        llm_client,
        validator,
        executor,
    )
    return workflow, python_generation, execution, sql_generation, validation


@pytest.mark.parametrize("raw_output", ["ProcessType.PYTHON", "Python analysis needed"])
def test_python_branch_triggered_for_normalized_process_type(raw_output):
    workflow, python_generation, execution, _, _ = make_workflow(raw_output)
    workflow.run("question")
    python_generation.generate.assert_called_once()
    execution.execute_code.assert_called_once()


def test_python_branch_skipped_when_not_python():
    workflow, python_generation, execution, sql_generation, _ = make_workflow(
        ProcessType.SQL
    )
    workflow.run("question")
    python_generation.generate.assert_not_called()
    execution.execute_code.assert_not_called()
    sql_generation.generate.assert_called_once()
    execution.run_query.assert_called_once()


def test_validation_failure_raises_value_error():
    workflow, python_generation, execution, sql_generation, validation = make_workflow(
        ProcessType.PYTHON
    )
    validation.validate.return_value = False
    with pytest.raises(RuntimeError):
        workflow.run("question")
    python_generation.generate.assert_called_once()
    validation.validate.assert_called_once()
    execution.execute_code.assert_not_called()
