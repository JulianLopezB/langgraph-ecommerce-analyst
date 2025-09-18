import os
import sys
from unittest.mock import Mock

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from application.use_cases import (
    execution,
    python_generation,
    sql_generation,
    validation,
)


def test_sql_generation_use_case():
    llm = Mock()
    llm.generate_code.return_value = "SELECT 1"
    uc = sql_generation.SQLGenerationUseCase(llm)
    result = uc.generate("question", {"schema": "info"})
    assert result == "SELECT 1"
    llm.generate_code.assert_called_once()


def test_python_generation_use_case():
    llm = Mock()
    llm.generate_code.return_value = "print('hi')"
    uc = python_generation.PythonGenerationUseCase(llm)
    result = uc.generate("prompt")
    assert result == "print('hi')"
    llm.generate_code.assert_called_once_with("prompt")


def test_code_validation_use_case():
    validator = Mock()
    validator.validate.return_value = True
    uc = validation.CodeValidationUseCase(validator)
    assert uc.validate("code") is True
    validator.validate.assert_called_once_with("code")


def test_code_execution_use_case():
    executor_mock = Mock()
    executor_mock.execute.return_value = {"result": 1}
    repo_mock = Mock()
    repo_mock.run_query.return_value = [1]
    uc = execution.CodeExecutionUseCase(executor_mock, repo_mock)
    assert uc.execute_code("print(1)", {"a": 1}) == {"result": 1}
    executor_mock.execute.assert_called_once_with("print(1)", {"a": 1})
    assert uc.run_query("SELECT 1") == [1]
    repo_mock.run_query.assert_called_once_with("SELECT 1")
