import os
import sys
from types import SimpleNamespace
from unittest.mock import Mock
import pandas as pd
import logging
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from infrastructure.persistence.bigquery import BigQueryRepository
from infrastructure.llm.gemini import GeminiClient
from infrastructure.execution.executor import SecureExecutor
from infrastructure.config import ExecutionLimits
from domain.entities import ExecutionStatus


class DummyTracer:
    def trace_bigquery_operation(self, **kwargs):
        from contextlib import contextmanager
        @contextmanager
        def cm():
            yield
        return cm()

    def log_metrics(self, metrics):
        pass


def dummy_contextmanager(*args, **kwargs):
    from contextlib import contextmanager
    @contextmanager
    def cm():
        yield
    return cm()


def test_bigquery_repository(monkeypatch):
    import infrastructure.persistence.bigquery as bigquery_module
    mock_df = pd.DataFrame({"a": [1]})
    mock_query_job = Mock()
    mock_query_job.result.return_value.to_dataframe.return_value = mock_df
    mock_client = Mock()
    mock_client.query.return_value = mock_query_job
    mock_client.get_table.return_value = SimpleNamespace(schema=[
        SimpleNamespace(name="id", field_type="INTEGER", mode="NULLABLE", description="id")
    ])

    monkeypatch.setattr(bigquery_module, "bigquery", SimpleNamespace(Client=lambda project=None: mock_client))
    monkeypatch.setattr(bigquery_module, "tracer", DummyTracer())

    repo = BigQueryRepository(project_id="pid", dataset_id="ds")
    df = repo.execute_query("SELECT 1")
    assert df.equals(mock_df)
    schema = repo.get_table_schema("orders")
    assert schema[0]["name"] == "id"


def test_bigquery_repository_query_error(monkeypatch, caplog):
    import infrastructure.persistence.bigquery as bigquery_module
    mock_client = Mock()
    mock_client.query.side_effect = Exception("query boom")
    monkeypatch.setattr(
        bigquery_module, "bigquery", SimpleNamespace(Client=lambda project=None: mock_client)
    )
    monkeypatch.setattr(bigquery_module, "tracer", DummyTracer())
    repo = BigQueryRepository(project_id="pid", dataset_id="ds")
    with caplog.at_level(logging.ERROR):
        with pytest.raises(Exception, match="query boom"):
            repo.execute_query("SELECT 1")
    assert "BigQuery execution failed: query boom" in caplog.text


def test_bigquery_repository_get_table_schema_error(monkeypatch, caplog):
    import infrastructure.persistence.bigquery as bigquery_module
    mock_client = Mock()
    mock_client.get_table.side_effect = Exception("schema boom")
    monkeypatch.setattr(
        bigquery_module, "bigquery", SimpleNamespace(Client=lambda project=None: mock_client)
    )
    monkeypatch.setattr(bigquery_module, "tracer", DummyTracer())
    repo = BigQueryRepository(project_id="pid", dataset_id="ds")
    with caplog.at_level(logging.ERROR):
        with pytest.raises(Exception, match="schema boom"):
            repo.get_table_schema("orders")
    assert "Failed to get schema for table orders: schema boom" in caplog.text


def test_gemini_client(monkeypatch):
    import infrastructure.llm.gemini as gemini_module

    os.environ['GEMINI_API_KEY'] = 'test'

    dummy_model = Mock()
    dummy_response = SimpleNamespace(
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text="hi")]))]
    )
    dummy_model.generate_content.return_value = dummy_response

    mock_genai = SimpleNamespace(
        configure=lambda api_key=None: None,
        GenerativeModel=lambda model_name, safety_settings: dummy_model,
        types=SimpleNamespace(GenerationConfig=lambda **kwargs: kwargs),
    )

    class HC:
        HARM_CATEGORY_HATE_SPEECH = 1
        HARM_CATEGORY_DANGEROUS_CONTENT = 2
        HARM_CATEGORY_SEXUALLY_EXPLICIT = 3
        HARM_CATEGORY_HARASSMENT = 4

    class HB:
        BLOCK_MEDIUM_AND_ABOVE = 1

    monkeypatch.setattr(gemini_module, "genai", mock_genai)
    monkeypatch.setattr(gemini_module, "tracer", DummyTracer())
    monkeypatch.setattr(gemini_module, "trace_llm_operation", dummy_contextmanager)
    monkeypatch.setattr(gemini_module, "HarmCategory", HC)
    monkeypatch.setattr(gemini_module, "HarmBlockThreshold", HB)

    client = GeminiClient(api_key="test")
    response = client.generate_text("prompt")
    assert response.content == "hi"
    dummy_model.generate_content.assert_called_once()


def test_secure_executor():
    limits = ExecutionLimits(max_execution_time=5, max_memory_mb=256, max_output_size_mb=1)
    executor = SecureExecutor(limits)
    # Avoid importing heavy optional libraries during tests
    executor._create_safe_globals = lambda: {"__builtins__": {"print": print}}
    executor._set_resource_limits = lambda: None
    code = "print('hi')\nanalysis_results={'x':1}"
    result = executor.execute_code(code)
    assert result.status == ExecutionStatus.SUCCESS
    assert result.output_data == {'x':1}
    assert 'hi' in result.stdout


def _minimal_executor(limits: ExecutionLimits) -> SecureExecutor:
    """Create a SecureExecutor with minimal builtins for testing."""
    executor = SecureExecutor(limits)
    safe_builtins = {
        'print': print,
        'range': range,
        'len': len,
        'int': int,
        'float': float,
        'list': list,
        'dict': dict,
        'set': set,
        'bool': bool,
    }
    executor._create_safe_globals = lambda: {"__builtins__": safe_builtins}
    executor._set_resource_limits = lambda: None
    return executor


def test_secure_executor_timeout():
    limits = ExecutionLimits(max_execution_time=1, max_memory_mb=512, max_output_size_mb=1)
    executor = _minimal_executor(limits)
    code = "while True:\n    pass"
    result = executor.execute_code(code)
    assert result.status == ExecutionStatus.TIMEOUT


def test_secure_executor_memory_limit():
    limits = ExecutionLimits(max_execution_time=5, max_memory_mb=256, max_output_size_mb=1)
    executor = _minimal_executor(limits)
    code = "x = [0] * (10**10)"  # Attempt to allocate huge memory
    result = executor.execute_code(code)
    assert result.status == ExecutionStatus.FAILED
    assert 'memory limit' in (result.error_message or '').lower()


def test_secure_executor_zero_division():
    limits = ExecutionLimits(max_execution_time=5, max_memory_mb=512, max_output_size_mb=1)
    executor = _minimal_executor(limits)
    code = "1/0"
    result = executor.execute_code(code)
    assert result.status == ExecutionStatus.FAILED
    assert 'ZeroDivisionError' in result.stderr


def test_secure_executor_import_blocked():
    limits = ExecutionLimits(max_execution_time=5, max_memory_mb=512, max_output_size_mb=1)
    executor = _minimal_executor(limits)
    code = "__import__('os')"
    result = executor.execute_code(code)
    assert result.status == ExecutionStatus.FAILED
    assert '__import__' in (result.error_message or '')
    limits = ExecutionLimits(max_execution_time=5, max_memory_mb=50, max_output_size_mb=1)
    executor = SecureExecutor(limits)
    executor._set_resource_limits = lambda: None
    executor._create_safe_globals = lambda: {"__builtins__": {"print": print, "MemoryError": MemoryError}}
    code = "raise MemoryError('too big')"
    result = executor.execute_code(code)
    assert result.status == ExecutionStatus.FAILED
    assert "memory limit" in result.error_message.lower()


def test_secure_executor_exception():
    limits = ExecutionLimits(max_execution_time=5, max_memory_mb=256, max_output_size_mb=1)
    executor = SecureExecutor(limits)
    executor._set_resource_limits = lambda: None
    executor._create_safe_globals = lambda: {"__builtins__": {"print": print}}
    code = "1/0"
    result = executor.execute_code(code)
    assert result.status == ExecutionStatus.FAILED
    assert "zerodivisionerror" in result.stderr.lower()
