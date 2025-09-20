import logging
import os
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from domain.entities import ExecutionStatus
from pydantic import BaseModel
from infrastructure.config import ExecutionLimits
from infrastructure.execution.executor import SecureExecutor
from infrastructure.llm.gemini import GeminiClient
from infrastructure.persistence.bigquery import BigQueryRepository


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
    mock_client.get_table.return_value = SimpleNamespace(
        schema=[
            SimpleNamespace(
                name="id", field_type="INTEGER", mode="NULLABLE", description="id"
            )
        ]
    )

    monkeypatch.setattr(
        bigquery_module,
        "bigquery",
        SimpleNamespace(Client=lambda project=None: mock_client),
    )
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
        bigquery_module,
        "bigquery",
        SimpleNamespace(Client=lambda project=None: mock_client),
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
        bigquery_module,
        "bigquery",
        SimpleNamespace(Client=lambda project=None: mock_client),
    )
    monkeypatch.setattr(bigquery_module, "tracer", DummyTracer())
    repo = BigQueryRepository(project_id="pid", dataset_id="ds")
    with caplog.at_level(logging.ERROR):
        with pytest.raises(Exception, match="schema boom"):
            repo.get_table_schema("orders")
    assert "Failed to get schema for table orders: schema boom" in caplog.text


def test_gemini_client(monkeypatch):
    import infrastructure.llm.gemini as gemini_module

    os.environ["GEMINI_API_KEY"] = "test"

    mock_genai = SimpleNamespace(configure=lambda api_key=None: None)

    class HC:
        HARM_CATEGORY_HATE_SPEECH = 1
        HARM_CATEGORY_DANGEROUS_CONTENT = 2
        HARM_CATEGORY_SEXUALLY_EXPLICIT = 3
        HARM_CATEGORY_HARASSMENT = 4

    class HB:
        BLOCK_MEDIUM_AND_ABOVE = 1

    class DummyLangChainModel:
        def __init__(self):
            self.prompts = []

        def __or__(self, parser):
            assert isinstance(parser, gemini_module.StrOutputParser)

            class DummyChain:
                def __init__(self_outer, outer):
                    self_outer.outer = outer

                def invoke(self_outer, prompt):
                    self_outer.outer.prompts.append(prompt)
                    return "hi"

            return DummyChain(self)

        def with_structured_output(self, schema):
            class DummyStructured:
                def invoke(self_inner, prompt):
                    raise AssertionError("Structured output should not be used")

            return DummyStructured()

    dummy_model = DummyLangChainModel()

    monkeypatch.setattr(gemini_module, "genai", mock_genai)
    monkeypatch.setattr(gemini_module, "tracer", DummyTracer())
    monkeypatch.setattr(gemini_module, "trace_llm_operation", dummy_contextmanager)
    monkeypatch.setattr(gemini_module, "HarmCategory", HC)
    monkeypatch.setattr(gemini_module, "HarmBlockThreshold", HB)
    monkeypatch.setattr(
        gemini_module.GeminiClient,
        "_create_langchain_model",
        lambda self, temperature, max_tokens: dummy_model,
    )

    client = GeminiClient(api_key="test")
    response = client.generate_text("prompt")
    assert response.content == "hi"
    assert dummy_model.prompts == ["prompt"]


def test_gemini_client_structured(monkeypatch):
    import infrastructure.llm.gemini as gemini_module

    os.environ["GEMINI_API_KEY"] = "test"

    mock_genai = SimpleNamespace(configure=lambda api_key=None: None)

    class HC:
        HARM_CATEGORY_HATE_SPEECH = 1
        HARM_CATEGORY_DANGEROUS_CONTENT = 2
        HARM_CATEGORY_SEXUALLY_EXPLICIT = 3
        HARM_CATEGORY_HARASSMENT = 4

    class HB:
        BLOCK_MEDIUM_AND_ABOVE = 1

    class DummyLangChainModel:
        def __init__(self, structured_response):
            self.structured_response = structured_response
            self.prompts = []

        def __or__(self, parser):
            raise AssertionError("Text chain not expected in structured test")

        def with_structured_output(self, schema):
            assert schema is StructuredResponse

            class DummyStructured:
                def __init__(self_outer, outer):
                    self_outer.outer = outer

                def invoke(self_outer, prompt):
                    self_outer.outer.prompts.append(prompt)
                    return self_outer.outer.structured_response

            return DummyStructured(self)

    class StructuredResponse(BaseModel):
        value: int = 1

    dummy_model = DummyLangChainModel(StructuredResponse(value=5))

    monkeypatch.setattr(gemini_module, "genai", mock_genai)
    monkeypatch.setattr(gemini_module, "tracer", DummyTracer())
    monkeypatch.setattr(gemini_module, "trace_llm_operation", dummy_contextmanager)
    monkeypatch.setattr(gemini_module, "HarmCategory", HC)
    monkeypatch.setattr(gemini_module, "HarmBlockThreshold", HB)
    monkeypatch.setattr(
        gemini_module.GeminiClient,
        "_create_langchain_model",
        lambda self, temperature, max_tokens: dummy_model,
    )

    client = GeminiClient(api_key="test")
    response = client.generate_structured("prompt", StructuredResponse)
    assert isinstance(response, StructuredResponse)
    assert response.value == 5
    assert dummy_model.prompts == ["prompt"]


def test_secure_executor():
    limits = ExecutionLimits(
        max_execution_time=5, max_memory_mb=256, max_output_size_mb=1
    )
    executor = SecureExecutor(limits)
    # Avoid importing heavy optional libraries during tests
    executor._create_safe_globals = lambda: {"__builtins__": {"print": print}}
    executor._set_resource_limits = lambda: None
    code = "print('hi')\nanalysis_results={'x':1}"
    result = executor.execute_code(code)
    assert result.status == ExecutionStatus.SUCCESS
    assert result.output_data == {"x": 1}
    assert "hi" in result.stdout


def _minimal_executor(limits: ExecutionLimits) -> SecureExecutor:
    """Create a SecureExecutor with minimal builtins for testing."""
    executor = SecureExecutor(limits)
    safe_builtins = {
        "print": print,
        "range": range,
        "len": len,
        "int": int,
        "float": float,
        "list": list,
        "dict": dict,
        "set": set,
        "bool": bool,
    }
    executor._create_safe_globals = lambda: {"__builtins__": safe_builtins}
    executor._set_resource_limits = lambda: None
    return executor


def test_secure_executor_timeout():
    limits = ExecutionLimits(
        max_execution_time=1, max_memory_mb=512, max_output_size_mb=1
    )
    executor = _minimal_executor(limits)
    code = "while True:\n    pass"
    result = executor.execute_code(code)
    assert result.status == ExecutionStatus.TIMEOUT


def test_secure_executor_memory_limit():
    limits = ExecutionLimits(
        max_execution_time=5, max_memory_mb=256, max_output_size_mb=1
    )
    executor = _minimal_executor(limits)
    code = "x = [0] * (10**10)"  # Attempt to allocate huge memory
    result = executor.execute_code(code)
    assert result.status == ExecutionStatus.FAILED
    assert "memory limit" in (result.error_message or "").lower()


def test_secure_executor_zero_division():
    limits = ExecutionLimits(
        max_execution_time=5, max_memory_mb=512, max_output_size_mb=1
    )
    executor = _minimal_executor(limits)
    code = "1/0"
    result = executor.execute_code(code)
    assert result.status == ExecutionStatus.FAILED
    assert "ZeroDivisionError" in result.stderr


def test_secure_executor_exception():
    limits = ExecutionLimits(
        max_execution_time=5, max_memory_mb=256, max_output_size_mb=1
    )
    executor = SecureExecutor(limits)
    executor._set_resource_limits = lambda: None
    executor._create_safe_globals = lambda: {"__builtins__": {"print": print}}
    code = "1/0"
    result = executor.execute_code(code)
    assert result.status == ExecutionStatus.FAILED
    assert "zerodivisionerror" in result.stderr.lower()
