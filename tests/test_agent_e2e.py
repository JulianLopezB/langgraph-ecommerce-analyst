import os
import sys
from types import SimpleNamespace

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from agents.process_classifier import ProcessTypeResult
from domain.entities import ExecutionStatus, ProcessType
from workflow.graph import DataAnalysisAgent


def setup_common(monkeypatch, process_type):
    class DummyRepo:
        def get_table_schema(self, table):
            return [
                {
                    "name": "col",
                    "type": "INTEGER",
                    "mode": "NULLABLE",
                    "description": "",
                }
            ]

        def execute_query(self, sql):
            return pd.DataFrame({"col": [1, 2, 3]})

    repo = DummyRepo()
    monkeypatch.setattr("infrastructure.persistence.data_repository", repo)

    class DummyClassifier:
        def classify(self, prompt, schema):
            return ProcessTypeResult(
                process_type=process_type,
                confidence=0.9,
                reasoning="stub",
                complexity_level="low",
                suggested_tables=["orders"],
            )

    classifier = DummyClassifier()
    for path in [
        "agents.process_classifier.process_classifier",
        "workflow.nodes.query_understanding.process_classifier",
    ]:
        monkeypatch.setattr(path, classifier)

    def understand_data(prompt, schema, proc_type):
        return SimpleNamespace(
            query_intent="intent",
            relevant_tables=[SimpleNamespace(name="orders")],
            target_metrics=[SimpleNamespace(name="col")],
            grouping_dimensions=[],
            complexity_score=0.1,
        )

    monkeypatch.setattr(
        "agents.schema_agent.schema_agent.understand_data", understand_data
    )
    monkeypatch.setattr(
        "workflow.nodes.sql_generation.schema_agent",
        SimpleNamespace(understand_data=understand_data),
    )

    def generate_sql(prompt, data_understanding, process_result):
        return SimpleNamespace(
            sql_query="SELECT 1 as col",
            explanation="dummy",
            estimated_complexity="low",
            optimization_applied=[],
            tables_used=["orders"],
            metrics_computed=["col"],
            confidence=0.9,
        )

    monkeypatch.setattr("agents.sql_agent.sql_agent.generate_sql", generate_sql)
    monkeypatch.setattr(
        "workflow.nodes.sql_generation.sql_agent",
        SimpleNamespace(generate_sql=generate_sql),
    )

    class DummyLLM:
        def __init__(self):
            self.code = "print('hi')"

        def generate_insights(self, analysis_results, query):
            return "final insights"

        def generate_adaptive_python_code(self, analysis_context):
            return self.code

    llm = DummyLLM()
    monkeypatch.setattr("infrastructure.llm.llm_client", llm)

    return repo, classifier, llm


def test_data_analysis_agent_sql_e2e(monkeypatch):
    setup_common(monkeypatch, ProcessType.SQL)
    agent = DataAnalysisAgent()
    result = agent.analyze("How many orders?")
    assert result["workflow_complete"] is True
    assert result["insights"] == "final insights"
    assert isinstance(result["analysis_outputs"]["result_1"], pd.DataFrame)


def test_data_analysis_agent_python_e2e(monkeypatch):
    setup_common(monkeypatch, ProcessType.PYTHON)

    validation = SimpleNamespace(
        is_valid=True,
        security_score=1.0,
        syntax_errors=[],
        security_warnings=[],
        performance_warnings=[],
        validation_time=0.0,
    )

    class DummyValidator:
        def validate(self, code):
            return validation

    validator = DummyValidator()
    monkeypatch.setattr("infrastructure.execution.validator", validator)

    class DummyExecutor:
        def execute_code(self, code, context):
            return SimpleNamespace(
                status=ExecutionStatus.SUCCESS,
                output_data={"total": 6},
                execution_time=0.1,
                memory_used_mb=0.1,
                error_message=None,
                stdout="",
                stderr="",
            )

    executor = DummyExecutor()
    monkeypatch.setattr("infrastructure.execution.secure_executor", executor)

    agent = DataAnalysisAgent()
    result = agent.analyze("Run python analysis")
    assert result["workflow_complete"] is True
    assert result["analysis_outputs"]["python_results"] == {"total": 6}
    assert result["insights"] == "final insights"
