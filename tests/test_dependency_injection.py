import pytest
from unittest.mock import MagicMock

from tracing.langsmith_setup import LangSmithTracer
from services.llm_service import GeminiService
from bq_client import BigQueryRunner
from agents.process_classifier import ProcessTypeClassifier
from agents.schema_agent import SchemaIntelligenceAgent
from agents.sql_agent import SQLGenerationAgent
from code_generation.validators import CodeValidator
from execution.sandbox import SecureExecutor
from workflow.nodes import WorkflowNodes
from workflow.graph import DataAnalysisAgent


def test_workflow_nodes_dependency_injection():
    tracer = LangSmithTracer()
    llm = MagicMock(spec=GeminiService)
    bq = MagicMock(spec=BigQueryRunner)
    pc = MagicMock(spec=ProcessTypeClassifier)
    sa = MagicMock(spec=SchemaIntelligenceAgent)
    sqla = MagicMock(spec=SQLGenerationAgent)
    validator = MagicMock(spec=CodeValidator)
    executor = MagicMock(spec=SecureExecutor)

    nodes = WorkflowNodes(llm, bq, pc, sa, sqla, validator, executor, tracer)

    assert nodes.llm_service is llm
    assert nodes.bq_client is bq
    assert nodes.process_classifier is pc
    assert nodes.schema_agent is sa
    assert nodes.sql_agent is sqla
    assert nodes.validator is validator
    assert nodes.secure_executor is executor


def test_data_analysis_agent_dependency_injection():
    tracer = LangSmithTracer()
    llm = MagicMock(spec=GeminiService)
    bq = MagicMock(spec=BigQueryRunner)
    pc = MagicMock(spec=ProcessTypeClassifier)
    sa = MagicMock(spec=SchemaIntelligenceAgent)
    sqla = MagicMock(spec=SQLGenerationAgent)
    validator = MagicMock(spec=CodeValidator)
    executor = MagicMock(spec=SecureExecutor)

    agent = DataAnalysisAgent(
        tracer=tracer,
        llm_service=llm,
        bq_client=bq,
        process_classifier=pc,
        schema_agent=sa,
        sql_agent=sqla,
        validator=validator,
        secure_executor=executor,
    )

    assert agent.llm_service is llm
    assert agent.workflow_nodes.process_classifier is pc
    assert agent.workflow_nodes.bq_client is bq
