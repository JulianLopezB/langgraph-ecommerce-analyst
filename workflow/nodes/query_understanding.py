"""Query understanding node."""
from datetime import datetime
from typing import Dict, Any

from agents.process_classifier import process_classifier
from infrastructure.persistence import data_repository
from infrastructure.logging import get_logger
from tracing.langsmith_setup import tracer, trace_agent_operation
from workflow.state import AnalysisState
from domain.entities import ProcessType, ConversationMessage

logger = get_logger(__name__)
data_repo = data_repository


def _get_schema_info() -> Dict[str, Any]:
    """Get schema information for core tables."""
    try:
        schema_info: Dict[str, Any] = {}
        core_tables = ["orders", "order_items", "products", "users"]
        for table in core_tables:
            try:
                columns = data_repo.get_table_schema(table)
                schema_info[table] = {"columns": columns}
            except Exception as e:
                logger.warning(f"Could not get schema for {table}: {e}")
        return schema_info
    except Exception as e:
        logger.error(f"Error getting schema info: {e}")
        return {}


def understand_query(state: AnalysisState) -> AnalysisState:
    """Parse and understand user intent using AI agents."""
    with trace_agent_operation(
        name="understand_query_ai",
        user_query=state["user_query"],
        session_id=state["session_id"],
    ):
        logger.info("Understanding user query with AI agents")
        try:
            schema_info = _get_schema_info()
            state["data_schema"] = schema_info

            process_result = process_classifier.classify(
                state["user_query"], schema_info
            )

            state["process_type"] = process_result.process_type
            state["confidence_score"] = process_result.confidence
            state["needs_python_analysis"] = (
                process_result.process_type == ProcessType.PYTHON
            )

            state["analysis_outputs"]["process_data"] = {
                "process_type": process_result.process_type.value,
                "confidence": process_result.confidence,
                "reasoning": process_result.reasoning,
                "complexity_level": process_result.complexity_level,
                "suggested_tables": process_result.suggested_tables,
            }

            tracer.log_metrics(
                {
                    "process_type": process_result.process_type.value,
                    "confidence_score": process_result.confidence,
                    "needs_python_analysis": state["needs_python_analysis"],
                    "query_length": len(state["user_query"]),
                    "complexity_level": process_result.complexity_level,
                }
            )

            message = ConversationMessage(
                timestamp=datetime.now(),
                role="assistant",
                content=(
                    f"I'll handle this using {process_result.process_type.value.upper()} "
                    f"processing. {process_result.reasoning}"
                ),
                message_type="query",
            )
            state["conversation_history"].append(message)

            if process_result.confidence > 0.7:
                if process_result.process_type == ProcessType.SQL:
                    state["next_step"] = "generate_sql"
                elif process_result.process_type == ProcessType.PYTHON:
                    state["next_step"] = "generate_sql"
                elif process_result.process_type == ProcessType.VISUALIZATION:
                    state["next_step"] = "generate_sql"
            else:
                state["next_step"] = "clarify_query"

            logger.info(
                "Process type classified: %s (confidence: %.2f)",
                process_result.process_type.value,
                process_result.confidence,
            )
        except Exception as e:
            logger.error(f"Error understanding query: {str(e)}")
            state["error_context"]["understanding_error"] = str(e)
            state["next_step"] = "handle_error"
        return state
