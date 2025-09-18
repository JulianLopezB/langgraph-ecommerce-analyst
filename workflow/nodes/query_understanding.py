"""Query understanding node."""
from datetime import datetime
from typing import Dict, Any
import re
import pandas as pd

from agents.process_classifier import process_classifier
from infrastructure.logging import get_logger
from tracing.langsmith_setup import tracer, trace_agent_operation
from workflow.state import AnalysisState
from domain.entities import ProcessType, ConversationMessage

logger = get_logger(__name__)


def _get_schema_info() -> Dict[str, Any]:
    """Get schema information for core tables."""
    try:
        # Import data_repository dynamically to avoid None reference
        from infrastructure.persistence import data_repository
        
        if data_repository is None:
            logger.error("Data repository is not initialized")
            return {}
            
        schema_info: Dict[str, Any] = {}
        core_tables = ["orders", "order_items", "products", "users"]
        for table in core_tables:
            try:
                columns = data_repository.get_table_schema(table)
                schema_info[table] = {"columns": columns}
            except Exception as e:
                logger.warning(f"Could not get schema for {table}: {e}")
        return schema_info
    except Exception as e:
        logger.error(f"Error getting schema info: {e}")
        return {}


def _build_contextual_prompt(
    history: list[ConversationMessage], user_query: str, max_history: int = 1
) -> str:
    """Combine recent conversation history with the current user query.

    Args:
        history: Full conversation history.
        user_query: Current user question.
        max_history: Number of most recent messages to include.

    Returns:
        A single string containing recent context followed by the new query.
    """
    recent: list[str] = []
    for message in history[-max_history:]:
        recent.append(f"{message.role.capitalize()}: {message.content}")
    recent.append(f"User: {user_query}")
    return "\n".join(recent)


def _resolve_context(state: AnalysisState) -> None:
    """Resolve references to previous artifacts in the user query.

    This allows follow-up questions like "the result" to be replaced with the
    most recent artifact key and optionally captures explicit naming requests.
    """

    query = state.get("user_query", "")
    artifacts = state.get("analysis_outputs", {})

    # Check for explicit naming e.g. "as my_result"
    name_match = re.search(r"\bas\s+(?P<name>[A-Za-z_]\w*)", query, re.IGNORECASE)
    if name_match:
        state["requested_artifact_name"] = name_match.group("name")
        query = (query[: name_match.start()].strip() + " " + query[name_match.end() :].strip()).strip()

    if artifacts:
        lower_query = query.lower()
        if "the result" in lower_query:
            last_key = None
            for key in reversed(list(artifacts.keys())):
                if isinstance(artifacts[key], pd.DataFrame):
                    last_key = key
                    break
            if last_key:
                query = re.sub(r"\bthe result\b", last_key, query, flags=re.IGNORECASE)
                referenced = artifacts.get(last_key)
                if referenced is not None:
                    state["raw_dataset"] = referenced
                    state["active_dataframe"] = last_key

    state["user_query"] = query


def understand_query(state: AnalysisState) -> AnalysisState:
    """Parse and understand user intent using AI agents."""
    with trace_agent_operation(
        name="understand_query_ai",
        user_query=state["user_query"],
        session_id=state["session_id"],
    ):
        logger.info("Understanding user query with AI agents")
        try:
            _resolve_context(state)

            schema_info = _get_schema_info()
            state["data_schema"] = schema_info

            contextual_prompt = _build_contextual_prompt(
                state.get("conversation_history", []), state["user_query"]
            )
            state["contextual_prompt"] = contextual_prompt

            process_result = process_classifier.classify(
                contextual_prompt, schema_info
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
                    "query_length": len(contextual_prompt),
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
                if (
                    state.get("raw_dataset") is not None
                    and process_result.process_type in {ProcessType.PYTHON, ProcessType.VISUALIZATION}
                ):
                    state["next_step"] = "generate_python_code"
                else:
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
