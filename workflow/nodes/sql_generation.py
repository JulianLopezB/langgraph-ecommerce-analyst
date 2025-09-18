"""SQL generation node."""

from datetime import datetime

from agents.process_classifier import ProcessTypeResult
from agents.schema_agent import schema_agent
from agents.sql_agent import sql_agent
from infrastructure.logging import get_logger
from workflow.state import AnalysisState
from domain.entities import ConversationMessage

logger = get_logger(__name__)


def generate_sql(state: AnalysisState) -> AnalysisState:
    """Generate SQL query using AI agents."""
    logger.info("Generating SQL query using AI agents")

    try:
        process_data = state["analysis_outputs"]["process_data"]
        process_result = ProcessTypeResult(
            process_type=state["process_type"],
            confidence=state["confidence_score"],
            reasoning=process_data["reasoning"],
            complexity_level=process_data["complexity_level"],
            suggested_tables=process_data["suggested_tables"],
        )

        contextual_prompt = state.get("contextual_prompt", state["user_query"])
        data_understanding = schema_agent.understand_data(
            contextual_prompt,
            state["data_schema"],
            state["process_type"],
        )

        state["analysis_outputs"]["data_understanding"] = {
            "query_intent": data_understanding.query_intent,
            "relevant_tables": [
                table.name for table in data_understanding.relevant_tables
            ],
            "target_metrics": [
                metric.name for metric in data_understanding.target_metrics
            ],
            "grouping_dimensions": [
                dim.name for dim in data_understanding.grouping_dimensions
            ],
            "complexity_score": data_understanding.complexity_score,
        }

        sql_result = sql_agent.generate_sql(
            contextual_prompt,
            data_understanding,
            process_result,
        )

        state["sql_query"] = sql_result.sql_query
        state["analysis_outputs"]["sql_metadata"] = {
            "explanation": sql_result.explanation,
            "complexity": sql_result.estimated_complexity,
            "optimizations": sql_result.optimization_applied,
            "tables_used": sql_result.tables_used,
            "metrics_computed": sql_result.metrics_computed,
            "confidence": sql_result.confidence,
        }

        message = ConversationMessage(
            timestamp=datetime.now(),
            role="assistant",
            content=f"Generated optimized SQL query: {sql_result.explanation}",
            message_type="query",
        )
        state["conversation_history"].append(message)

        state["next_step"] = "execute_sql"
        logger.info(
            "AI-generated SQL complete: %s complexity, confidence: %.2f",
            sql_result.estimated_complexity,
            sql_result.confidence,
        )

    except Exception as e:
        logger.error(f"Error generating SQL with AI agents: {str(e)}")
        state["error_context"]["sql_generation_error"] = str(e)
        state["next_step"] = "handle_error"

    return state
