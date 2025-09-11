"""Utilities for parsing SQL generation responses."""

import json
from typing import TYPE_CHECKING

from agents.schema_agent import DataUnderstanding
from config import config
from logging_config import get_logger

DATASET_ID = config.api_configurations.dataset_id

logger = get_logger(__name__)


if TYPE_CHECKING:  # pragma: no cover
    from agents.sql_agent import SQLGenerationResult


def parse_sql_response(
    response_content: str, data_understanding: DataUnderstanding
):
    """Parse the AI response into an SQLGenerationResult."""
    from agents.sql_agent import SQLGenerationResult

    try:
        cleaned_response = response_content.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]

        result_data = json.loads(cleaned_response)

        return SQLGenerationResult(
            sql_query=result_data.get("sql_query", ""),
            explanation=result_data.get("explanation", "SQL query generated"),
            estimated_complexity=result_data.get("estimated_complexity", "medium"),
            optimization_applied=result_data.get("optimization_applied", []),
            tables_used=result_data.get("tables_used", []),
            metrics_computed=result_data.get("metrics_computed", []),
            confidence=float(result_data.get("confidence", 0.7)),
        )

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.warning(f"Failed to parse SQL response: {e}")
        logger.debug(f"Raw response: {response_content}")

        sql_query = _extract_sql_from_text(response_content)

        return SQLGenerationResult(
            sql_query=sql_query,
            explanation="SQL extracted from response",
            estimated_complexity="medium",
            optimization_applied=["text_extraction"],
            tables_used=[table.name for table in data_understanding.relevant_tables],
            metrics_computed=[metric.name for metric in data_understanding.target_metrics],
            confidence=0.5,
        )


def _extract_sql_from_text(text: str) -> str:
    """Extract SQL query from plain text response."""
    lines = text.split("\n")
    sql_lines = []
    in_sql = False

    for line in lines:
        line = line.strip()
        if line.upper().startswith(("SELECT", "WITH")):
            in_sql = True
            sql_lines.append(line)
        elif in_sql:
            if line and not line.startswith("--") and not line.startswith("//"):
                sql_lines.append(line)
            elif line == "" and sql_lines:
                break

    if sql_lines:
        return " ".join(sql_lines)

    return f"SELECT * FROM `{DATASET_ID}.orders` LIMIT 100"

