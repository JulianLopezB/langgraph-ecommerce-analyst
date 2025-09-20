"""Utilities for SQL optimization and validation."""

from typing import TYPE_CHECKING

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from agents.schema_agent import DataUnderstanding
from infrastructure.logging import get_logger
from utils.sql_utils import clean_sql_query, ensure_valid_group_by

logger = get_logger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from agents.sql_agent import SQLGenerationResult


def init_sql_validator(api_key: str | None, dataset_id: str):
    """Initialize LangChain SQL validation chain."""
    try:
        langchain_llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.1,
        )

        validation_system_prompt = f"""Double check the BigQuery SQL query for common mistakes, including:
            - Using NOT IN with NULL values
            - Using UNION when UNION ALL should have been used
            - Using BETWEEN for exclusive ranges
            - Data type mismatch in predicates
            - Properly quoting identifiers with backticks for BigQuery
            - Using the correct number of arguments for functions
            - Casting to the correct data type
            - Using the proper columns for joins
            - GROUP BY aggregation rules: ALL non-aggregate columns in SELECT must be in GROUP BY
            - Do NOT include aggregate functions (SUM, COUNT, AVG) in GROUP BY clause
            - Table alias consistency between SELECT and GROUP BY
            - BigQuery date functions: DATE(timestamp_column), DATE_DIFF syntax
            - Complete table paths: `{dataset_id}.TABLE_NAME`

            CRITICAL: If selecting individual columns with aggregates, ensure every non-aggregate column appears in GROUP BY.

            Examples of CORRECT GROUP BY:
            ✅ SELECT DATE(o.created_at), SUM(oi.sale_price) GROUP BY DATE(o.created_at)
            ✅ SELECT o.user_id, COUNT(*) GROUP BY o.user_id

            Examples of WRONG GROUP BY:
            ❌ SELECT o.user_id, o.order_id, SUM(oi.sale_price) GROUP BY o.user_id  (missing o.order_id in GROUP BY)
            ❌ SELECT DATE(o.created_at), SUM(oi.sale_price) GROUP BY SUM(oi.sale_price)  (aggregate in GROUP BY)

            If there are any mistakes, rewrite the query correctly.
            If there are no mistakes, reproduce the original query exactly.
            Output ONLY the final SQL query, no explanations."""

        validation_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", validation_system_prompt),
                ("human", "Validate this BigQuery SQL query:\n\n{query}"),
            ]
        )

        sql_validation_chain = validation_prompt | langchain_llm | StrOutputParser()

        logger.info("LangChain SQL validator initialized successfully")
        return sql_validation_chain

    except Exception as e:  # pragma: no cover - initialization is best effort
        logger.warning(f"Failed to initialize LangChain SQL validator: {e}")
        return None


def validate_sql_with_langchain(
    sql_validation_chain, sql_query: str, dataset_id: str, max_results: int
) -> str:
    """Validate SQL query using LangChain validation chain."""
    if sql_validation_chain is None:
        logger.warning("LangChain SQL validator not available, using basic cleaning")
        return clean_sql_query(
            sql_query, dataset_id, max_results, add_dataset_prefix=False, add_limit=True
        )

    try:
        validated_sql = sql_validation_chain.invoke({"query": sql_query})
        validated_sql = clean_sql_query(
            validated_sql,
            dataset_id,
            max_results,
            add_dataset_prefix=False,
            add_limit=True,
        )
        validated_sql = ensure_valid_group_by(validated_sql)
        logger.info("SQL query validated successfully with LangChain")
        return validated_sql
    except Exception as e:  # pragma: no cover - best effort
        logger.warning(
            f"LangChain SQL validation failed: {e}, falling back to basic cleaning"
        )
        return ensure_valid_group_by(
            clean_sql_query(
                sql_query, dataset_id, max_results, add_dataset_prefix=False, add_limit=True
            )
        )


def optimize_and_validate(
    sql_result, data_understanding: DataUnderstanding, dataset_id: str, max_results: int
):
    """Apply final optimizations and validations to the SQL."""
    from agents.sql_agent import SQLGenerationResult

    sql_query = sql_result.sql_query
    optimizations = list(sql_result.optimization_applied)

    if dataset_id not in sql_query:
        for table_name in ["orders", "order_items", "products", "users"]:
            sql_query = sql_query.replace(
                f" {table_name} ", f" `{dataset_id}.{table_name}` "
            )
            sql_query = sql_query.replace(
                f"FROM {table_name}", f"FROM `{dataset_id}.{table_name}`"
            )
            sql_query = sql_query.replace(
                f"JOIN {table_name}", f"JOIN `{dataset_id}.{table_name}`"
            )
        optimizations.append("dataset_prefix_added")

    if "LIMIT" not in sql_query.upper():
        sql_query += f" LIMIT {max_results}"
        optimizations.append("limit_added")

    sql_query = sql_query.strip()
    if sql_query.endswith(";"):
        sql_query = sql_query[:-1]

    return SQLGenerationResult(
        sql_query=sql_query,
        explanation=sql_result.explanation,
        estimated_complexity=sql_result.estimated_complexity,
        optimization_applied=optimizations,
        tables_used=sql_result.tables_used,
        metrics_computed=sql_result.metrics_computed,
        confidence=sql_result.confidence,
    )


def create_fallback_sql(
    query: str, data_understanding: DataUnderstanding, dataset_id: str
):
    """Create a basic fallback SQL when generation fails."""
    from agents.sql_agent import SQLGenerationResult

    logger.info("Creating fallback SQL query")

    primary_table = "orders"
    if data_understanding.relevant_tables:
        primary_table = data_understanding.relevant_tables[0].name

    fallback_sql = f"""
        SELECT *
        FROM `{dataset_id}.{primary_table}`
        LIMIT 100
        """

    return SQLGenerationResult(
        sql_query=fallback_sql.strip(),
        explanation=f"Fallback query for table: {primary_table}",
        estimated_complexity="low",
        optimization_applied=["fallback_generation"],
        tables_used=[primary_table],
        metrics_computed=[],
        confidence=0.3,
    )
