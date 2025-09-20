"""AI agent for intelligent schema analysis and data understanding."""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from domain.entities import ProcessType
from infrastructure import llm
from infrastructure.logging import get_logger
from tracing.langsmith_setup import trace_agent_operation, tracer

logger = get_logger(__name__)


# Simplified Pydantic models matching natural AI responses
class SchemaAnalysisResponse(BaseModel):
    """Simplified schema analysis response from LLM."""

    query_intent: str = Field(
        default="Unknown intent", description="Intent of the user's query"
    )
    relevant_tables: List[str] = Field(
        default_factory=list, description="List of table names"
    )
    target_metrics: List[str] = Field(
        default_factory=list, description="List of metric column names"
    )
    grouping_dimensions: List[str] = Field(
        default_factory=list, description="List of dimension column names"
    )
    filter_columns: List[str] = Field(
        default_factory=list, description="List of filter column names"
    )
    join_strategy: List[str] = Field(
        default_factory=list, description="How to join tables (must be strings)"
    )
    aggregation_needed: bool = Field(
        default=True, description="Whether aggregation is needed"
    )
    complexity_score: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Complexity score from 0.0 to 1.0"
    )


# Legacy dataclasses for backward compatibility
@dataclass
class ColumnAnalysis:
    """Analysis of a specific column."""

    name: str
    data_type: str
    purpose: str  # What this column represents semantically
    is_metric: bool = False  # Whether this is a measurable metric
    is_dimension: bool = False  # Whether this is a grouping dimension
    is_identifier: bool = False  # Whether this is an ID/key column
    aggregation_type: Optional[str] = None  # SUM, COUNT, AVG, etc.


@dataclass
class TableAnalysis:
    """Analysis of a table's structure and purpose."""

    name: str
    purpose: str  # What this table represents
    primary_metrics: List[ColumnAnalysis]  # Key measurement columns
    dimensions: List[ColumnAnalysis]  # Grouping/categorization columns
    identifiers: List[ColumnAnalysis]  # ID/key columns
    relationships: List[str]  # How this table relates to others


@dataclass
class DataUnderstanding:
    """Comprehensive understanding of data schema for a specific query."""

    query_intent: str
    relevant_tables: List[TableAnalysis]
    target_metrics: List[ColumnAnalysis]  # The main metrics to analyze
    grouping_dimensions: List[ColumnAnalysis]  # How to group/segment data
    filter_columns: List[ColumnAnalysis]  # Columns for filtering
    join_strategy: List[str]  # How to join tables
    aggregation_needed: bool
    complexity_score: float  # 0.0 to 1.0


class SchemaIntelligenceAgent:
    """AI agent that provides intelligent schema analysis and data understanding."""

    def __init__(self):
        """Initialize the schema intelligence agent."""
        logger.info("SchemaIntelligenceAgent initialized")

    def understand_data(
        self,
        query_or_messages: Union[str, List[Dict[str, str]]],
        schema_info: Dict[str, Any],
        process_type: ProcessType,
    ) -> DataUnderstanding:
        """
        Analyze the data schema in context of the user's query.

        Args:
            query_or_messages: User query or list of chat messages providing context
            schema_info: Available table schemas
            process_type: The determined process type (SQL, PYTHON, VISUALIZATION)

        Returns:
            DataUnderstanding with semantic analysis of relevant data
        """
        if isinstance(query_or_messages, list):
            query = "\n".join(
                f"{m.get('role', 'user').capitalize()}: {m.get('content', '')}"
                for m in query_or_messages
            )
        else:
            query = query_or_messages

        with trace_agent_operation(
            name="analyze_data_schema",
            query=query,
            process_type=process_type.value,
            schema_tables=len(schema_info),
        ):
            logger.info(f"Analyzing data schema for: {query[:100]}...")

            try:
                # Create schema analysis prompt
                prompt = self._create_schema_analysis_prompt(
                    query, schema_info, process_type
                )

                # Get AI analysis using structured output
                structured_response = llm.llm_client.generate_structured(
                    prompt, SchemaAnalysisResponse, temperature=0.1
                )

                # Convert structured response to DataUnderstanding
                understanding = self._convert_structured_response(structured_response)

                # Log metrics
                tracer.log_metrics(
                    {
                        "relevant_tables_found": len(understanding.relevant_tables),
                        "target_metrics_identified": len(understanding.target_metrics),
                        "grouping_dimensions_found": len(
                            understanding.grouping_dimensions
                        ),
                        "complexity_score": understanding.complexity_score,
                    }
                )

                logger.info(
                    f"Schema analysis complete: {len(understanding.relevant_tables)} tables, {len(understanding.target_metrics)} metrics identified"
                )
                return understanding

            except Exception as e:
                logger.error(f"Schema analysis failed: {str(e)}")
                return self._create_fallback_understanding(query, schema_info)

    def _convert_structured_response(
        self, response: SchemaAnalysisResponse
    ) -> DataUnderstanding:
        """Convert simplified structured response to legacy DataUnderstanding format."""

        # Convert table names to TableAnalysis objects
        relevant_tables = []
        for table_name in response.relevant_tables:
            relevant_tables.append(
                TableAnalysis(
                    name=table_name,
                    purpose=f"Table for {response.query_intent}",
                    primary_metrics=[],
                    dimensions=[],
                    identifiers=[],
                    relationships=[],
                )
            )

        # Convert metric names to ColumnAnalysis objects
        target_metrics = []
        for metric_name in response.target_metrics:
            target_metrics.append(
                ColumnAnalysis(
                    name=metric_name,
                    data_type="unknown",
                    purpose=f"Metric for {response.query_intent}",
                    is_metric=True,
                    aggregation_type=None,
                )
            )

        # Convert dimension names to ColumnAnalysis objects
        grouping_dimensions = []
        for dim_name in response.grouping_dimensions:
            grouping_dimensions.append(
                ColumnAnalysis(
                    name=dim_name,
                    data_type="unknown",
                    purpose=f"Grouping dimension for {response.query_intent}",
                    is_dimension=True,
                )
            )

        # Convert filter column names to ColumnAnalysis objects
        filter_columns = []
        for filter_name in response.filter_columns:
            filter_columns.append(
                ColumnAnalysis(
                    name=filter_name,
                    data_type="unknown",
                    purpose=f"Filter column for {response.query_intent}",
                    is_dimension=True,
                )
            )

        return DataUnderstanding(
            query_intent=response.query_intent,
            relevant_tables=relevant_tables,
            target_metrics=target_metrics,
            grouping_dimensions=grouping_dimensions,
            filter_columns=filter_columns,
            join_strategy=response.join_strategy,
            aggregation_needed=response.aggregation_needed,
            complexity_score=response.complexity_score,
        )

    def _create_schema_analysis_prompt(
        self, query: str, schema_info: Dict[str, Any], process_type: ProcessType
    ) -> str:
        """Create the schema analysis prompt for structured output."""
        schema_details = self._format_schema_for_prompt(schema_info)

        return f"""
Analyze the database schema and user query to provide structured data understanding.

USER QUERY: "{query}"
PROCESS TYPE: {process_type.value}

AVAILABLE DATABASE SCHEMA:
{schema_details}

ANALYSIS INSTRUCTIONS:
1. Determine the user's intent and what data analysis is needed
2. Identify relevant tables required for this analysis
3. Identify target metrics (measurable values like revenue, count, etc.)
4. Identify grouping dimensions (how to segment/categorize the data)
5. Identify filter columns (what might be filtered by time, status, etc.)
6. Determine join strategy (how tables should be connected)
7. Assess if aggregation is needed
8. Estimate complexity (0.0 = simple select, 1.0 = complex multi-table analysis)

IMPORTANT REQUIREMENTS:
- For "join_strategy", provide simple JOIN statements as strings like "JOIN table1 ON condition"
- Focus on the business context of the query, not just technical column matching
- Consider user intent: analysis vs reporting vs exploration
- Make realistic complexity assessments based on query scope

EXAMPLES:
For "churn rate by month":
- query_intent: "Calculate customer churn metrics over time"
- relevant_tables: ["users", "orders"]
- target_metrics: ["churn_rate", "active_customers", "churned_customers"]
- grouping_dimensions: ["month", "year"]
- join_strategy: ["LEFT JOIN orders ON users.id = orders.user_id"]
- aggregation_needed: true
- complexity_score: 0.7

For "top products by revenue":
- query_intent: "Rank products by sales performance"
- relevant_tables: ["products", "order_items"]
- target_metrics: ["total_revenue", "units_sold"]
- grouping_dimensions: ["product_name", "category"]
- join_strategy: ["JOIN order_items ON products.id = order_items.product_id"]
- aggregation_needed: true
- complexity_score: 0.4

IMPORTANT: All list fields should contain simple strings, not complex objects.
"""

    def _format_schema_for_prompt(self, schema_info: Dict[str, Any]) -> str:
        """Format schema information for the AI prompt."""
        if not schema_info:
            return "No schema information available"

        formatted_parts = []
        for table_name, schema in schema_info.items():
            columns_info = []
            for col in schema["columns"][:15]:  # Limit to avoid token limits
                if isinstance(col, dict):
                    col_name = col.get("name", "unknown")
                    col_type = col.get("type", "unknown")
                    columns_info.append(f"  - {col_name} ({col_type})")
                else:
                    columns_info.append(f"  - {str(col)}")

            formatted_parts.append(f"Table: {table_name}")
            formatted_parts.extend(columns_info)
            formatted_parts.append("")  # Empty line between tables

        return "\\n".join(formatted_parts)

    def _create_fallback_understanding(
        self, query: str, schema_info: Dict[str, Any]
    ) -> DataUnderstanding:
        """Create a basic fallback understanding when AI analysis fails."""
        logger.info("Creating fallback data understanding")

        # Basic analysis of available tables
        relevant_tables = []
        for table_name in schema_info.keys():
            relevant_tables.append(
                TableAnalysis(
                    name=table_name,
                    purpose=f"Data table: {table_name}",
                    primary_metrics=[],
                    dimensions=[],
                    identifiers=[],
                    relationships=[],
                )
            )

        return DataUnderstanding(
            query_intent=query,
            relevant_tables=relevant_tables,
            target_metrics=[],
            grouping_dimensions=[],
            filter_columns=[],
            join_strategy=[],
            aggregation_needed=True,
            complexity_score=0.8,  # High complexity since we couldn't analyze it
        )


# Global schema agent instance
schema_agent = SchemaIntelligenceAgent()
