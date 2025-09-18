"""AI-driven process type classification agent."""

import json
from dataclasses import dataclass, field
from typing import Dict, Any, List, Union

from domain.entities import ProcessType
from infrastructure.logging import get_logger
from infrastructure.llm import llm_client
from tracing.langsmith_setup import tracer, trace_agent_operation

logger = get_logger(__name__)


@dataclass
class ProcessTypeResult:
    """Result from process type classification."""

    process_type: ProcessType
    confidence: float
    reasoning: str
    requires_aggregation: bool = False
    complexity_level: str = "medium"  # low, medium, high
    suggested_tables: List[str] = field(default_factory=list)


class ProcessTypeClassifier:
    """AI agent that determines the optimal process type for a query."""

    def __init__(self):
        """Initialize the process classifier."""
        self.llm_service = llm_client
        logger.info("ProcessTypeClassifier initialized")

    def classify(
        self,
        query_or_messages: Union[str, List[Dict[str, str]]],
        schema_info: Dict[str, Any],
    ) -> ProcessTypeResult:
        """
        Classify the optimal process type using AI analysis.

        Args:
            query_or_messages: User query or list of chat messages providing context
            schema_info: Available table schemas and metadata

        Returns:
            ProcessTypeResult with classification and metadata
        """
        if isinstance(query_or_messages, list):
            query = "\n".join(
                f"{m.get('role', 'user').capitalize()}: {m.get('content', '')}"
                for m in query_or_messages
            )
        else:
            query = query_or_messages

        with trace_agent_operation(
            name="classify_process_type", query=query, schema_tables=len(schema_info)
        ):
            logger.info(f"Classifying process type for query: {query[:100]}...")

            try:
                # Prepare schema summary for LLM
                schema_summary = self._prepare_schema_summary(schema_info)

                # Create classification prompt
                prompt = self._create_classification_prompt(query, schema_summary)

                # Get LLM response
                response = self.llm_service.generate_text(prompt, temperature=0.1)

                # Parse the response
                result = self._parse_classification_response(response.content, query)

                # Log metrics
                tracer.log_metrics(
                    {
                        "process_type_classified": result.process_type.value,
                        "classification_confidence": result.confidence,
                        "complexity_level": result.complexity_level,
                        "query_length": len(query),
                        "tables_suggested": len(result.suggested_tables),
                    }
                )

                logger.info(
                    f"Classified as {result.process_type.value} with confidence {result.confidence:.2f}"
                )
                return result

            except Exception as e:
                logger.error(f"Error in process classification: {e}")
                # Fallback to SQL with low confidence
                return ProcessTypeResult(
                    process_type=ProcessType.SQL,
                    confidence=0.3,
                    reasoning=f"Classification failed, defaulting to SQL: {str(e)}",
                    complexity_level="high",
                )

    def _prepare_schema_summary(self, schema_info: Dict[str, Any]) -> str:
        """Prepare a concise schema summary for the LLM."""
        if not schema_info:
            return "No schema information available"

        summary_parts = []
        for table_name, schema in schema_info.items():
            columns = [
                col.get("name", str(col)) for col in schema["columns"][:10]
            ]  # Limit columns
            summary_parts.append(f"- {table_name}: {', '.join(columns)}")

        return "\\n".join(summary_parts)

    def _create_classification_prompt(self, query: str, schema_summary: str) -> str:
        """Create the AI prompt for process type classification."""
        return f"""
You are an expert data analysis strategist. Analyze this query and determine the optimal process type.

Query: "{query}"

Available data tables:
{schema_summary}

PROCESS TYPE GUIDELINES:

🔹 SQL Process Type:
Use for queries that can be solved with database operations:
- Data retrieval, filtering, aggregation (SUM, COUNT, AVG, etc.)
- Joins across tables, grouping, sorting
- Business metrics (revenue, sales, customers, products)
- Trend analysis with date functions
- Statistical aggregations and window functions
- Most KPIs and business intelligence queries
Examples: "top products by revenue", "customer count by region", "monthly sales trends"

🔹 PYTHON Process Type:
Use for queries requiring advanced computation:
- Machine learning models (clustering, classification, regression)
- Advanced statistical analysis (correlation matrices, hypothesis testing)
- Forecasting and predictive analytics
- Complex data transformations beyond SQL capabilities
- Custom algorithms or mathematical models
Examples: "predict customer churn", "cluster customers by behavior", "correlation analysis"

🔹 VISUALIZATION Process Type:
Use when explicitly requesting visual output:
- Charts, graphs, plots, dashboards
- Visual representations of data
- "Show me a chart", "create a graph", "visualize the data"

ANALYSIS REQUIREMENTS:
1. Can this be answered with SQL aggregations/joins? → SQL
2. Does it need ML/advanced stats/predictions? → PYTHON
3. Does it explicitly request charts/graphs? → VISUALIZATION

Respond in JSON format:
{{
    "process_type": "sql|python|visualization",
    "confidence": 0.95,
    "reasoning": "Clear explanation of why this process type was chosen",
    "requires_aggregation": true,
    "complexity_level": "low|medium|high",
    "suggested_tables": ["table1", "table2"]
}}

Think step by step:
1. What is the user trying to accomplish?
2. What type of analysis is needed?
3. Can SQL handle this alone, or does it need Python/visualization?
"""

    def _parse_classification_response(
        self, response_content: str, original_query: str
    ) -> ProcessTypeResult:
        """Parse the LLM response into a ProcessTypeResult."""
        try:
            # Clean the response and extract JSON
            cleaned_response = response_content.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]

            # Parse JSON response
            result_data = json.loads(cleaned_response)

            # Convert process type string to enum
            process_type_str = result_data.get("process_type", "sql").lower()
            if process_type_str == "sql":
                process_type = ProcessType.SQL
            elif process_type_str == "python":
                process_type = ProcessType.PYTHON
            elif process_type_str in ["visualization", "viz"]:
                process_type = ProcessType.VISUALIZATION
            else:
                logger.warning(
                    f"Unknown process type: {process_type_str}, defaulting to SQL"
                )
                process_type = ProcessType.SQL

            return ProcessTypeResult(
                process_type=process_type,
                confidence=float(result_data.get("confidence", 0.5)),
                reasoning=result_data.get("reasoning", "AI classification completed"),
                requires_aggregation=bool(
                    result_data.get("requires_aggregation", False)
                ),
                complexity_level=result_data.get("complexity_level", "medium"),
                suggested_tables=result_data.get("suggested_tables", []),
            )

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse classification response: {e}")
            logger.debug(f"Raw response: {response_content}")

            # Fallback analysis based on keywords (temporary)
            if any(
                word in original_query.lower()
                for word in ["predict", "forecast", "model", "cluster", "correlation"]
            ):
                return ProcessTypeResult(
                    process_type=ProcessType.PYTHON,
                    confidence=0.6,
                    reasoning="Fallback: Query contains ML/statistical keywords",
                    complexity_level="high",
                )
            elif any(
                word in original_query.lower()
                for word in ["chart", "graph", "plot", "visualize", "show"]
            ):
                return ProcessTypeResult(
                    process_type=ProcessType.VISUALIZATION,
                    confidence=0.6,
                    reasoning="Fallback: Query requests visualization",
                    complexity_level="medium",
                )
            else:
                return ProcessTypeResult(
                    process_type=ProcessType.SQL,
                    confidence=0.6,
                    reasoning="Fallback: Default to SQL for data analysis",
                    complexity_level="medium",
                )


# Global classifier instance
process_classifier = ProcessTypeClassifier()
