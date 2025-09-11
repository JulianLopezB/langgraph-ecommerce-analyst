"""AI agent for intelligent schema analysis and data understanding."""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import json

from workflow.state import ProcessType
from services.llm_service import GeminiService
from logging_config import get_logger
from tracing.langsmith_setup import LangSmithTracer, trace_agent_operation

logger = get_logger(__name__)


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
    """AI agent that understands data schemas semantically, not just syntactically."""

    def __init__(self, llm_service: GeminiService, tracer: LangSmithTracer):
        """Initialize the schema intelligence agent with dependencies."""
        self.llm_service = llm_service
        self.tracer = tracer
        logger.info("SchemaIntelligenceAgent initialized")
    
    def understand_data(self, query: str, schema_info: Dict[str, Any], 
                       process_type: ProcessType) -> DataUnderstanding:
        """
        Analyze the data schema in context of the user's query.
        
        Args:
            query: User's natural language query
            schema_info: Available table schemas
            process_type: The determined process type (SQL, PYTHON, VISUALIZATION)
            
        Returns:
            DataUnderstanding with semantic analysis of relevant data
        """
        with trace_agent_operation(
            self.tracer,
            name="analyze_data_schema",
            query=query,
            process_type=process_type.value,
            schema_tables=len(schema_info)
        ):
            logger.info(f"Analyzing data schema for: {query[:100]}...")
            
            try:
                # Create schema analysis prompt
                prompt = self._create_schema_analysis_prompt(query, schema_info, process_type)
                
                # Get AI analysis
                response = self.llm_service.generate_text(prompt, temperature=0.1)
                
                # Parse the response
                understanding = self._parse_schema_analysis(response.content, query)
                
                # Log metrics
                self.tracer.log_metrics({
                    "relevant_tables_found": len(understanding.relevant_tables),
                    "target_metrics_identified": len(understanding.target_metrics),
                    "grouping_dimensions_found": len(understanding.grouping_dimensions),
                    "complexity_score": understanding.complexity_score,
                    "aggregation_needed": understanding.aggregation_needed
                })
                
                logger.info(f"Schema analysis complete: {len(understanding.relevant_tables)} tables, "
                          f"{len(understanding.target_metrics)} metrics identified")
                
                return understanding
                
            except Exception as e:
                logger.error(f"Error in schema analysis: {e}")
                # Return basic fallback understanding
                return self._create_fallback_understanding(query, schema_info)
    
    def _create_schema_analysis_prompt(self, query: str, schema_info: Dict[str, Any], 
                                     process_type: ProcessType) -> str:
        """Create the AI prompt for schema analysis."""
        
        # Format schema information for the prompt
        schema_details = self._format_schema_for_prompt(schema_info)
        
        return f"""
You are an expert data analyst. Analyze this query and data schema to understand what data is needed.

Query: "{query}"
Process Type: {process_type.value}

Available Tables and Schemas:
{schema_details}

SEMANTIC ANALYSIS TASK:
Think beyond column names - understand the semantic meaning and relationships.

For example:
- "revenue" could be: revenue, sales, amount, total_price, gross, net_sales, income
- "products" could be: product_name, item_title, sku, product_id, item_description
- "customers" could be: user_id, customer_name, client_id, buyer_name, account_id
- "time" could be: created_at, order_date, timestamp, date_purchased, year, month

ANALYSIS REQUIREMENTS:

1. IDENTIFY RELEVANT TABLES:
   Which tables contain data needed to answer this query?

2. FIND TARGET METRICS:
   What are the main measurements/values the user wants to analyze?
   (revenue, counts, averages, totals, etc.)

3. DETERMINE GROUPING DIMENSIONS:
   How should the data be grouped or segmented?
   (by product, by customer, by time period, by category, etc.)

4. PLAN DATA RELATIONSHIPS:
   How do the tables connect? What joins are needed?

5. ASSESS COMPLEXITY:
   How complex is this analysis? (0.1 = simple aggregation, 1.0 = complex multi-table analysis)

Respond in JSON format:
{{
    "query_intent": "Brief description of what user wants to accomplish",
    "relevant_tables": [
        {{
            "name": "table_name",
            "purpose": "What this table represents",
            "relevance_score": 0.9,
            "key_columns": ["col1", "col2"]
        }}
    ],
    "target_metrics": [
        {{
            "column_name": "revenue",
            "table": "orders", 
            "purpose": "Total sales amount",
            "aggregation": "SUM"
        }}
    ],
    "grouping_dimensions": [
        {{
            "column_name": "product_name",
            "table": "products",
            "purpose": "Group by individual products"
        }}
    ],
    "filter_columns": [
        {{
            "column_name": "order_date",
            "table": "orders",
            "purpose": "Filter by time period"
        }}
    ],
    "join_strategy": [
        "JOIN products ON orders.product_id = products.id"
    ],
    "aggregation_needed": true,
    "complexity_score": 0.7
}}

Think semantically about data relationships and user intent, not just column name matching.
"""
    
    def _format_schema_for_prompt(self, schema_info: Dict[str, Any]) -> str:
        """Format schema information for the AI prompt."""
        if not schema_info:
            return "No schema information available"
        
        formatted_parts = []
        for table_name, schema in schema_info.items():
            columns_info = []
            for col in schema['columns'][:15]:  # Limit to avoid token limits
                if isinstance(col, dict):
                    col_name = col.get('name', 'unknown')
                    col_type = col.get('type', 'unknown')
                    columns_info.append(f"  - {col_name} ({col_type})")
                else:
                    columns_info.append(f"  - {str(col)}")

            formatted_parts.append(f"Table: {table_name}")
            formatted_parts.extend(columns_info)
            formatted_parts.append("")  # Empty line between tables

        return "\\n".join(formatted_parts)
    
    def _parse_schema_analysis(self, response_content: str, original_query: str) -> DataUnderstanding:
        """Parse the AI response into a DataUnderstanding object."""
        try:
            # Clean and parse JSON response
            cleaned_response = response_content.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            
            analysis_data = json.loads(cleaned_response)
            
            # Parse relevant tables
            relevant_tables = []
            for table_data in analysis_data.get("relevant_tables", []):
                relevant_tables.append(TableAnalysis(
                    name=table_data.get("name", ""),
                    purpose=table_data.get("purpose", ""),
                    primary_metrics=[],  # Will be populated from target_metrics
                    dimensions=[],  # Will be populated from grouping_dimensions
                    identifiers=[],
                    relationships=[]
                ))
            
            # Parse target metrics
            target_metrics = []
            for metric_data in analysis_data.get("target_metrics", []):
                target_metrics.append(ColumnAnalysis(
                    name=metric_data.get("column_name", ""),
                    data_type="numeric",
                    purpose=metric_data.get("purpose", ""),
                    is_metric=True,
                    aggregation_type=metric_data.get("aggregation", "SUM")
                ))
            
            # Parse grouping dimensions
            grouping_dimensions = []
            for dim_data in analysis_data.get("grouping_dimensions", []):
                grouping_dimensions.append(ColumnAnalysis(
                    name=dim_data.get("column_name", ""),
                    data_type="string",
                    purpose=dim_data.get("purpose", ""),
                    is_dimension=True
                ))
            
            # Parse filter columns
            filter_columns = []
            for filter_data in analysis_data.get("filter_columns", []):
                filter_columns.append(ColumnAnalysis(
                    name=filter_data.get("column_name", ""),
                    data_type="unknown",
                    purpose=filter_data.get("purpose", ""),
                    is_dimension=True
                ))
            
            return DataUnderstanding(
                query_intent=analysis_data.get("query_intent", original_query),
                relevant_tables=relevant_tables,
                target_metrics=target_metrics,
                grouping_dimensions=grouping_dimensions,
                filter_columns=filter_columns,
                join_strategy=analysis_data.get("join_strategy", []),
                aggregation_needed=bool(analysis_data.get("aggregation_needed", True)),
                complexity_score=float(analysis_data.get("complexity_score", 0.5))
            )
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse schema analysis: {e}")
            logger.debug(f"Raw response: {response_content}")
            return self._create_fallback_understanding(original_query, {})
    
    def _create_fallback_understanding(self, query: str, schema_info: Dict[str, Any]) -> DataUnderstanding:
        """Create a basic fallback understanding when AI analysis fails."""
        logger.info("Creating fallback data understanding")
        
        # Basic analysis of available tables
        relevant_tables = []
        for table_name in schema_info.keys():
            relevant_tables.append(TableAnalysis(
                name=table_name,
                purpose=f"Data table: {table_name}",
                primary_metrics=[],
                dimensions=[],
                identifiers=[],
                relationships=[]
            ))
        
        return DataUnderstanding(
            query_intent=query,
            relevant_tables=relevant_tables,
            target_metrics=[],
            grouping_dimensions=[],
            filter_columns=[],
            join_strategy=[],
            aggregation_needed=True,
            complexity_score=0.8  # High complexity since we couldn't analyze it
        )
