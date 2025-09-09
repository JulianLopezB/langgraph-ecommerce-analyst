"""AI agent for intelligent SQL query generation."""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import json

from agents.schema_agent import DataUnderstanding, ColumnAnalysis
from agents.process_classifier import ProcessTypeResult
from services.llm_service import GeminiService
from logging_config import get_logger
from tracing.langsmith_setup import tracer, trace_agent_operation

# LangChain SQL validation imports
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

logger = get_logger(__name__)


@dataclass
class SQLGenerationResult:
    """Result from SQL generation."""
    sql_query: str
    explanation: str
    estimated_complexity: str  # low, medium, high
    optimization_applied: List[str]  # List of optimizations applied
    tables_used: List[str]
    metrics_computed: List[str]
    confidence: float


class SQLGenerationAgent:
    """AI agent that generates intelligent SQL queries based on data understanding."""
    
    def __init__(self):
        """Initialize the SQL generation agent."""
        self.llm_service = GeminiService()
        
        # Initialize LangChain SQL validator
        self._init_sql_validator()
        
        logger.info("SQLGenerationAgent initialized")
    
    def generate_sql(self, query: str, data_understanding: DataUnderstanding, 
                    process_result: ProcessTypeResult) -> SQLGenerationResult:
        """
        Generate an optimized SQL query based on AI understanding of data and intent.
        
        Args:
            query: Original user query
            data_understanding: AI analysis of relevant data schema
            process_result: Process type classification result
            
        Returns:
            SQLGenerationResult with generated query and metadata
        """
        with trace_agent_operation(
            name="generate_intelligent_sql",
            query=query,
            target_tables=len(data_understanding.relevant_tables),
            target_metrics=len(data_understanding.target_metrics)
        ):
            logger.info(f"Generating SQL for: {query[:100]}...")
            
            try:
                # Create SQL generation prompt
                prompt = self._create_sql_generation_prompt(query, data_understanding, process_result)
                
                # Generate SQL using AI
                response = self.llm_service.generate_text(prompt, temperature=0.1)  # Low temp for precision
                
                # Parse the response
                sql_result = self._parse_sql_response(response.content, data_understanding)
                
                # Apply final optimizations and validation
                sql_result = self._optimize_and_validate(sql_result, data_understanding)
                
                # Use LangChain SQL validation for robust query checking
                sql_result.sql_query = self._validate_sql_with_langchain(sql_result.sql_query)
                
                # Log metrics
                tracer.log_metrics({
                    "sql_generation_success": True,
                    "query_length": len(sql_result.sql_query),
                    "tables_used": len(sql_result.tables_used),
                    "metrics_computed": len(sql_result.metrics_computed),
                    "complexity": sql_result.estimated_complexity,
                    "confidence": sql_result.confidence
                })
                
                logger.info(f"SQL generated successfully: {sql_result.estimated_complexity} complexity, "
                          f"{len(sql_result.tables_used)} tables")
                
                return sql_result
                
            except Exception as e:
                logger.error(f"Error generating SQL: {e}")
                # Return fallback SQL
                return self._create_fallback_sql(query, data_understanding)
    
    def _create_sql_generation_prompt(self, query: str, data_understanding: DataUnderstanding,
                                    process_result: ProcessTypeResult) -> str:
        """Create the AI prompt for SQL generation."""
        
        # Format data understanding for prompt
        tables_info = self._format_tables_for_prompt(data_understanding.relevant_tables)
        metrics_info = self._format_metrics_for_prompt(data_understanding.target_metrics)
        dimensions_info = self._format_dimensions_for_prompt(data_understanding.grouping_dimensions)
        schema_details = self._format_detailed_schema(data_understanding)
        
        return f"""
You are an expert SQL developer specializing in BigQuery. Generate an optimized SQL query.

QUERY REQUEST: "{query}"

DATA ANALYSIS CONTEXT:
Intent: {data_understanding.query_intent}
Aggregation Needed: {data_understanding.aggregation_needed}
Complexity Score: {data_understanding.complexity_score}

AVAILABLE TABLES AND COLUMNS:
{schema_details}

TARGET METRICS TO COMPUTE:
{metrics_info}

GROUPING DIMENSIONS:
{dimensions_info}

JOIN STRATEGY:
{'; '.join(data_understanding.join_strategy)}

SQL GENERATION REQUIREMENTS:

1. **CRITICAL - Table Usage**: 
   - ONLY use the 4 existing tables shown in the schema above
   - Do NOT create new tables, CTEs with made-up names, or reference non-existent tables
   - Every table reference must include full path: `bigquery-public-data.thelook_ecommerce.TABLE_NAME`

2. **Performance**: Optimize for cost and speed
   - Use appropriate WHERE clauses for filtering
   - Limit results to reasonable size (≤10,000 rows)
   - Use efficient JOIN strategies
   - Apply aggregations correctly

3. **Structure**: Generate clean, readable SQL
   - Meaningful column aliases
   - Proper indentation and formatting
   - Comments for complex logic

4. **Analytics Focus**: 
   - Focus on the metrics the user actually wants
   - Include appropriate aggregations (SUM, COUNT, AVG, etc.)
   - Handle NULL values appropriately
   - Order results meaningfully (usually by primary metric DESC)

5. **BigQuery Optimizations**:
   - Use proper date/time filtering
   - Leverage BigQuery functions where beneficial
   - Consider partitioning in WHERE clauses

6. **CRITICAL - Date Functions**:
   - Use DATE_DIFF(CURRENT_DATE(), DATE(timestamp_column), DAY) for days between dates
   - Use TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), timestamp_column, DAY) for timestamp differences
   - Convert TIMESTAMP to DATE using DATE(timestamp_column) when needed
   - For RFM: DATE_DIFF(CURRENT_DATE(), DATE(o.created_at), DAY) AS days_since_last_order

7. **CRITICAL - GROUP BY Rules**:
   - EVERY non-aggregate column in SELECT must be in GROUP BY
   - Do NOT include aggregate functions (SUM, COUNT, AVG) in GROUP BY  
   - For time series: GROUP BY DATE(created_at), not SUM(revenue)
   - If selecting o.order_id, SUM(oi.sale_price) → must GROUP BY o.order_id
   - If selecting DATE(o.created_at), SUM(revenue) → must GROUP BY DATE(o.created_at)
   - If selecting p.name, COUNT(*) → must GROUP BY p.name
   - Use table aliases consistently: GROUP BY o.user_id (not just user_id)
   
   EXAMPLES:
   ✅ CORRECT: SELECT DATE(o.created_at), SUM(oi.sale_price) GROUP BY DATE(o.created_at)
   ✅ CORRECT: SELECT o.user_id, COUNT(o.order_id) GROUP BY o.user_id  
   ❌ WRONG: SELECT o.user_id, o.order_id, SUM(oi.sale_price) GROUP BY o.user_id
   ❌ WRONG: SELECT DATE(o.created_at), SUM(oi.sale_price) GROUP BY SUM(oi.sale_price)

8. **CRITICAL - Column References**:
   - Use EXACT column names from the schema details above
   - Always use proper table aliases (e.g., p.name, oi.sale_price)
   - Double-check that all referenced columns exist in their respective tables
   - For revenue analysis, use order_items.sale_price (NOT product_id from orders)

9. **CRITICAL - SQL Syntax Validation**:
   - Ensure all parentheses are balanced
   - No trailing commas in SELECT clauses
   - Proper WHERE clause syntax
   - Valid date range filtering
   - Complete statements (no fragments)

10. **CRITICAL - FORECASTING QUERIES**:
   - For "forecast", "predict", or "future" requests: SQL should ONLY retrieve HISTORICAL data
   - DO NOT generate future dates or forecasted values in SQL - that's Python's job!
   - Retrieve past 12+ months of historical sales data for time series analysis
   - Use actual dates from database (o.created_at), not artificially generated future dates
   - Python will analyze historical patterns and generate future predictions
   
   Example for forecasting: Get historical monthly sales for Python to analyze
   ✅ CORRECT: SELECT DATE_TRUNC(DATE(o.created_at), MONTH) as month, SUM(oi.sale_price) as revenue
   ❌ WRONG: SELECT '2025-10-01' as future_month, AVG(revenue) as forecasted_revenue

RESPONSE FORMAT:
{{
    "sql_query": "SELECT ... FROM ... WHERE ... ORDER BY ...",
    "explanation": "Brief explanation of what this query does",
    "estimated_complexity": "low|medium|high",
    "optimization_applied": ["optimization1", "optimization2"],
    "tables_used": ["table1", "table2"],
    "metrics_computed": ["metric1", "metric2"],
    "confidence": 0.95
}}

GENERATE THE SQL:
Think step by step:
1. What tables do I need to join?
2. What metrics do I need to calculate?
3. How should I group and aggregate the data?
4. What filters make sense?
5. How should I order the results?

EXAMPLE for "highest revenue products":
```sql
SELECT 
    p.name as product_name,
    p.brand,
    SUM(oi.sale_price) as total_revenue
FROM `bigquery-public-data.thelook_ecommerce.order_items` oi
JOIN `bigquery-public-data.thelook_ecommerce.products` p 
    ON oi.product_id = p.id
GROUP BY p.name, p.brand
ORDER BY total_revenue DESC
LIMIT 10
```

EXAMPLE for "RFM analysis":
```sql
SELECT 
    o.user_id,
    DATE_DIFF(CURRENT_DATE(), DATE(MAX(o.created_at)), DAY) as recency_days,
    COUNT(DISTINCT o.order_id) as frequency_orders,
    SUM(oi.sale_price) as monetary_value
FROM `bigquery-public-data.thelook_ecommerce.orders` o
JOIN `bigquery-public-data.thelook_ecommerce.order_items` oi 
    ON o.order_id = oi.order_id
GROUP BY o.user_id
LIMIT 1000
```

EXAMPLE for "Forecast sales" (HISTORICAL data for Python forecasting):
```sql
SELECT 
    DATE_TRUNC(DATE(o.created_at), MONTH) as month,
    SUM(oi.sale_price) as monthly_revenue,
    COUNT(DISTINCT o.order_id) as monthly_orders,
    COUNT(DISTINCT o.user_id) as unique_customers
FROM `bigquery-public-data.thelook_ecommerce.orders` o
JOIN `bigquery-public-data.thelook_ecommerce.order_items` oi 
    ON o.order_id = oi.order_id
WHERE o.created_at >= DATE_SUB(CURRENT_DATE(), INTERVAL 18 MONTH)
GROUP BY DATE_TRUNC(DATE(o.created_at), MONTH)
ORDER BY month
LIMIT 1000
```

CRITICAL FOR FORECASTING: 
- SQL retrieves HISTORICAL monthly data (past 18 months)
- Python will analyze trends and predict future 3 months
- NO future dates generated in SQL - only actual historical dates!

EXAMPLE for "Customer churn analysis":
```sql
SELECT 
    u.id as user_id,
    u.first_name,
    u.last_name,
    DATE_DIFF(CURRENT_DATE(), DATE(MAX(o.created_at)), DAY) as days_since_last_order,
    COUNT(DISTINCT o.order_id) as total_orders,
    SUM(oi.sale_price) as total_spent,
    CASE 
        WHEN DATE_DIFF(CURRENT_DATE(), DATE(MAX(o.created_at)), DAY) > 180 THEN 'Churned'
        WHEN DATE_DIFF(CURRENT_DATE(), DATE(MAX(o.created_at)), DAY) > 90 THEN 'At Risk'
        ELSE 'Active'
    END as churn_status
FROM `bigquery-public-data.thelook_ecommerce.users` u
LEFT JOIN `bigquery-public-data.thelook_ecommerce.orders` o ON u.id = o.user_id
LEFT JOIN `bigquery-public-data.thelook_ecommerce.order_items` oi ON o.order_id = oi.order_id
GROUP BY u.id, u.first_name, u.last_name
ORDER BY days_since_last_order DESC
LIMIT 1000
```

Generate a BigQuery SQL query that directly answers the user's question.
"""
    
    def _format_tables_for_prompt(self, tables: List) -> str:
        """Format table information for the prompt."""
        if not tables:
            return "No specific tables identified"
        
        table_descriptions = []
        for table in tables:
            table_descriptions.append(f"- {table.name}: {table.purpose}")
        
        return "\\n".join(table_descriptions)
    
    def _format_metrics_for_prompt(self, metrics: List[ColumnAnalysis]) -> str:
        """Format target metrics for the prompt."""
        if not metrics:
            return "No specific metrics identified"
        
        metric_descriptions = []
        for metric in metrics:
            agg_type = metric.aggregation_type or "UNKNOWN"
            metric_descriptions.append(f"- {metric.name}: {metric.purpose} (Aggregation: {agg_type})")
        
        return "\\n".join(metric_descriptions)
    
    def _format_dimensions_for_prompt(self, dimensions: List[ColumnAnalysis]) -> str:
        """Format grouping dimensions for the prompt."""
        if not dimensions:
            return "No specific grouping needed"
        
        dim_descriptions = []
        for dim in dimensions:
            dim_descriptions.append(f"- {dim.name}: {dim.purpose}")
        
        return "\\n".join(dim_descriptions)
    
    def _format_detailed_schema(self, data_understanding: DataUnderstanding) -> str:
        """Format detailed schema information showing exact column names for each table."""
        # This method will show the actual BigQuery schema structure
        # to help the AI generate accurate column references
        
        schema_template = """
CRITICAL: Use ONLY these 4 existing tables. Do NOT create or reference any other tables:

1. `bigquery-public-data.thelook_ecommerce.products` (alias: p)
   - id (INTEGER) - Product ID  
   - name (STRING) - Product name
   - brand (STRING) - Product brand
   - category (STRING) - Product category
   - retail_price (FLOAT) - Product retail price

2. `bigquery-public-data.thelook_ecommerce.order_items` (alias: oi)  
   - id (INTEGER) - Order item ID
   - order_id (INTEGER) - Order ID
   - product_id (INTEGER) - Links to products.id
   - sale_price (FLOAT) - ACTUAL REVENUE AMOUNT (use this for revenue calculations)
   - inventory_item_id (INTEGER) - Inventory item

3. `bigquery-public-data.thelook_ecommerce.orders` (alias: o)
   - order_id (INTEGER) - Order ID
   - user_id (INTEGER) - Customer ID  
   - status (STRING) - Order status
   - created_at (TIMESTAMP) - Order date (use DATE(o.created_at) for date operations)

4. `bigquery-public-data.thelook_ecommerce.users` (alias: u)
   - id (INTEGER) - User ID
   - first_name (STRING) - User first name
   - last_name (STRING) - User last name
   - email (STRING) - User email
   - created_at (TIMESTAMP) - User registration date

CRITICAL CONSTRAINTS:
- ONLY use tables 1-4 above. Do NOT create tables like "ChurnAnalysis", "CustomerMetrics", etc.
- ALL table references must include full dataset path: `bigquery-public-data.thelook_ecommerce.TABLE_NAME`
- Use WITH clauses for complex calculations, but still reference the 4 base tables only
- For churn analysis: calculate customer metrics using existing tables, don't create new ones

ANALYSIS PATTERNS:
- Revenue: oi.sale_price (from order_items table)
- Dates: DATE(o.created_at) to convert TIMESTAMP to DATE
- Customer behavior: Join users with orders and order_items
- Churn analysis: Calculate days since last order using existing orders table
"""
        return schema_template
    
    def _parse_sql_response(self, response_content: str, data_understanding: DataUnderstanding) -> SQLGenerationResult:
        """Parse the AI response into an SQLGenerationResult."""
        try:
            # Clean and parse JSON response
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
                confidence=float(result_data.get("confidence", 0.7))
            )
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse SQL response: {e}")
            logger.debug(f"Raw response: {response_content}")
            
            # Try to extract SQL from raw response
            sql_query = self._extract_sql_from_text(response_content)
            
            return SQLGenerationResult(
                sql_query=sql_query,
                explanation="SQL extracted from response",
                estimated_complexity="medium",
                optimization_applied=["text_extraction"],
                tables_used=[table.name for table in data_understanding.relevant_tables],
                metrics_computed=[metric.name for metric in data_understanding.target_metrics],
                confidence=0.5
            )
    
    def _extract_sql_from_text(self, text: str) -> str:
        """Extract SQL query from plain text response."""
        # Look for SQL patterns
        lines = text.split('\\n')
        sql_lines = []
        in_sql = False
        
        for line in lines:
            line = line.strip()
            if line.upper().startswith(('SELECT', 'WITH')):
                in_sql = True
                sql_lines.append(line)
            elif in_sql:
                if line and not line.startswith('--') and not line.startswith('//'):
                    sql_lines.append(line)
                elif line == '' and sql_lines:
                    break
        
        if sql_lines:
            return ' '.join(sql_lines)
        
        # Fallback: return a basic query structure
        return "SELECT * FROM `bigquery-public-data.thelook_ecommerce.orders` LIMIT 100"
    
    def _optimize_and_validate(self, sql_result: SQLGenerationResult, 
                              data_understanding: DataUnderstanding) -> SQLGenerationResult:
        """Apply final optimizations and validations to the SQL."""
        sql_query = sql_result.sql_query
        optimizations = list(sql_result.optimization_applied)
        
        # Ensure dataset prefix
        if "bigquery-public-data.thelook_ecommerce" not in sql_query:
            # Add dataset prefix to common tables
            for table_name in ["orders", "order_items", "products", "users"]:
                sql_query = sql_query.replace(
                    f" {table_name} ", 
                    f" `bigquery-public-data.thelook_ecommerce.{table_name}` "
                )
                sql_query = sql_query.replace(
                    f"FROM {table_name}",
                    f"FROM `bigquery-public-data.thelook_ecommerce.{table_name}`"
                )
                sql_query = sql_query.replace(
                    f"JOIN {table_name}",
                    f"JOIN `bigquery-public-data.thelook_ecommerce.{table_name}`"
                )
            optimizations.append("dataset_prefix_added")
        
        # Ensure LIMIT clause for performance
        if "LIMIT" not in sql_query.upper():
            sql_query += " LIMIT 10000"
            optimizations.append("limit_added")
        
        # Clean up formatting
        sql_query = sql_query.strip()
        if sql_query.endswith(';'):
            sql_query = sql_query[:-1]
        
        return SQLGenerationResult(
            sql_query=sql_query,
            explanation=sql_result.explanation,
            estimated_complexity=sql_result.estimated_complexity,
            optimization_applied=optimizations,
            tables_used=sql_result.tables_used,
            metrics_computed=sql_result.metrics_computed,
            confidence=sql_result.confidence
        )
    
    
    def _init_sql_validator(self):
        """Initialize LangChain SQL validation chain."""
        try:
            # Initialize Gemini for LangChain validation
            from config import config
            
            self.langchain_llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=config.api_configurations.gemini_api_key,
                temperature=0.1
            )
            
            # Define the BigQuery-specific validation prompt
            validation_system_prompt = """Double check the BigQuery SQL query for common mistakes, including:
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
            - Complete table paths: `bigquery-public-data.thelook_ecommerce.TABLE_NAME`
            
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
            
            # Create the validation prompt template
            self.validation_prompt = ChatPromptTemplate.from_messages([
                ("system", validation_system_prompt),
                ("human", "Validate this BigQuery SQL query:\n\n{query}")
            ])
            
            # Create the validation chain
            self.sql_validation_chain = self.validation_prompt | self.langchain_llm | StrOutputParser()
            
            logger.info("LangChain SQL validator initialized successfully")
            
        except Exception as e:
            logger.warning(f"Failed to initialize LangChain SQL validator: {e}")
            self.sql_validation_chain = None
    
    def _validate_sql_with_langchain(self, sql_query: str) -> str:
        """Validate SQL query using LangChain validation chain."""
        if not hasattr(self, 'sql_validation_chain') or self.sql_validation_chain is None:
            logger.warning("LangChain SQL validator not available, using basic cleaning")
            return self._validate_and_clean_sql(sql_query)
        
        try:
            # Use LangChain validation chain
            validated_sql = self.sql_validation_chain.invoke({"query": sql_query})
            
            # Clean up the response
            validated_sql = validated_sql.strip()
            if validated_sql.startswith("```sql"):
                validated_sql = validated_sql[6:]
            elif validated_sql.startswith("```"):
                validated_sql = validated_sql[3:]
            if validated_sql.endswith("```"):
                validated_sql = validated_sql[:-3]
            
            # Basic cleanup
            validated_sql = validated_sql.strip()
            if validated_sql.endswith(';'):
                validated_sql = validated_sql[:-1]
            
            # Ensure LIMIT clause
            if 'LIMIT' not in validated_sql.upper():
                validated_sql += ' LIMIT 10000'
            
            logger.info("SQL query validated successfully with LangChain")
            return validated_sql
            
        except Exception as e:
            logger.warning(f"LangChain SQL validation failed: {e}, falling back to basic cleaning")
            # Basic fallback cleaning only
            validated_sql = sql_query.strip()
            if validated_sql.startswith("```sql"):
                validated_sql = validated_sql[6:]
            elif validated_sql.startswith("```"):
                validated_sql = validated_sql[3:]
            if validated_sql.endswith("```"):
                validated_sql = validated_sql[:-3]
            validated_sql = validated_sql.strip()
            if validated_sql.endswith(';'):
                validated_sql = validated_sql[:-1]
            if 'LIMIT' not in validated_sql.upper():
                validated_sql += ' LIMIT 10000'
            return validated_sql
    
    def _create_fallback_sql(self, query: str, data_understanding: DataUnderstanding) -> SQLGenerationResult:
        """Create a basic fallback SQL when generation fails."""
        logger.info("Creating fallback SQL query")
        
        # Use the most relevant table or default to orders
        primary_table = "orders"
        if data_understanding.relevant_tables:
            primary_table = data_understanding.relevant_tables[0].name
        
        fallback_sql = f"""
        SELECT *
        FROM `bigquery-public-data.thelook_ecommerce.{primary_table}`
        LIMIT 100
        """
        
        return SQLGenerationResult(
            sql_query=fallback_sql.strip(),
            explanation=f"Fallback query for table: {primary_table}",
            estimated_complexity="low",
            optimization_applied=["fallback_generation"],
            tables_used=[primary_table],
            metrics_computed=[],
            confidence=0.3
        )


# Global SQL agent instance
sql_agent = SQLGenerationAgent()
