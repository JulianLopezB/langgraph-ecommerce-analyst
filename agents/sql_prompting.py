"""Utilities for building SQL generation prompts."""

from typing import List

from agents.process_classifier import ProcessTypeResult
from agents.schema_agent import ColumnAnalysis, DataUnderstanding


def create_sql_generation_prompt(
    query: str,
    data_understanding: DataUnderstanding,
    process_result: ProcessTypeResult,
    dataset_id: str,
) -> str:
    """Create the AI prompt for SQL generation."""

    metrics_info = _format_metrics_for_prompt(data_understanding.target_metrics)
    dimensions_info = _format_dimensions_for_prompt(
        data_understanding.grouping_dimensions
    )
    schema_details = _format_detailed_schema(data_understanding)

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
   - Every table reference must include full path: `{DATASET_ID}.TABLE_NAME`

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
   - For quarters: Use EXTRACT(QUARTER FROM DATE(timestamp_column)) NOT QUARTER(timestamp_column)
   - For months: Use EXTRACT(MONTH FROM DATE(timestamp_column))
   - For years: Use EXTRACT(YEAR FROM DATE(timestamp_column))
   - For RFM: DATE_DIFF(CURRENT_DATE(), DATE(o.created_at), DAY) AS days_since_last_order

   CRITICAL: BigQuery uses EXTRACT(part FROM date) syntax, not QUARTER(), MONTH(), YEAR() functions!

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
FROM `{DATASET_ID}.order_items` oi
JOIN `{DATASET_ID}.products` p
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
FROM `{DATASET_ID}.orders` o
JOIN `{DATASET_ID}.order_items` oi
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
FROM `{DATASET_ID}.orders` o
JOIN `{DATASET_ID}.order_items` oi
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

EXAMPLE for "quarterly sales trends":
```sql
SELECT
    EXTRACT(YEAR FROM DATE(o.created_at)) as year,
    EXTRACT(QUARTER FROM DATE(o.created_at)) as quarter,
    SUM(oi.sale_price) as quarterly_revenue,
    COUNT(DISTINCT o.order_id) as quarterly_orders
FROM `{DATASET_ID}.orders` o
JOIN `{DATASET_ID}.order_items` oi
    ON o.order_id = oi.order_id
WHERE o.created_at >= DATE_SUB(CURRENT_DATE(), INTERVAL 2 YEAR)
GROUP BY EXTRACT(YEAR FROM DATE(o.created_at)), EXTRACT(QUARTER FROM DATE(o.created_at))
ORDER BY year, quarter
LIMIT 1000
```

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
FROM `{DATASET_ID}.users` u
LEFT JOIN `{DATASET_ID}.orders` o ON u.id = o.user_id
LEFT JOIN `{DATASET_ID}.order_items` oi ON o.order_id = oi.order_id
GROUP BY u.id, u.first_name, u.last_name
ORDER BY days_since_last_order DESC
LIMIT 1000
```

Generate a BigQuery SQL query that directly answers the user's question.
""".format(
        DATASET_ID=dataset_id
    )


def _format_tables_for_prompt(tables: List) -> str:
    """Format table information for the prompt."""
    if not tables:
        return "No specific tables identified"

    table_descriptions = []
    for table in tables:
        table_descriptions.append(f"- {table.name}: {table.purpose}")

    return "\n".join(table_descriptions)


def _format_metrics_for_prompt(metrics: List[ColumnAnalysis]) -> str:
    """Format target metrics for the prompt."""
    if not metrics:
        return "No specific metrics identified"

    metric_descriptions = []
    for metric in metrics:
        agg_type = metric.aggregation_type or "UNKNOWN"
        metric_descriptions.append(
            f"- {metric.name}: {metric.purpose} (Aggregation: {agg_type})"
        )

    return "\n".join(metric_descriptions)


def _format_dimensions_for_prompt(dimensions: List[ColumnAnalysis]) -> str:
    """Format grouping dimensions for the prompt."""
    if not dimensions:
        return "No specific grouping needed"

    dim_descriptions = []
    for dim in dimensions:
        dim_descriptions.append(f"- {dim.name}: {dim.purpose}")

    return "\n".join(dim_descriptions)


def _format_detailed_schema(data_understanding: DataUnderstanding) -> str:
    """Format detailed schema information showing exact column names for each table."""
    schema_template = f"""
CRITICAL: Use ONLY these 4 existing tables. Do NOT create or reference any other tables:

1. `{DATASET_ID}.products` (alias: p)
   - id (INTEGER) - Product ID
   - name (STRING) - Product name
   - brand (STRING) - Product brand
   - category (STRING) - Product category
   - retail_price (FLOAT) - Product retail price

2. `{DATASET_ID}.order_items` (alias: oi)
   - id (INTEGER) - Order item ID
   - order_id (INTEGER) - Order ID
   - product_id (INTEGER) - Links to products.id
   - sale_price (FLOAT) - ACTUAL REVENUE AMOUNT (use this for revenue calculations)
   - inventory_item_id (INTEGER) - Inventory item

3. `{DATASET_ID}.orders` (alias: o)
   - order_id (INTEGER) - Order ID
   - user_id (INTEGER) - Customer ID
   - status (STRING) - Order status
   - created_at (TIMESTAMP) - Order date (use DATE(o.created_at) for date operations)

4. `{DATASET_ID}.users` (alias: u)
   - id (INTEGER) - User ID
   - first_name (STRING) - User first name
   - last_name (STRING) - User last name
   - email (STRING) - User email
   - created_at (TIMESTAMP) - User registration date

CRITICAL CONSTRAINTS:
- ONLY use tables 1-4 above. Do NOT create tables like "ChurnAnalysis", "CustomerMetrics", etc.
- ALL table references must include full dataset path: `{DATASET_ID}.TABLE_NAME`
- Use WITH clauses for complex calculations, but still reference the 4 base tables only
- For churn analysis: calculate customer metrics using existing tables, don't create new ones

ANALYSIS PATTERNS:
- Revenue: oi.sale_price (from order_items table)
- Dates: DATE(o.created_at) to convert TIMESTAMP to DATE
- Customer behavior: Join users with orders and order_items
- Churn analysis: Calculate days since last order using existing orders table
""".format(
        DATASET_ID=dataset_id
    )
    return schema_template
