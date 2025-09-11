"""Shared SQL utility functions."""

from utils.config_helpers import DATASET_ID, MAX_RESULTS


def clean_sql_query(sql_query: str, add_dataset_prefix: bool = True, add_limit: bool = True) -> str:
    """
    Clean and format SQL query by removing markdown and adding necessary prefixes.
    
    Args:
        sql_query: Raw SQL query string
        add_dataset_prefix: Whether to add BigQuery dataset prefix
        add_limit: Whether to ensure LIMIT clause exists
        
    Returns:
        Cleaned SQL query string
    """
    # Remove markdown formatting if present
    sql_query = sql_query.strip()
    if sql_query.startswith("```sql"):
        sql_query = sql_query[6:]
    elif sql_query.startswith("```"):
        sql_query = sql_query[3:]
    if sql_query.endswith("```"):
        sql_query = sql_query[:-3]
    
    # Remove trailing semicolon if present
    sql_query = sql_query.strip()
    if sql_query.endswith(';'):
        sql_query = sql_query[:-1]
    
    # Add dataset prefix if missing and requested
    if add_dataset_prefix and DATASET_ID not in sql_query:
        for table in ["orders", "order_items", "products", "users"]:
            sql_query = sql_query.replace(f" {table} ", f" `{DATASET_ID}.{table}` ")
            sql_query = sql_query.replace(f"FROM {table}", f"FROM `{DATASET_ID}.{table}`")
            sql_query = sql_query.replace(f"JOIN {table}", f"JOIN `{DATASET_ID}.{table}`")
    
    # Ensure LIMIT clause for performance if requested
    if add_limit and "LIMIT" not in sql_query.upper():
        sql_query += f" LIMIT {MAX_RESULTS}"
    
    return sql_query.strip()


def format_error_message(error_type: str, error_msg: str) -> str:
    """
    Create user-friendly error messages.
    
    Args:
        error_type: Type of error
        error_msg: Original error message
        
    Returns:
        User-friendly error message
    """
    friendly_messages = {
        'sql_execution_error': "I had trouble running the database query",
        'code_generation_error': "I had difficulty creating the analysis code",  
        'execution_error': "The analysis code ran into an issue",
        'validation_error': "I found a problem with the generated code",
        'understanding_error': "I had trouble understanding your question",
        'sql_generation_error': "I couldn't create the right database query"
    }
    
    friendly_msg = friendly_messages.get(error_type, "I encountered an unexpected issue")
    
    # Add specific details if they're helpful
    error_msg_str = str(error_msg) if error_msg else ""
    if "timeout" in error_msg_str.lower():
        friendly_msg += " (it took too long to complete)"
    elif "memory" in error_msg_str.lower():
        friendly_msg += " (it needed too much memory)"
    elif "syntax" in error_msg_str.lower():
        friendly_msg += " (there was a formatting issue)"

    return friendly_msg
