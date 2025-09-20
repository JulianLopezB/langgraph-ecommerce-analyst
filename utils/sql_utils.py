"""Shared SQL utility functions."""

import re

AGGREGATE_FUNCTION_PATTERN = re.compile(
    r"\b(" +
    "|".join(
        [
            "SUM",
            "COUNT",
            "AVG",
            "MIN",
            "MAX",
            "ARRAY_AGG",
            "STRING_AGG",
            "ANY_VALUE",
            "APPROX_COUNT_DISTINCT",
            "BIT_AND",
            "BIT_OR",
            "BIT_XOR",
            "LOGICAL_AND",
            "LOGICAL_OR",
            "BOOL_AND",
            "BOOL_OR",
            "CORR",
            "COVAR_POP",
            "COVAR_SAMP",
            "STDDEV",
            "STDDEV_POP",
            "STDDEV_SAMP",
            "VARIANCE",
            "VAR_POP",
            "VAR_SAMP",
            "VARIANCE_POP",
            "VARIANCE_SAMP",
            "PERCENTILE_CONT",
            "PERCENTILE_DISC",
            "RANK",
            "DENSE_RANK",
            "ROW_NUMBER",
            "NTILE",
            "LEAD",
            "LAG",
            "FIRST_VALUE",
            "LAST_VALUE",
            "NTH_VALUE",
        ]
    )
    + r")\s*\(",
    re.IGNORECASE,
)


def clean_sql_query(
    sql_query: str,
    dataset_id: str,
    max_results: int,
    add_dataset_prefix: bool = True,
    add_limit: bool = True,
) -> str:
    """
    Clean and format SQL query by removing markdown and adding necessary prefixes.
    Table names are normalized using regex-based whole-word matching that is
    case-insensitive and only prefixes unqualified table names with the
    configured dataset. Substrings such as ``preorders`` remain untouched.

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
    if sql_query.endswith(";"):
        sql_query = sql_query[:-1]

    # Add dataset prefix if requested using whole-word matching
    if add_dataset_prefix:
        pattern = re.compile(
            rf"(?<![\w`]\.)\b({'|'.join(['orders', 'order_items', 'products', 'users'])})\b",
            re.IGNORECASE,
        )
        sql_query = pattern.sub(
            lambda m: f"`{dataset_id}.{m.group(0).lower()}`",
            sql_query,
        )

    # Ensure LIMIT clause for performance if requested
    if add_limit and "LIMIT" not in sql_query.upper():
        sql_query += f" LIMIT {max_results}"

    return sql_query.strip()


def _split_sql_expressions(clause: str) -> list[str]:
    """Split a comma-separated SQL expression list while respecting parentheses."""

    expressions: list[str] = []
    current: list[str] = []
    depth = 0

    for char in clause:
        if char == "(":
            depth += 1
        elif char == ")" and depth > 0:
            depth -= 1

        if char == "," and depth == 0:
            expr = "".join(current).strip()
            if expr:
                expressions.append(expr)
            current = []
            continue

        current.append(char)

    tail = "".join(current).strip()
    if tail:
        expressions.append(tail)

    return expressions


def ensure_valid_group_by(sql_query: str) -> str:
    """Ensure GROUP BY clauses do not contain aggregate expressions."""

    group_by_pattern = re.compile(
        r"(GROUP\s+BY)\s+(.+?)(?=(ORDER\s+BY|LIMIT|HAVING|QUALIFY|WINDOW|$))",
        re.IGNORECASE | re.DOTALL,
    )

    def _clean_group_by(match: re.Match) -> str:
        keyword = match.group(1)
        clause = match.group(2)

        expressions = _split_sql_expressions(clause)
        filtered = [
            expr
            for expr in expressions
            if not AGGREGATE_FUNCTION_PATTERN.search(expr)
        ]

        if not filtered:
            replacement = ""
        else:
            replacement = f"{keyword} {', '.join(filtered)}"

        if match.end() < len(match.string) and not match.string[match.end()].isspace():
            replacement += " "

        return replacement

    cleaned_query = group_by_pattern.sub(_clean_group_by, sql_query)
    return cleaned_query.strip()


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
        "sql_execution_error": "I had trouble running the database query",
        "code_generation_error": "I had difficulty creating the analysis code",
        "execution_error": "The analysis code ran into an issue",
        "validation_error": "I found a problem with the generated code",
        "understanding_error": "I had trouble understanding your question",
        "sql_generation_error": "I couldn't create the right database query",
    }

    friendly_msg = friendly_messages.get(
        error_type, "I encountered an unexpected issue"
    )

    # Add specific details if they're helpful
    error_msg_str = str(error_msg) if error_msg else ""
    if "timeout" in error_msg_str.lower():
        friendly_msg += " (it took too long to complete)"
    elif "memory" in error_msg_str.lower():
        friendly_msg += " (it needed too much memory)"
    elif "syntax" in error_msg_str.lower():
        friendly_msg += " (there was a formatting issue)"

    return friendly_msg
