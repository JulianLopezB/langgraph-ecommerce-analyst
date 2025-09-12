import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.sql_utils import clean_sql_query

DATASET = "bigquery-public-data.thelook_ecommerce"
MAX_RESULTS = 10000


def test_add_dataset_prefix_with_alias():
    query = "SELECT * FROM orders o JOIN order_items oi ON o.id = oi.order_id"
    expected = (
        f"SELECT * FROM `{DATASET}.orders` o "
        f"JOIN `{DATASET}.order_items` oi ON o.id = oi.order_id"
    )
    assert clean_sql_query(query, DATASET, MAX_RESULTS, add_limit=False) == expected


def test_does_not_modify_substrings():
    query = "SELECT * FROM preorders"
    assert clean_sql_query(query, DATASET, MAX_RESULTS, add_limit=False) == "SELECT * FROM preorders"


def test_handles_mixed_case_table_names():
    query = "SELECT * FROM Orders"
    expected = f"SELECT * FROM `{DATASET}.orders`"
    assert clean_sql_query(query, DATASET, MAX_RESULTS, add_limit=False) == expected


def test_removes_markdown_fences_and_semicolon():
    query = "```sql\nSELECT * FROM orders;\n```"
    expected = "SELECT * FROM orders"
    assert (
        clean_sql_query(
            query, DATASET, MAX_RESULTS, add_dataset_prefix=False, add_limit=False
        )
        == expected
    )


def test_appends_limit_when_missing():
    query = "SELECT * FROM orders"
    expected = f"SELECT * FROM orders LIMIT {MAX_RESULTS}"
    assert (
        clean_sql_query(query, DATASET, MAX_RESULTS, add_dataset_prefix=False)
        == expected
    )


def test_does_not_duplicate_limit_clause():
    query = "SELECT * FROM orders LIMIT 5"
    expected = "SELECT * FROM orders LIMIT 5"
    assert (
        clean_sql_query(query, DATASET, MAX_RESULTS, add_dataset_prefix=False)
        == expected
    )

