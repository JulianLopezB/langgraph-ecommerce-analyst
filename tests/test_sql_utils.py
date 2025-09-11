import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import config
from utils.sql_utils import clean_sql_query

DATASET = config.api_configurations.dataset_id


def test_add_dataset_prefix_with_alias():
    query = "SELECT * FROM orders o JOIN order_items oi ON o.id = oi.order_id"
    expected = (
        f"SELECT * FROM `{DATASET}.orders` o "
        f"JOIN `{DATASET}.order_items` oi ON o.id = oi.order_id"
    )
    assert clean_sql_query(query, add_limit=False) == expected


def test_does_not_modify_substrings():
    query = "SELECT * FROM preorders"
    assert clean_sql_query(query, add_limit=False) == "SELECT * FROM preorders"


def test_handles_mixed_case_table_names():
    query = "SELECT * FROM Orders"
    expected = f"SELECT * FROM `{DATASET}.orders`"
    assert clean_sql_query(query, add_limit=False) == expected
