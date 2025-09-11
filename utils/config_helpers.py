"""Helper utilities for accessing configuration values."""

from config import config

DATASET_ID: str = config.api_configurations.dataset_id
MAX_RESULTS: int = config.api_configurations.max_query_results
