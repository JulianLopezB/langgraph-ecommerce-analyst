"""Persistence layer bindings."""
from .base import DataRepository
from .bigquery import BigQueryRepository

# Default repository binding used across the application
data_repository: DataRepository = BigQueryRepository()

__all__ = ["DataRepository", "data_repository", "BigQueryRepository"]
