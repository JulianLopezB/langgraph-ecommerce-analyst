"""Persistence layer bindings."""
from .base import DataRepository
from .bigquery import BigQueryRepository

# Default repository binding used across the application; injected at runtime
data_repository: DataRepository | None = None

__all__ = ["DataRepository", "data_repository", "BigQueryRepository"]
