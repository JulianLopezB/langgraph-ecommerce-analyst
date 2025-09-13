"""Persistence layer bindings."""
from .base import DataRepository
from .bigquery import BigQueryRepository
from .in_memory_session_store import InMemorySessionStore
from .filesystem_artifact_store import FilesystemArtifactStore

# Default repository binding used across the application; injected at runtime
data_repository: DataRepository | None = None

__all__ = [
    "DataRepository",
    "data_repository",
    "BigQueryRepository",
    "InMemorySessionStore",
    "FilesystemArtifactStore",
]
