from abc import ABC, abstractmethod
from typing import Any, Dict


class DataRepository(ABC):
    """Interface for accessing analytical data sources."""

    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """Return schema information for available datasets."""

    @abstractmethod
    def run_query(self, query: str) -> Any:
        """Execute a query against the data source and return results."""
