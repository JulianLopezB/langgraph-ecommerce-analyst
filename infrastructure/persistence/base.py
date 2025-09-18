from abc import ABC, abstractmethod
from typing import Any, Dict, List

import pandas as pd


class DataRepository(ABC):
    """Interface for data repositories."""

    @abstractmethod
    def execute_query(self, sql_query: str) -> pd.DataFrame:
        """Execute a SQL query and return a DataFrame."""
        raise NotImplementedError

    @abstractmethod
    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """Retrieve schema information for a table."""
        raise NotImplementedError
