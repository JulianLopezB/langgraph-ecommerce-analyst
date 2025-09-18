"""Use case for retrieving and analyzing data schema information."""

from typing import Any, Dict

from domain.services import DataRepository


class SchemaAnalysisUseCase:
    """Fetch schema details for use in further analysis."""

    def __init__(self, data_repository: DataRepository) -> None:
        """Initialize with required service interfaces.

        Args:
            data_repository: Interface to access schema information.
        """
        self._data_repository = data_repository

    def analyze(self) -> Dict[str, Any]:
        """Retrieve schema information from the repository."""
        return self._data_repository.get_schema()
