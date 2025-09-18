"""Use case for executing code or queries."""

from typing import Any

from domain.services import CodeExecutor, DataRepository


class CodeExecutionUseCase:
    """Execute validated code or SQL queries."""

    def __init__(self, executor: CodeExecutor, data_repository: DataRepository) -> None:
        """Initialize with required service interfaces.

        Args:
            executor: Interface for executing code snippets.
            data_repository: Interface for running SQL queries.
        """
        self._executor = executor
        self._data_repository = data_repository

    def execute_code(self, code: str, data: Any) -> Any:
        """Execute Python code within a data context."""
        return self._executor.execute(code, data)

    def run_query(self, query: str) -> Any:
        """Execute a SQL query through the repository."""
        return self._data_repository.run_query(query)
