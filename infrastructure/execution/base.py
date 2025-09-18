from abc import ABC, abstractmethod
from typing import Any, Dict

from domain.entities import ExecutionResults


class CodeExecutor(ABC):
    """Interface for secure code execution environments."""

    @abstractmethod
    def execute_code(
        self, code: str, context: Dict[str, Any] | None = None
    ) -> ExecutionResults:
        """Execute code within a controlled environment."""
        raise NotImplementedError
