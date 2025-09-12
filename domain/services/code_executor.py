from abc import ABC, abstractmethod
from typing import Any


class CodeExecutor(ABC):
    """Interface for executing validated code snippets."""

    @abstractmethod
    def execute(self, code: str, data: Any) -> Any:
        """Execute code within a provided data context and return the results."""
