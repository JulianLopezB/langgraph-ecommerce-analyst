from abc import ABC, abstractmethod


class CodeValidator(ABC):
    """Interface for validating generated code snippets."""

    @abstractmethod
    def validate(self, code: str) -> bool:
        """Validate the given code and return True if it is safe to execute."""
