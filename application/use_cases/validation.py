"""Use case for validating generated code."""
from domain.services import CodeValidator


class CodeValidationUseCase:
    """Validate source code before execution."""

    def __init__(self, validator: CodeValidator) -> None:
        """Initialize with required service interfaces.

        Args:
            validator: Interface for code validation.
        """
        self._validator = validator

    def validate(self, code: str) -> bool:
        """Validate the provided code snippet."""
        return self._validator.validate(code)
