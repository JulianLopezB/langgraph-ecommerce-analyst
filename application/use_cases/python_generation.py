"""Use case for generating Python analysis code."""

from domain.services import LLMClient


class PythonGenerationUseCase:
    """Generate Python code using a language model."""

    def __init__(self, llm_client: LLMClient) -> None:
        """Initialize with required service interfaces.

        Args:
            llm_client: Interface for language model operations.
        """
        self._llm_client = llm_client

    def generate(self, prompt: str) -> str:
        """Generate Python code based on the given prompt."""
        return self._llm_client.generate_code(prompt)
