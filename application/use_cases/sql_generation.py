"""Use case for generating SQL queries from natural language."""

from typing import Any, Dict

from domain.services import LLMClient


class SQLGenerationUseCase:
    """Generate SQL using a language model."""

    def __init__(self, llm_client: LLMClient) -> None:
        """Initialize with required service interfaces.

        Args:
            llm_client: Interface for language model operations.
        """
        self._llm_client = llm_client

    def generate(self, query: str, schema_info: Dict[str, Any]) -> str:
        """Generate SQL code for the given query and schema."""
        prompt = f"Generate SQL for query: {query}\nSchema: {schema_info}"
        return self._llm_client.generate_code(prompt)
