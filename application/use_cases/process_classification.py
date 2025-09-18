"""Use case for classifying analytical processes."""

from typing import Any, Dict

from domain.services import LLMClient


class ProcessClassificationUseCase:
    """Classify the type of process needed for a user's query."""

    def __init__(self, llm_client: LLMClient) -> None:
        """Initialize with required service interfaces.

        Args:
            llm_client: Interface for language model operations.
        """
        self._llm_client = llm_client

    def classify(self, query: str, schema_info: Dict[str, Any]) -> str:
        """Use the LLM to classify a query into a process type."""
        prompt = f"Classify the process for query: {query}\nSchema: {schema_info}"
        return self._llm_client.generate_text(prompt)
