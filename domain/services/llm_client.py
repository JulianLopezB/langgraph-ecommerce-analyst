from abc import ABC, abstractmethod


class LLMClient(ABC):
    """Interface for language model clients."""

    @abstractmethod
    def generate_text(self, prompt: str, **kwargs: object) -> str:
        """Generate natural language text from the given prompt."""

    @abstractmethod
    def generate_code(self, prompt: str, **kwargs: object) -> str:
        """Generate source code based on the provided prompt."""
