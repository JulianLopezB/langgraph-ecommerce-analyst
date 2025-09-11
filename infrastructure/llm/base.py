from abc import ABC, abstractmethod
from typing import Dict, Any


class LLMClient(ABC):
    """Interface for large language model clients."""

    @abstractmethod
    def generate_text(self, prompt: str, temperature: float = 0.3, max_tokens: int = 2048):
        """Generate text from a prompt."""
        raise NotImplementedError

    @abstractmethod
    def generate_adaptive_python_code(self, analysis_context: Dict[str, Any]) -> str:
        """Generate Python code based on analysis context."""
        raise NotImplementedError

    @abstractmethod
    def generate_insights(self, analysis_results: Dict[str, Any], original_query: str) -> str:
        """Generate insights from analysis results."""
        raise NotImplementedError
