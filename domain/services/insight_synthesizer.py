from abc import ABC, abstractmethod
from typing import Any


class InsightSynthesizer(ABC):
    """Interface for transforming analysis results into business insights."""

    @abstractmethod
    def synthesize(self, analysis_results: Any, original_query: str) -> str:
        """Produce user-facing insights from raw analysis results."""
