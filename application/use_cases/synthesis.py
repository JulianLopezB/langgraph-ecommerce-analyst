"""Use case for synthesizing analysis results into insights."""
from typing import Any

from domain.services import InsightSynthesizer


class InsightSynthesisUseCase:
    """Convert analysis results into user-facing insights."""

    def __init__(self, synthesizer: InsightSynthesizer) -> None:
        """Initialize with required service interfaces.

        Args:
            synthesizer: Interface for insight generation.
        """
        self._synthesizer = synthesizer

    def synthesize(self, analysis_results: Any, original_query: str) -> str:
        """Create insights from analysis results and the original query."""
        return self._synthesizer.synthesize(analysis_results, original_query)
