"""Reflection system for analyzing execution failures and learning from them."""

from .error_categorization import (
    ErrorCategory,
    ErrorClassifier,
    ErrorAnalysis,
    CategorizedError,
)
from .reflection_engine import (
    ReflectionEngine,
    ReflectionResult,
    FailureContext,
)
from .pattern_detection import (
    FailurePattern,
    PatternDetector,
    PatternMatch,
)
from .learning_system import (
    LearningSystem,
    LearningRecord,
    ImprovementSuggestion,
)

__all__ = [
    "ErrorCategory",
    "ErrorClassifier", 
    "ErrorAnalysis",
    "CategorizedError",
    "ReflectionEngine",
    "ReflectionResult",
    "FailureContext",
    "FailurePattern",
    "PatternDetector",
    "PatternMatch",
    "LearningSystem",
    "LearningRecord",
    "ImprovementSuggestion",
]