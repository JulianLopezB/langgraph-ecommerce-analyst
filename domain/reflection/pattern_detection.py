"""Pattern detection system for identifying common failure modes."""

import json
import pickle
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict, Counter
import hashlib

from infrastructure.logging import get_logger
from .error_categorization import CategorizedError, ErrorCategory

logger = get_logger(__name__)


@dataclass
class FailurePattern:
    """Represents a common failure pattern."""
    
    pattern_id: str
    name: str
    description: str
    error_category: ErrorCategory
    frequency: int = 0
    success_rate: float = 0.0  # Rate of successful resolution
    
    # Pattern characteristics
    common_triggers: List[str] = field(default_factory=list)
    common_fixes: List[str] = field(default_factory=list)
    code_patterns: List[str] = field(default_factory=list)
    query_patterns: List[str] = field(default_factory=list)
    
    # Learning data
    examples: List[Dict[str, Any]] = field(default_factory=list)
    last_updated: float = 0.0
    confidence: float = 0.0


@dataclass
class PatternMatch:
    """Represents a match between a failure and a known pattern."""
    
    pattern: FailurePattern
    similarity_score: float
    matching_features: List[str]
    suggested_resolution: str
    confidence: float


class PatternDetector:
    """Detects patterns in execution failures and learns from them."""
    
    def __init__(self, storage_path: Optional[str] = None):
        """Initialize the pattern detector."""
        self.logger = logger.getChild("PatternDetector")
        self.storage_path = Path(storage_path or "data/reflection_patterns.pkl")
        self.patterns: Dict[str, FailurePattern] = {}
        self.pattern_embeddings: Dict[str, List[float]] = {}
        
        # Ensure storage directory exists
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing patterns
        self._load_patterns()
        
        # Initialize with common patterns if none exist
        if not self.patterns:
            self._initialize_common_patterns()

    def detect_patterns(
        self, 
        categorized_error: CategorizedError,
        query: str,
        code: str,
        context: Dict[str, Any] = None
    ) -> List[PatternMatch]:
        """
        Detect patterns that match the given failure.
        
        Args:
            categorized_error: The categorized error
            query: User query that led to failure
            code: Generated code that failed
            context: Additional context
            
        Returns:
            List of pattern matches sorted by similarity
        """
        context = context or {}
        matches = []
        
        # Extract features from the current failure
        features = self._extract_failure_features(categorized_error, query, code, context)
        
        # Compare with known patterns
        for pattern in self.patterns.values():
            similarity = self._calculate_pattern_similarity(features, pattern)
            
            if similarity > 0.3:  # Minimum similarity threshold
                matching_features = self._identify_matching_features(features, pattern)
                suggested_resolution = self._generate_resolution_suggestion(pattern, features)
                
                match = PatternMatch(
                    pattern=pattern,
                    similarity_score=similarity,
                    matching_features=matching_features,
                    suggested_resolution=suggested_resolution,
                    confidence=similarity * pattern.confidence
                )
                matches.append(match)
        
        # Sort by similarity score
        matches.sort(key=lambda x: x.similarity_score, reverse=True)
        
        self.logger.info(f"Found {len(matches)} pattern matches for failure")
        return matches[:5]  # Return top 5 matches

    def learn_from_failure(
        self,
        categorized_error: CategorizedError,
        query: str, 
        code: str,
        resolution_success: bool = False,
        resolution_method: str = "",
        context: Dict[str, Any] = None
    ) -> None:
        """
        Learn from a failure by updating or creating patterns.
        
        Args:
            categorized_error: The categorized error
            query: User query
            code: Failed code
            resolution_success: Whether the failure was successfully resolved
            resolution_method: Method used to resolve (if successful)
            context: Additional context
        """
        context = context or {}
        
        features = self._extract_failure_features(categorized_error, query, code, context)
        
        # Try to find existing pattern to update
        existing_pattern = self._find_best_matching_pattern(features)
        
        if existing_pattern and self._calculate_pattern_similarity(features, existing_pattern) > 0.7:
            # Update existing pattern
            self._update_pattern(existing_pattern, features, resolution_success, resolution_method)
            self.logger.info(f"Updated existing pattern: {existing_pattern.name}")
        else:
            # Create new pattern
            new_pattern = self._create_new_pattern(features, categorized_error, resolution_success, resolution_method)
            self.patterns[new_pattern.pattern_id] = new_pattern
            self.logger.info(f"Created new pattern: {new_pattern.name}")
        
        # Save updated patterns
        self._save_patterns()

    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get statistics about learned patterns."""
        if not self.patterns:
            return {"total_patterns": 0}
            
        category_counts = Counter(p.error_category for p in self.patterns.values())
        avg_frequency = sum(p.frequency for p in self.patterns.values()) / len(self.patterns)
        avg_success_rate = sum(p.success_rate for p in self.patterns.values()) / len(self.patterns)
        
        return {
            "total_patterns": len(self.patterns),
            "category_distribution": dict(category_counts),
            "average_frequency": avg_frequency,
            "average_success_rate": avg_success_rate,
            "most_common_pattern": max(self.patterns.values(), key=lambda p: p.frequency).name,
            "highest_success_pattern": max(self.patterns.values(), key=lambda p: p.success_rate).name,
        }

    def _extract_failure_features(
        self,
        categorized_error: CategorizedError,
        query: str,
        code: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract features from a failure for pattern matching."""
        features = {
            # Error features
            "error_category": categorized_error.analysis.category.value,
            "error_type": type(categorized_error.original_error).__name__,
            "error_message_hash": self._hash_error_message(categorized_error.error_message),
            "primary_cause": categorized_error.analysis.primary_cause,
            "severity": categorized_error.analysis.severity,
            
            # Query features
            "query_length": len(query),
            "query_words": len(query.split()),
            "query_keywords": self._extract_query_keywords(query),
            "query_intent": self._classify_query_intent(query),
            
            # Code features
            "code_length": len(code),
            "code_lines": len(code.split('\n')),
            "code_functions": self._extract_code_functions(code),
            "code_libraries": self._extract_code_libraries(code),
            "code_patterns": self._extract_code_patterns(code),
            
            # Context features
            "has_dataframe": "raw_dataset" in context,
            "dataframe_shape": getattr(context.get("raw_dataset"), "shape", None),
            "dataframe_columns": list(getattr(context.get("raw_dataset"), "columns", [])),
        }
        
        return features

    def _hash_error_message(self, error_message: str) -> str:
        """Create a hash of the error message for similarity comparison."""
        # Normalize the error message (remove specific values, paths, etc.)
        normalized = error_message.lower()
        normalized = re.sub(r'\d+', 'NUM', normalized)  # Replace numbers
        normalized = re.sub(r"'[^']*'", 'STR', normalized)  # Replace quoted strings
        normalized = re.sub(r'/[^\s]*', 'PATH', normalized)  # Replace paths
        
        return hashlib.md5(normalized.encode()).hexdigest()[:8]

    def _extract_query_keywords(self, query: str) -> List[str]:
        """Extract important keywords from the query."""
        # Common data analysis keywords
        keywords = [
            "plot", "chart", "graph", "visualize", "show",
            "correlation", "relationship", "compare", "analyze",
            "group", "aggregate", "sum", "count", "average", "mean",
            "filter", "select", "where", "sort", "order",
            "prediction", "forecast", "model", "trend",
            "statistics", "distribution", "summary"
        ]
        
        query_lower = query.lower()
        found_keywords = [kw for kw in keywords if kw in query_lower]
        return found_keywords

    def _classify_query_intent(self, query: str) -> str:
        """Classify the intent of the query."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["plot", "chart", "graph", "visualize"]):
            return "visualization"
        elif any(word in query_lower for word in ["correlation", "relationship", "compare"]):
            return "analysis"
        elif any(word in query_lower for word in ["group", "aggregate", "sum", "count"]):
            return "aggregation"
        elif any(word in query_lower for word in ["prediction", "forecast", "model"]):
            return "prediction"
        elif any(word in query_lower for word in ["filter", "select", "where"]):
            return "filtering"
        else:
            return "general"

    def _extract_code_functions(self, code: str) -> List[str]:
        """Extract function calls from code."""
        import re
        # Find function calls (word followed by parentheses)
        pattern = r'(\w+)\s*\('
        functions = re.findall(pattern, code)
        return list(set(functions))  # Remove duplicates

    def _extract_code_libraries(self, code: str) -> List[str]:
        """Extract imported libraries from code."""
        import re
        libraries = []
        
        # Find import statements
        import_patterns = [
            r'import\s+(\w+)',
            r'from\s+(\w+)\s+import',
        ]
        
        for pattern in import_patterns:
            matches = re.findall(pattern, code)
            libraries.extend(matches)
            
        return list(set(libraries))

    def _extract_code_patterns(self, code: str) -> List[str]:
        """Extract common code patterns."""
        patterns = []
        code_lower = code.lower()
        
        # Common pandas patterns
        if ".groupby(" in code_lower:
            patterns.append("groupby_operation")
        if ".merge(" in code_lower or ".join(" in code_lower:
            patterns.append("data_joining")
        if ".plot(" in code_lower or "plt." in code_lower:
            patterns.append("plotting")
        if ".apply(" in code_lower:
            patterns.append("apply_function")
        if "for " in code_lower and " in " in code_lower:
            patterns.append("iteration")
            
        return patterns

    def _calculate_pattern_similarity(self, features: Dict[str, Any], pattern: FailurePattern) -> float:
        """Calculate similarity between failure features and a pattern."""
        similarity_scores = []
        
        # Error category match (high weight)
        if features["error_category"] == pattern.error_category.value:
            similarity_scores.append(0.4)
        else:
            similarity_scores.append(0.0)
            
        # Query intent match
        query_intent = features.get("query_intent", "")
        if query_intent in pattern.query_patterns:
            similarity_scores.append(0.2)
        else:
            similarity_scores.append(0.0)
            
        # Code pattern match
        code_patterns = features.get("code_patterns", [])
        pattern_overlap = len(set(code_patterns) & set(pattern.code_patterns))
        if pattern.code_patterns:
            similarity_scores.append(0.2 * pattern_overlap / len(pattern.code_patterns))
        else:
            similarity_scores.append(0.0)
            
        # Query keywords match
        query_keywords = features.get("query_keywords", [])
        keyword_overlap = len(set(query_keywords) & set(pattern.common_triggers))
        if pattern.common_triggers:
            similarity_scores.append(0.2 * keyword_overlap / len(pattern.common_triggers))
        else:
            similarity_scores.append(0.0)
            
        return sum(similarity_scores)

    def _identify_matching_features(self, features: Dict[str, Any], pattern: FailurePattern) -> List[str]:
        """Identify which features match between failure and pattern."""
        matches = []
        
        if features["error_category"] == pattern.error_category.value:
            matches.append(f"Error category: {features['error_category']}")
            
        query_keywords = features.get("query_keywords", [])
        common_keywords = set(query_keywords) & set(pattern.common_triggers)
        if common_keywords:
            matches.append(f"Query keywords: {', '.join(common_keywords)}")
            
        code_patterns = features.get("code_patterns", [])
        common_patterns = set(code_patterns) & set(pattern.code_patterns)
        if common_patterns:
            matches.append(f"Code patterns: {', '.join(common_patterns)}")
            
        return matches

    def _generate_resolution_suggestion(self, pattern: FailurePattern, features: Dict[str, Any]) -> str:
        """Generate a resolution suggestion based on the pattern."""
        if pattern.common_fixes:
            # Use the most common fix for this pattern
            return pattern.common_fixes[0]
        else:
            return f"Apply general fixes for {pattern.error_category.value} errors"

    def _find_best_matching_pattern(self, features: Dict[str, Any]) -> Optional[FailurePattern]:
        """Find the best matching existing pattern."""
        best_pattern = None
        best_similarity = 0.0
        
        for pattern in self.patterns.values():
            similarity = self._calculate_pattern_similarity(features, pattern)
            if similarity > best_similarity:
                best_similarity = similarity
                best_pattern = pattern
                
        return best_pattern if best_similarity > 0.5 else None

    def _update_pattern(
        self,
        pattern: FailurePattern,
        features: Dict[str, Any],
        resolution_success: bool,
        resolution_method: str
    ) -> None:
        """Update an existing pattern with new data."""
        pattern.frequency += 1
        
        # Update success rate
        if resolution_success:
            pattern.success_rate = (pattern.success_rate * (pattern.frequency - 1) + 1) / pattern.frequency
        else:
            pattern.success_rate = (pattern.success_rate * (pattern.frequency - 1)) / pattern.frequency
            
        # Add new triggers and fixes
        query_keywords = features.get("query_keywords", [])
        for keyword in query_keywords:
            if keyword not in pattern.common_triggers:
                pattern.common_triggers.append(keyword)
                
        if resolution_success and resolution_method:
            if resolution_method not in pattern.common_fixes:
                pattern.common_fixes.append(resolution_method)
                
        # Add code patterns
        code_patterns = features.get("code_patterns", [])
        for code_pattern in code_patterns:
            if code_pattern not in pattern.code_patterns:
                pattern.code_patterns.append(code_pattern)
                
        # Update confidence based on frequency and success rate
        pattern.confidence = min(1.0, pattern.frequency / 10.0) * pattern.success_rate
        
        pattern.last_updated = time.time()

    def _create_new_pattern(
        self,
        features: Dict[str, Any],
        categorized_error: CategorizedError,
        resolution_success: bool,
        resolution_method: str
    ) -> FailurePattern:
        """Create a new failure pattern."""
        import time
        
        pattern_id = f"{features['error_category']}_{features.get('query_intent', 'unknown')}_{int(time.time())}"
        
        pattern = FailurePattern(
            pattern_id=pattern_id,
            name=f"{features['error_category'].title()} in {features.get('query_intent', 'unknown')} queries",
            description=f"Common {features['error_category']} errors when handling {features.get('query_intent', 'unknown')} requests",
            error_category=categorized_error.analysis.category,
            frequency=1,
            success_rate=1.0 if resolution_success else 0.0,
            common_triggers=features.get("query_keywords", []),
            common_fixes=[resolution_method] if resolution_success and resolution_method else [],
            code_patterns=features.get("code_patterns", []),
            query_patterns=[features.get("query_intent", "")],
            last_updated=time.time(),
            confidence=0.1,  # Start with low confidence
        )
        
        return pattern

    def _initialize_common_patterns(self) -> None:
        """Initialize with common failure patterns."""
        import time
        
        common_patterns = [
            {
                "name": "DataFrame Column Not Found",
                "description": "Errors caused by referencing non-existent DataFrame columns",
                "category": ErrorCategory.DATA,
                "triggers": ["column", "attribute", "key"],
                "fixes": ["Verify column names", "Check DataFrame structure", "Handle missing columns"],
                "code_patterns": ["dataframe_access"],
                "query_patterns": ["filtering", "analysis"],
            },
            {
                "name": "Import Error in Analysis",
                "description": "Missing library imports in generated code",
                "category": ErrorCategory.DEPENDENCY,
                "triggers": ["import", "library", "module"],
                "fixes": ["Add missing imports", "Check library availability", "Use alternative libraries"],
                "code_patterns": ["plotting", "analysis"],
                "query_patterns": ["visualization", "analysis"],
            },
            {
                "name": "Syntax Error in Generated Code",
                "description": "Python syntax errors in generated analysis code",
                "category": ErrorCategory.SYNTAX,
                "triggers": ["syntax", "indentation", "parentheses"],
                "fixes": ["Fix syntax errors", "Check indentation", "Validate code structure"],
                "code_patterns": ["complex_logic"],
                "query_patterns": ["general"],
            },
        ]
        
        for pattern_data in common_patterns:
            pattern_id = f"common_{pattern_data['category'].value}_{int(time.time())}"
            pattern = FailurePattern(
                pattern_id=pattern_id,
                name=pattern_data["name"],
                description=pattern_data["description"],
                error_category=pattern_data["category"],
                frequency=0,
                success_rate=0.8,  # Assume good success rate for common patterns
                common_triggers=pattern_data["triggers"],
                common_fixes=pattern_data["fixes"],
                code_patterns=pattern_data["code_patterns"],
                query_patterns=pattern_data["query_patterns"],
                last_updated=time.time(),
                confidence=0.8,
            )
            self.patterns[pattern_id] = pattern
            
        self.logger.info(f"Initialized {len(common_patterns)} common patterns")

    def _load_patterns(self) -> None:
        """Load patterns from storage."""
        try:
            if self.storage_path.exists():
                with open(self.storage_path, 'rb') as f:
                    data = pickle.load(f)
                    self.patterns = data.get("patterns", {})
                    self.pattern_embeddings = data.get("embeddings", {})
                self.logger.info(f"Loaded {len(self.patterns)} patterns from storage")
            else:
                self.logger.info("No existing patterns found, starting fresh")
        except Exception as e:
            self.logger.error(f"Failed to load patterns: {e}")
            self.patterns = {}
            self.pattern_embeddings = {}

    def _save_patterns(self) -> None:
        """Save patterns to storage."""
        try:
            data = {
                "patterns": self.patterns,
                "embeddings": self.pattern_embeddings,
            }
            with open(self.storage_path, 'wb') as f:
                pickle.dump(data, f)
            self.logger.debug("Patterns saved to storage")
        except Exception as e:
            self.logger.error(f"Failed to save patterns: {e}")