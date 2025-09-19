"""Learning system for continuous improvement from failure patterns."""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict, Counter
from enum import Enum

from infrastructure.logging import get_logger
from infrastructure.llm.client import LLMClient
from .error_categorization import CategorizedError, ErrorCategory
from .pattern_detection import FailurePattern, PatternMatch

logger = get_logger(__name__)


class ImprovementType(Enum):
    """Types of improvements the system can learn."""
    
    CODE_GENERATION = "code_generation"
    ERROR_HANDLING = "error_handling" 
    QUERY_UNDERSTANDING = "query_understanding"
    VALIDATION = "validation"
    PERFORMANCE = "performance"


@dataclass
class ImprovementSuggestion:
    """Represents a learned improvement suggestion."""
    
    improvement_id: str
    improvement_type: ImprovementType
    title: str
    description: str
    confidence: float
    impact_score: float  # Expected impact (0.0 to 1.0)
    
    # Context
    applicable_patterns: List[str] = field(default_factory=list)
    applicable_categories: List[ErrorCategory] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    
    # Evidence
    supporting_evidence: List[str] = field(default_factory=list)
    success_examples: List[str] = field(default_factory=list)
    failure_examples: List[str] = field(default_factory=list)
    
    # Metadata
    created_time: float = 0.0
    last_validated: float = 0.0
    validation_count: int = 0
    success_count: int = 0


@dataclass
class LearningRecord:
    """Records a learning event for analysis."""
    
    record_id: str
    timestamp: float
    event_type: str  # "failure", "success", "pattern_detected", "improvement_applied"
    
    # Context
    error_category: Optional[ErrorCategory] = None
    query_intent: Optional[str] = None
    resolution_method: Optional[str] = None
    success: bool = False
    
    # Data
    features: Dict[str, Any] = field(default_factory=dict)
    outcome: Dict[str, Any] = field(default_factory=dict)
    
    # Learning insights
    insights: List[str] = field(default_factory=list)
    patterns_matched: List[str] = field(default_factory=list)
    improvements_suggested: List[str] = field(default_factory=list)


class LearningSystem:
    """System for continuous learning from execution failures and successes."""
    
    def __init__(self, llm_client: LLMClient, storage_path: Optional[str] = None):
        """Initialize the learning system."""
        self.llm_client = llm_client
        self.logger = logger.getChild("LearningSystem")
        self.storage_path = Path(storage_path or "data/learning_records.json")
        
        # Learning data
        self.learning_records: List[LearningRecord] = []
        self.improvement_suggestions: Dict[str, ImprovementSuggestion] = {}
        self.performance_metrics: Dict[str, List[float]] = defaultdict(list)
        
        # Ensure storage directory exists
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing data
        self._load_learning_data()

    def record_failure_event(
        self,
        categorized_error: CategorizedError,
        query: str,
        code: str,
        pattern_matches: List[PatternMatch],
        context: Dict[str, Any] = None
    ) -> str:
        """
        Record a failure event for learning.
        
        Args:
            categorized_error: The categorized error
            query: User query
            code: Failed code
            pattern_matches: Patterns that matched this failure
            context: Additional context
            
        Returns:
            Record ID for tracking
        """
        context = context or {}
        record_id = f"failure_{int(time.time())}_{hash(query) % 10000}"
        
        # Extract features for learning
        features = {
            "error_category": categorized_error.analysis.category.value,
            "error_type": type(categorized_error.original_error).__name__,
            "query_length": len(query),
            "code_length": len(code),
            "severity": categorized_error.analysis.severity,
            "confidence": categorized_error.analysis.confidence,
            "contributing_factors": categorized_error.analysis.contributing_factors,
        }
        
        # Record the event
        record = LearningRecord(
            record_id=record_id,
            timestamp=time.time(),
            event_type="failure",
            error_category=categorized_error.analysis.category,
            query_intent=self._classify_query_intent(query),
            success=False,
            features=features,
            outcome={
                "error_message": categorized_error.error_message,
                "suggested_fixes": categorized_error.analysis.suggested_fixes,
            },
            insights=categorized_error.analysis.contributing_factors,
            patterns_matched=[match.pattern.pattern_id for match in pattern_matches],
        )
        
        self.learning_records.append(record)
        self._save_learning_data()
        
        # Analyze for potential improvements
        self._analyze_for_improvements(record)
        
        self.logger.info(f"Recorded failure event: {record_id}")
        return record_id

    def record_success_event(
        self,
        query: str,
        code: str,
        execution_time: float,
        resolution_method: str = "",
        context: Dict[str, Any] = None
    ) -> str:
        """
        Record a successful execution for learning.
        
        Args:
            query: User query
            code: Successful code
            execution_time: Time taken to execute
            resolution_method: Method used to achieve success
            context: Additional context
            
        Returns:
            Record ID for tracking
        """
        context = context or {}
        record_id = f"success_{int(time.time())}_{hash(query) % 10000}"
        
        features = {
            "query_length": len(query),
            "code_length": len(code),
            "execution_time": execution_time,
            "resolution_method": resolution_method,
        }
        
        record = LearningRecord(
            record_id=record_id,
            timestamp=time.time(),
            event_type="success",
            query_intent=self._classify_query_intent(query),
            resolution_method=resolution_method,
            success=True,
            features=features,
            outcome={
                "execution_time": execution_time,
                "success_method": resolution_method,
            },
        )
        
        self.learning_records.append(record)
        self._save_learning_data()
        
        # Update performance metrics
        self.performance_metrics["execution_time"].append(execution_time)
        self.performance_metrics["success_rate"].append(1.0)
        
        self.logger.info(f"Recorded success event: {record_id}")
        return record_id

    def generate_improvement_suggestions(self, limit: int = 10) -> List[ImprovementSuggestion]:
        """
        Generate improvement suggestions based on learning history.
        
        Args:
            limit: Maximum number of suggestions to return
            
        Returns:
            List of improvement suggestions
        """
        self.logger.info("Generating improvement suggestions from learning data")
        
        # Analyze failure patterns
        failure_analysis = self._analyze_failure_patterns()
        
        # Generate suggestions using LLM
        llm_suggestions = self._generate_llm_improvements(failure_analysis)
        
        # Combine with existing suggestions
        all_suggestions = list(self.improvement_suggestions.values()) + llm_suggestions
        
        # Sort by impact and confidence
        all_suggestions.sort(key=lambda x: x.impact_score * x.confidence, reverse=True)
        
        return all_suggestions[:limit]

    def validate_improvement(self, improvement_id: str, success: bool, evidence: str = "") -> None:
        """
        Validate an improvement suggestion with real-world results.
        
        Args:
            improvement_id: ID of the improvement being validated
            success: Whether the improvement was successful
            evidence: Evidence of success/failure
        """
        if improvement_id in self.improvement_suggestions:
            suggestion = self.improvement_suggestions[improvement_id]
            suggestion.validation_count += 1
            suggestion.last_validated = time.time()
            
            if success:
                suggestion.success_count += 1
                if evidence:
                    suggestion.success_examples.append(evidence)
            else:
                if evidence:
                    suggestion.failure_examples.append(evidence)
                    
            # Update confidence based on validation results
            success_rate = suggestion.success_count / suggestion.validation_count
            suggestion.confidence = (suggestion.confidence + success_rate) / 2
            
            self.logger.info(f"Validated improvement {improvement_id}: success={success}")
            self._save_learning_data()

    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights from the learning system."""
        if not self.learning_records:
            return {"message": "No learning data available"}
            
        total_records = len(self.learning_records)
        failure_records = [r for r in self.learning_records if not r.success]
        success_records = [r for r in self.learning_records if r.success]
        
        # Analyze failure categories
        failure_categories = Counter(r.error_category.value for r in failure_records if r.error_category)
        
        # Analyze query intents
        query_intents = Counter(r.query_intent for r in self.learning_records if r.query_intent)
        
        # Calculate success rate trend
        recent_records = sorted(self.learning_records, key=lambda x: x.timestamp)[-100:]
        recent_success_rate = sum(1 for r in recent_records if r.success) / len(recent_records) if recent_records else 0
        
        # Top patterns
        all_patterns = []
        for record in self.learning_records:
            all_patterns.extend(record.patterns_matched)
        top_patterns = Counter(all_patterns).most_common(5)
        
        return {
            "total_learning_events": total_records,
            "failure_events": len(failure_records),
            "success_events": len(success_records),
            "overall_success_rate": len(success_records) / total_records if total_records > 0 else 0,
            "recent_success_rate": recent_success_rate,
            "failure_categories": dict(failure_categories),
            "query_intents": dict(query_intents),
            "top_failure_patterns": [{"pattern": p, "count": c} for p, c in top_patterns],
            "active_improvements": len(self.improvement_suggestions),
            "learning_period_days": (max(r.timestamp for r in self.learning_records) - 
                                   min(r.timestamp for r in self.learning_records)) / 86400 if total_records > 1 else 0,
        }

    def _classify_query_intent(self, query: str) -> str:
        """Classify the intent of a query."""
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

    def _analyze_for_improvements(self, record: LearningRecord) -> None:
        """Analyze a failure record for potential improvements."""
        # Look for recurring patterns
        similar_failures = [
            r for r in self.learning_records 
            if (r.error_category == record.error_category and 
                r.query_intent == record.query_intent and
                r.record_id != record.record_id)
        ]
        
        if len(similar_failures) >= 3:  # Threshold for pattern recognition
            # Generate improvement suggestion
            improvement_id = f"improvement_{record.error_category.value}_{record.query_intent}_{int(time.time())}"
            
            suggestion = ImprovementSuggestion(
                improvement_id=improvement_id,
                improvement_type=self._determine_improvement_type(record),
                title=f"Improve {record.error_category.value} handling for {record.query_intent} queries",
                description=f"Recurring pattern detected: {len(similar_failures)} similar failures",
                confidence=0.7,
                impact_score=0.6,
                applicable_categories=[record.error_category],
                supporting_evidence=[f"Pattern occurs in {len(similar_failures)} cases"],
                created_time=time.time(),
            )
            
            self.improvement_suggestions[improvement_id] = suggestion
            self.logger.info(f"Generated improvement suggestion: {improvement_id}")

    def _determine_improvement_type(self, record: LearningRecord) -> ImprovementType:
        """Determine the type of improvement needed based on the record."""
        if record.error_category == ErrorCategory.SYNTAX:
            return ImprovementType.CODE_GENERATION
        elif record.error_category == ErrorCategory.DATA:
            return ImprovementType.VALIDATION
        elif record.error_category == ErrorCategory.PERFORMANCE:
            return ImprovementType.PERFORMANCE
        elif record.error_category in [ErrorCategory.RUNTIME, ErrorCategory.LOGIC]:
            return ImprovementType.ERROR_HANDLING
        else:
            return ImprovementType.QUERY_UNDERSTANDING

    def _analyze_failure_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in failure records."""
        failure_records = [r for r in self.learning_records if not r.success]
        
        if not failure_records:
            return {}
            
        # Group by error category and query intent
        patterns = defaultdict(list)
        for record in failure_records:
            key = f"{record.error_category.value if record.error_category else 'unknown'}_{record.query_intent or 'unknown'}"
            patterns[key].append(record)
            
        # Analyze each pattern
        pattern_analysis = {}
        for pattern_key, records in patterns.items():
            if len(records) >= 2:  # Only analyze patterns with multiple occurrences
                pattern_analysis[pattern_key] = {
                    "frequency": len(records),
                    "avg_confidence": sum(r.features.get("confidence", 0) for r in records) / len(records),
                    "common_factors": self._extract_common_factors(records),
                    "severity_distribution": Counter(r.features.get("severity") for r in records),
                }
                
        return pattern_analysis

    def _extract_common_factors(self, records: List[LearningRecord]) -> List[str]:
        """Extract common contributing factors from multiple records."""
        all_factors = []
        for record in records:
            all_factors.extend(record.insights)
            
        # Find factors that appear in at least 50% of records
        factor_counts = Counter(all_factors)
        threshold = len(records) * 0.5
        common_factors = [factor for factor, count in factor_counts.items() if count >= threshold]
        
        return common_factors

    def _generate_llm_improvements(self, failure_analysis: Dict[str, Any]) -> List[ImprovementSuggestion]:
        """Generate improvement suggestions using LLM analysis."""
        if not failure_analysis:
            return []
            
        prompt = f"""Analyze the following failure patterns and suggest specific improvements:

Failure Pattern Analysis:
{json.dumps(failure_analysis, indent=2)}

For each significant pattern, provide:
1. A specific improvement suggestion
2. Expected impact (0.0 to 1.0)
3. Implementation approach
4. Success criteria

Focus on actionable improvements that can be implemented in the code generation or error handling systems."""

        try:
            response = self.llm_client.generate_text(
                prompt=prompt,
                max_tokens=1000,
                temperature=0.3,
            )
            
            # Parse the response and create improvement suggestions
            suggestions = self._parse_llm_improvements(response)
            return suggestions
            
        except Exception as e:
            self.logger.error(f"LLM improvement generation failed: {e}")
            return []

    def _parse_llm_improvements(self, llm_response: str) -> List[ImprovementSuggestion]:
        """Parse LLM response into structured improvement suggestions."""
        suggestions = []
        
        # Simple parsing - look for numbered suggestions
        lines = llm_response.split('\n')
        current_suggestion = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for numbered items or clear section headers
            if (line[0].isdigit() and '.' in line[:3]) or line.startswith('**'):
                # Save previous suggestion if complete
                if current_suggestion.get('title'):
                    suggestions.append(self._create_suggestion_from_parsed(current_suggestion))
                    
                # Start new suggestion
                current_suggestion = {'title': line, 'description': ''}
                
            elif current_suggestion:
                # Add to description
                current_suggestion['description'] += line + ' '
                
        # Save last suggestion
        if current_suggestion.get('title'):
            suggestions.append(self._create_suggestion_from_parsed(current_suggestion))
            
        return suggestions[:5]  # Limit to 5 suggestions

    def _create_suggestion_from_parsed(self, parsed_data: Dict[str, str]) -> ImprovementSuggestion:
        """Create an ImprovementSuggestion from parsed LLM data."""
        improvement_id = f"llm_improvement_{int(time.time())}_{hash(parsed_data['title']) % 10000}"
        
        return ImprovementSuggestion(
            improvement_id=improvement_id,
            improvement_type=ImprovementType.CODE_GENERATION,  # Default type
            title=parsed_data['title'][:100],  # Truncate long titles
            description=parsed_data['description'][:500],  # Truncate long descriptions
            confidence=0.6,  # Medium confidence for LLM suggestions
            impact_score=0.5,  # Medium impact by default
            created_time=time.time(),
        )

    def _load_learning_data(self) -> None:
        """Load learning data from storage."""
        try:
            if self.storage_path.exists():
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    
                # Load learning records
                records_data = data.get("learning_records", [])
                self.learning_records = []
                for record_data in records_data:
                    # Convert error_category back to enum
                    if record_data.get("error_category"):
                        record_data["error_category"] = ErrorCategory(record_data["error_category"])
                    record = LearningRecord(**record_data)
                    self.learning_records.append(record)
                    
                # Load improvement suggestions
                suggestions_data = data.get("improvement_suggestions", {})
                self.improvement_suggestions = {}
                for suggestion_id, suggestion_data in suggestions_data.items():
                    # Convert enums back
                    if suggestion_data.get("improvement_type"):
                        suggestion_data["improvement_type"] = ImprovementType(suggestion_data["improvement_type"])
                    if suggestion_data.get("applicable_categories"):
                        suggestion_data["applicable_categories"] = [
                            ErrorCategory(cat) for cat in suggestion_data["applicable_categories"]
                        ]
                    suggestion = ImprovementSuggestion(**suggestion_data)
                    self.improvement_suggestions[suggestion_id] = suggestion
                    
                self.logger.info(f"Loaded {len(self.learning_records)} learning records and {len(self.improvement_suggestions)} improvements")
                
        except Exception as e:
            self.logger.error(f"Failed to load learning data: {e}")

    def _save_learning_data(self) -> None:
        """Save learning data to storage."""
        try:
            # Prepare data for JSON serialization
            records_data = []
            for record in self.learning_records:
                record_dict = record.__dict__.copy()
                # Convert enums to strings
                if record_dict.get("error_category"):
                    record_dict["error_category"] = record_dict["error_category"].value
                records_data.append(record_dict)
                
            suggestions_data = {}
            for suggestion_id, suggestion in self.improvement_suggestions.items():
                suggestion_dict = suggestion.__dict__.copy()
                # Convert enums to strings
                if suggestion_dict.get("improvement_type"):
                    suggestion_dict["improvement_type"] = suggestion_dict["improvement_type"].value
                if suggestion_dict.get("applicable_categories"):
                    suggestion_dict["applicable_categories"] = [
                        cat.value for cat in suggestion_dict["applicable_categories"]
                    ]
                suggestions_data[suggestion_id] = suggestion_dict
                
            data = {
                "learning_records": records_data,
                "improvement_suggestions": suggestions_data,
                "performance_metrics": dict(self.performance_metrics),
            }
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            self.logger.debug("Learning data saved to storage")
            
        except Exception as e:
            self.logger.error(f"Failed to save learning data: {e}")