"""Context-aware error analysis that understands data and query intent."""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

import pandas as pd
from infrastructure.logging import get_logger
from .error_categorization import CategorizedError, ErrorCategory

logger = get_logger(__name__)


class QueryComplexity(Enum):
    """Levels of query complexity."""
    
    SIMPLE = "simple"         # Basic operations (filter, select)
    MODERATE = "moderate"     # Aggregations, grouping
    COMPLEX = "complex"       # Multiple operations, joins
    ADVANCED = "advanced"     # ML, statistical analysis


class DataCharacteristics(Enum):
    """Characteristics of the dataset."""
    
    SMALL = "small"           # < 1000 rows
    MEDIUM = "medium"         # 1000-100k rows
    LARGE = "large"           # 100k-1M rows
    VERY_LARGE = "very_large" # > 1M rows
    WIDE = "wide"             # Many columns (>50)
    SPARSE = "sparse"         # Many null values
    MIXED_TYPES = "mixed_types" # Mixed data types


@dataclass
class ContextualInsight:
    """Represents a contextual insight about the failure."""
    
    insight_type: str
    description: str
    confidence: float
    impact: str  # "low", "medium", "high"
    recommendation: str
    supporting_evidence: List[str] = field(default_factory=list)


@dataclass
class ContextAnalysisResult:
    """Result of context-aware analysis."""
    
    query_complexity: QueryComplexity
    data_characteristics: List[DataCharacteristics]
    intent_analysis: Dict[str, Any]
    contextual_insights: List[ContextualInsight]
    mismatch_analysis: Dict[str, Any]
    suggested_alternatives: List[str]


class ContextAwareAnalyzer:
    """Analyzes failures in the context of data characteristics and query intent."""
    
    def __init__(self):
        """Initialize the context-aware analyzer."""
        self.logger = logger.getChild("ContextAwareAnalyzer")
        
        # Query patterns for intent classification
        self.intent_patterns = {
            "exploration": [
                r"show.*data", r"display.*data", r"what.*data", r"describe.*data",
                r"overview", r"summary", r"head", r"sample", r"first.*rows"
            ],
            "filtering": [
                r"where", r"filter", r"select.*where", r"only.*show", r"exclude",
                r"rows.*that", r"records.*that", r"data.*where"
            ],
            "aggregation": [
                r"sum", r"count", r"average", r"mean", r"total", r"group.*by",
                r"aggregate", r"statistics", r"max", r"min", r"median"
            ],
            "comparison": [
                r"compare", r"difference", r"vs", r"versus", r"between",
                r"correlation", r"relationship", r"against"
            ],
            "visualization": [
                r"plot", r"chart", r"graph", r"visualize", r"show.*plot",
                r"histogram", r"scatter", r"bar.*chart", r"line.*chart"
            ],
            "prediction": [
                r"predict", r"forecast", r"model", r"machine.*learning",
                r"regression", r"classification", r"future", r"trend"
            ],
            "transformation": [
                r"transform", r"convert", r"change", r"modify", r"create.*column",
                r"derive", r"calculate.*new", r"add.*column"
            ]
        }

    def analyze_context(
        self, 
        categorized_error: CategorizedError,
        query: str,
        code: str,
        dataframe: Optional[pd.DataFrame] = None,
        additional_context: Dict[str, Any] = None
    ) -> ContextAnalysisResult:
        """
        Perform context-aware analysis of the failure.
        
        Args:
            categorized_error: The categorized error
            query: User query
            code: Failed code
            dataframe: The dataset being analyzed
            additional_context: Additional context information
            
        Returns:
            ContextAnalysisResult with comprehensive analysis
        """
        additional_context = additional_context or {}
        
        self.logger.info("Starting context-aware failure analysis")
        
        # Analyze query complexity and intent
        query_complexity = self._analyze_query_complexity(query)
        intent_analysis = self._analyze_query_intent(query)
        
        # Analyze data characteristics
        data_characteristics = self._analyze_data_characteristics(dataframe)
        
        # Generate contextual insights
        contextual_insights = self._generate_contextual_insights(
            categorized_error, query, code, dataframe, intent_analysis, query_complexity
        )
        
        # Analyze mismatches between intent and implementation
        mismatch_analysis = self._analyze_intent_implementation_mismatch(
            query, code, intent_analysis, categorized_error
        )
        
        # Generate alternative approaches
        suggested_alternatives = self._suggest_alternative_approaches(
            query, intent_analysis, categorized_error, data_characteristics
        )
        
        result = ContextAnalysisResult(
            query_complexity=query_complexity,
            data_characteristics=data_characteristics,
            intent_analysis=intent_analysis,
            contextual_insights=contextual_insights,
            mismatch_analysis=mismatch_analysis,
            suggested_alternatives=suggested_alternatives,
        )
        
        self.logger.info(f"Context analysis completed with {len(contextual_insights)} insights")
        return result

    def _analyze_query_complexity(self, query: str) -> QueryComplexity:
        """Analyze the complexity level of the query."""
        query_lower = query.lower()
        
        # Count complexity indicators
        complexity_indicators = {
            "advanced": [
                r"machine.*learning", r"regression", r"classification", r"clustering",
                r"neural.*network", r"deep.*learning", r"ai", r"model", r"predict"
            ],
            "complex": [
                r"join", r"merge", r"correlation", r"statistical.*test",
                r"pivot", r"reshape", r"multiple.*group", r"nested"
            ],
            "moderate": [
                r"group.*by", r"aggregate", r"sum", r"count", r"average",
                r"sort", r"order.*by", r"having"
            ],
            "simple": [
                r"select", r"where", r"filter", r"show", r"display", r"first"
            ]
        }
        
        # Check for advanced patterns first
        for level in ["advanced", "complex", "moderate", "simple"]:
            for pattern in complexity_indicators[level]:
                if re.search(pattern, query_lower):
                    return QueryComplexity(level)
                    
        # Default based on query length and word count
        if len(query.split()) > 20:
            return QueryComplexity.COMPLEX
        elif len(query.split()) > 10:
            return QueryComplexity.MODERATE
        else:
            return QueryComplexity.SIMPLE

    def _analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze the intent behind the query."""
        query_lower = query.lower()
        intent_scores = {}
        
        # Calculate scores for each intent
        for intent, patterns in self.intent_patterns.items():
            score = 0
            matched_patterns = []
            
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 1
                    matched_patterns.append(pattern)
                    
            if score > 0:
                intent_scores[intent] = {
                    "score": score,
                    "confidence": min(1.0, score / len(patterns)),
                    "matched_patterns": matched_patterns
                }
        
        # Determine primary intent
        primary_intent = None
        if intent_scores:
            primary_intent = max(intent_scores.keys(), key=lambda x: intent_scores[x]["score"])
            
        return {
            "primary_intent": primary_intent,
            "intent_scores": intent_scores,
            "ambiguous": len([i for i, s in intent_scores.items() if s["score"] > 0]) > 2
        }

    def _analyze_data_characteristics(self, dataframe: Optional[pd.DataFrame]) -> List[DataCharacteristics]:
        """Analyze characteristics of the dataset."""
        if dataframe is None:
            return []
            
        characteristics = []
        
        # Size characteristics
        rows, cols = dataframe.shape
        
        if rows < 1000:
            characteristics.append(DataCharacteristics.SMALL)
        elif rows < 100000:
            characteristics.append(DataCharacteristics.MEDIUM)
        elif rows < 1000000:
            characteristics.append(DataCharacteristics.LARGE)
        else:
            characteristics.append(DataCharacteristics.VERY_LARGE)
            
        # Width characteristics
        if cols > 50:
            characteristics.append(DataCharacteristics.WIDE)
            
        # Sparsity
        null_percentage = dataframe.isnull().sum().sum() / (rows * cols)
        if null_percentage > 0.3:
            characteristics.append(DataCharacteristics.SPARSE)
            
        # Mixed types
        type_counts = dataframe.dtypes.value_counts()
        if len(type_counts) > 3:
            characteristics.append(DataCharacteristics.MIXED_TYPES)
            
        return characteristics

    def _generate_contextual_insights(
        self,
        categorized_error: CategorizedError,
        query: str,
        code: str,
        dataframe: Optional[pd.DataFrame],
        intent_analysis: Dict[str, Any],
        query_complexity: QueryComplexity
    ) -> List[ContextualInsight]:
        """Generate contextual insights based on the analysis."""
        insights = []
        
        # Intent-Error mismatch insights
        if intent_analysis["primary_intent"] == "visualization" and categorized_error.analysis.category == ErrorCategory.DEPENDENCY:
            insights.append(ContextualInsight(
                insight_type="intent_mismatch",
                description="Visualization intent detected but missing plotting library",
                confidence=0.9,
                impact="high",
                recommendation="Import matplotlib or seaborn for plotting",
                supporting_evidence=["Primary intent: visualization", "Error category: dependency"]
            ))
            
        # Complexity-Error insights
        if query_complexity in [QueryComplexity.COMPLEX, QueryComplexity.ADVANCED] and categorized_error.analysis.category == ErrorCategory.SYNTAX:
            insights.append(ContextualInsight(
                insight_type="complexity_error",
                description="Complex query resulted in syntax error - may indicate code generation limits",
                confidence=0.8,
                impact="medium",
                recommendation="Break down complex query into simpler steps",
                supporting_evidence=[f"Query complexity: {query_complexity.value}", "Syntax error detected"]
            ))
            
        # Data-Error insights
        if dataframe is not None:
            if categorized_error.analysis.category == ErrorCategory.DATA:
                # Column-related errors
                if "column" in categorized_error.error_message.lower():
                    available_columns = list(dataframe.columns) if hasattr(dataframe, 'columns') else []
                    insights.append(ContextualInsight(
                        insight_type="data_schema",
                        description=f"Column reference error - dataset has {len(available_columns)} columns",
                        confidence=0.95,
                        impact="high",
                        recommendation=f"Verify column names: {available_columns[:5]}{'...' if len(available_columns) > 5 else ''}",
                        supporting_evidence=[f"Available columns: {len(available_columns)}", "Column reference in error"]
                    ))
                    
                # Size-related insights
                rows, cols = dataframe.shape
                if rows == 0:
                    insights.append(ContextualInsight(
                        insight_type="data_quality",
                        description="Dataset is empty - no rows to analyze",
                        confidence=1.0,
                        impact="critical",
                        recommendation="Check data loading and filtering logic",
                        supporting_evidence=["Dataset shape: (0, N)"]
                    ))
                    
        # Performance insights
        if categorized_error.analysis.category == ErrorCategory.PERFORMANCE:
            if dataframe is not None and len(dataframe) > 100000:
                insights.append(ContextualInsight(
                    insight_type="performance_scale",
                    description="Performance issue with large dataset",
                    confidence=0.8,
                    impact="medium",
                    recommendation="Consider data sampling or chunked processing",
                    supporting_evidence=[f"Dataset size: {len(dataframe)} rows", "Performance error category"]
                ))
                
        return insights

    def _analyze_intent_implementation_mismatch(
        self,
        query: str,
        code: str,
        intent_analysis: Dict[str, Any],
        categorized_error: CategorizedError
    ) -> Dict[str, Any]:
        """Analyze mismatches between query intent and code implementation."""
        primary_intent = intent_analysis.get("primary_intent")
        
        mismatches = []
        
        if primary_intent == "visualization":
            # Check if plotting code was generated
            if not any(plot_lib in code.lower() for plot_lib in ["plt.", "matplotlib", "seaborn", ".plot("]):
                mismatches.append({
                    "type": "missing_visualization",
                    "description": "Query requests visualization but no plotting code generated",
                    "severity": "high"
                })
                
        elif primary_intent == "aggregation":
            # Check if aggregation functions are used
            if not any(agg_func in code.lower() for agg_func in ["groupby", "sum(", "count(", "mean(", "aggregate"]):
                mismatches.append({
                    "type": "missing_aggregation",
                    "description": "Query requests aggregation but no aggregation functions found",
                    "severity": "medium"
                })
                
        elif primary_intent == "filtering":
            # Check if filtering is implemented
            if not any(filter_op in code.lower() for filter_op in ["[", "query(", "where", "loc[", "iloc["]):
                mismatches.append({
                    "type": "missing_filtering",
                    "description": "Query requests filtering but no filtering operations found",
                    "severity": "medium"
                })
                
        return {
            "mismatches": mismatches,
            "mismatch_count": len(mismatches),
            "overall_alignment": "poor" if len(mismatches) > 1 else "good" if len(mismatches) == 0 else "fair"
        }

    def _suggest_alternative_approaches(
        self,
        query: str,
        intent_analysis: Dict[str, Any],
        categorized_error: CategorizedError,
        data_characteristics: List[DataCharacteristics]
    ) -> List[str]:
        """Suggest alternative approaches based on context."""
        alternatives = []
        
        primary_intent = intent_analysis.get("primary_intent")
        error_category = categorized_error.analysis.category
        
        # Intent-based alternatives
        if primary_intent == "visualization":
            if error_category == ErrorCategory.DEPENDENCY:
                alternatives.extend([
                    "Use built-in pandas plotting: df.plot()",
                    "Try seaborn for statistical plots",
                    "Consider plotly for interactive visualizations"
                ])
            elif error_category == ErrorCategory.DATA:
                alternatives.append("Ensure data is properly formatted for plotting")
                
        elif primary_intent == "aggregation":
            if error_category == ErrorCategory.RUNTIME:
                alternatives.extend([
                    "Use df.groupby() for group-wise operations",
                    "Try df.agg() for multiple aggregations",
                    "Consider pivot tables for complex aggregations"
                ])
                
        # Data characteristic-based alternatives
        if DataCharacteristics.LARGE in data_characteristics:
            alternatives.extend([
                "Sample data for faster processing: df.sample(n=10000)",
                "Use chunked processing for large datasets",
                "Consider data reduction techniques"
            ])
            
        if DataCharacteristics.SPARSE in data_characteristics:
            alternatives.extend([
                "Handle missing values: df.dropna() or df.fillna()",
                "Consider imputation strategies for sparse data"
            ])
            
        # Error category-based alternatives
        if error_category == ErrorCategory.SYNTAX:
            alternatives.extend([
                "Simplify the analysis approach",
                "Break complex operations into steps",
                "Verify variable names and syntax"
            ])
        elif error_category == ErrorCategory.DATA:
            alternatives.extend([
                "Check column names and data types",
                "Validate data structure before operations",
                "Handle edge cases in data"
            ])
            
        return alternatives[:8]  # Limit to 8 alternatives