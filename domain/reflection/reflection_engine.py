"""Core reflection engine for analyzing execution failures."""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from infrastructure.llm.client import LLMClient
from infrastructure.logging import get_logger
from domain.entities import ExecutionResults, ExecutionStatus
from .error_categorization import ErrorClassifier, CategorizedError
from .context_analysis import ContextAwareAnalyzer, ContextAnalysisResult

logger = get_logger(__name__)


@dataclass
class FailureContext:
    """Context information for failure analysis."""
    
    user_query: str
    generated_code: str
    execution_results: ExecutionResults
    data_schema: Dict[str, Any] = field(default_factory=dict)
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    previous_attempts: List[Dict[str, Any]] = field(default_factory=list)
    session_context: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class ReflectionResult:
    """Result of reflection analysis."""
    
    categorized_error: Optional[CategorizedError]
    root_cause_analysis: str
    improvement_suggestions: List[str]
    corrected_approach: Optional[str]
    confidence_score: float
    should_retry: bool
    retry_strategy: Optional[str] = None
    learning_insights: List[str] = field(default_factory=list)
    context_insights: Dict[str, Any] = field(default_factory=dict)
    context_analysis: Optional[ContextAnalysisResult] = None


class ReflectionEngine:
    """Core engine for analyzing execution failures and generating improvements."""
    
    def __init__(self, llm_client: LLMClient):
        """Initialize the reflection engine."""
        self.llm_client = llm_client
        self.error_classifier = ErrorClassifier()
        self.context_analyzer = ContextAwareAnalyzer()
        self.logger = logger.getChild("ReflectionEngine")
        
    def analyze_failure(self, failure_context: FailureContext) -> ReflectionResult:
        """
        Analyze a failure and provide comprehensive reflection.
        
        Args:
            failure_context: Complete context of the failure
            
        Returns:
            ReflectionResult with detailed analysis and suggestions
        """
        self.logger.info("Starting failure analysis")
        start_time = time.time()
        
        try:
            # Step 1: Classify the error
            categorized_error = None
            if failure_context.execution_results.status == ExecutionStatus.FAILED:
                # Create a mock exception from the error message for classification
                try:
                    # Try to recreate the exception type from the error message
                    error_msg = failure_context.execution_results.error_message or "Unknown error"
                    stderr = failure_context.execution_results.stderr
                    
                    # Extract exception type if available
                    exception_type = self._extract_exception_type(error_msg, stderr)
                    mock_exception = exception_type(error_msg)
                    
                    categorized_error = self.error_classifier.classify_error(
                        error=mock_exception,
                        code=failure_context.generated_code,
                        context={
                            "raw_dataset": failure_context.session_context.get("raw_dataset"),
                            "user_query": failure_context.user_query,
                        }
                    )
                except Exception as e:
                    self.logger.warning(f"Error classification failed: {e}")
            
            # Step 2: Perform context-aware analysis
            context_analysis = None
            if categorized_error:
                context_analysis = self.context_analyzer.analyze_context(
                    categorized_error=categorized_error,
                    query=failure_context.user_query,
                    code=failure_context.generated_code,
                    dataframe=failure_context.session_context.get("raw_dataset"),
                    additional_context=failure_context.session_context
                )
            
            # Step 3: Perform root cause analysis using LLM (enhanced with context)
            root_cause_analysis = self._perform_root_cause_analysis(failure_context, categorized_error, context_analysis)
            
            # Step 4: Generate improvement suggestions (enhanced with context)
            improvement_suggestions = self._generate_improvement_suggestions(
                failure_context, categorized_error, root_cause_analysis, context_analysis
            )
            
            # Step 5: Generate corrected approach
            corrected_approach = self._generate_corrected_approach(
                failure_context, categorized_error, improvement_suggestions
            )
            
            # Step 6: Assess retry strategy
            should_retry, retry_strategy = self._assess_retry_strategy(
                failure_context, categorized_error
            )
            
            # Step 7: Extract learning insights
            learning_insights = self._extract_learning_insights(
                failure_context, categorized_error, root_cause_analysis
            )
            
            # Step 8: Generate context insights
            context_insights = self._generate_context_insights(failure_context, categorized_error)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(categorized_error, failure_context)
            
            analysis_time = time.time() - start_time
            self.logger.info(f"Failure analysis completed in {analysis_time:.2f}s")
            
            return ReflectionResult(
                categorized_error=categorized_error,
                root_cause_analysis=root_cause_analysis,
                improvement_suggestions=improvement_suggestions,
                corrected_approach=corrected_approach,
                confidence_score=confidence_score,
                should_retry=should_retry,
                retry_strategy=retry_strategy,
                learning_insights=learning_insights,
                context_insights=context_insights,
                context_analysis=context_analysis,
            )
            
        except Exception as e:
            self.logger.error(f"Reflection analysis failed: {e}", exc_info=True)
            
            # Return a basic reflection result
            return ReflectionResult(
                categorized_error=None,
                root_cause_analysis=f"Reflection analysis failed: {str(e)}",
                improvement_suggestions=["Manual review required due to reflection system error"],
                corrected_approach=None,
                confidence_score=0.1,
                should_retry=False,
                learning_insights=[f"Reflection system encountered error: {str(e)}"],
                context_insights={},
            )

    def _extract_exception_type(self, error_msg: str, stderr: str) -> type:
        """Extract the exception type from error message or stderr."""
        combined_text = f"{error_msg} {stderr}"
        
        # Common exception types
        exception_mapping = {
            "SyntaxError": SyntaxError,
            "NameError": NameError,
            "TypeError": TypeError,
            "AttributeError": AttributeError,
            "KeyError": KeyError,
            "ValueError": ValueError,
            "IndexError": IndexError,
            "ImportError": ImportError,
            "ModuleNotFoundError": ModuleNotFoundError,
            "ZeroDivisionError": ZeroDivisionError,
        }
        
        for exc_name, exc_type in exception_mapping.items():
            if exc_name in combined_text:
                return exc_type
                
        return Exception  # Default fallback

    def _perform_root_cause_analysis(
        self, 
        failure_context: FailureContext, 
        categorized_error: Optional[CategorizedError],
        context_analysis: Optional[ContextAnalysisResult] = None
    ) -> str:
        """Perform LLM-powered root cause analysis."""
        
        prompt = self._build_root_cause_prompt(failure_context, categorized_error, context_analysis)
        
        try:
            response = self.llm_client.generate_text(
                prompt=prompt,
                max_tokens=800,
                temperature=0.3,  # Lower temperature for more focused analysis
            )
            
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"LLM root cause analysis failed: {e}")
            
            # Fallback analysis
            if categorized_error:
                return f"Error classified as {categorized_error.analysis.category.value}: {categorized_error.analysis.primary_cause}"
            else:
                return f"Execution failed: {failure_context.execution_results.error_message}"

    def _build_root_cause_prompt(
        self, 
        failure_context: FailureContext, 
        categorized_error: Optional[CategorizedError],
        context_analysis: Optional[ContextAnalysisResult] = None
    ) -> str:
        """Build a comprehensive prompt for root cause analysis."""
        
        prompt_parts = [
            "You are an expert data analyst debugging a code execution failure.",
            "Analyze the following failure and provide a detailed root cause analysis.",
            "",
            "## User Query:",
            failure_context.user_query,
            "",
            "## Generated Code:",
            "```python",
            failure_context.generated_code,
            "```",
            "",
            "## Execution Results:",
            f"Status: {failure_context.execution_results.status.value}",
            f"Error: {failure_context.execution_results.error_message or 'None'}",
            f"Execution Time: {failure_context.execution_results.execution_time:.2f}s",
        ]
        
        if failure_context.execution_results.stderr:
            prompt_parts.extend([
                "",
                "## Error Details:",
                failure_context.execution_results.stderr,
            ])
            
        if categorized_error:
            prompt_parts.extend([
                "",
                "## Error Classification:",
                f"Category: {categorized_error.analysis.category.value}",
                f"Primary Cause: {categorized_error.analysis.primary_cause}",
                f"Confidence: {categorized_error.analysis.confidence:.2f}",
                f"Contributing Factors: {', '.join(categorized_error.analysis.contributing_factors)}",
            ])
            
        if failure_context.data_schema:
            prompt_parts.extend([
                "",
                "## Data Schema:",
                json.dumps(failure_context.data_schema, indent=2),
            ])
            
        if failure_context.previous_attempts:
            prompt_parts.extend([
                "",
                "## Previous Attempts:",
                f"Number of previous failures: {len(failure_context.previous_attempts)}",
            ])
            
        if context_analysis:
            prompt_parts.extend([
                "",
                "## Context Analysis:",
                f"Query Complexity: {context_analysis.query_complexity.value}",
                f"Primary Intent: {context_analysis.intent_analysis.get('primary_intent', 'unknown')}",
                f"Data Characteristics: {[char.value for char in context_analysis.data_characteristics]}",
                f"Key Insights: {len(context_analysis.contextual_insights)} contextual insights identified",
            ])
            
        prompt_parts.extend([
            "",
            "## Analysis Request:",
            "Provide a concise but thorough root cause analysis explaining:",
            "1. What exactly went wrong and why",
            "2. How the error relates to the user's intent",
            "3. What conditions led to this failure",
            "4. Whether this is a common or unusual failure pattern",
            "",
            "Focus on actionable insights that can prevent similar failures.",
        ])
        
        return "\n".join(prompt_parts)

    def _generate_improvement_suggestions(
        self,
        failure_context: FailureContext,
        categorized_error: Optional[CategorizedError], 
        root_cause_analysis: str,
        context_analysis: Optional[ContextAnalysisResult] = None
    ) -> List[str]:
        """Generate specific improvement suggestions."""
        
        suggestions = []
        
        # Add suggestions from error classification
        if categorized_error:
            suggestions.extend(categorized_error.analysis.suggested_fixes)
            
        # Generate LLM-powered suggestions
        llm_suggestions = self._generate_llm_suggestions(
            failure_context, categorized_error, root_cause_analysis
        )
        suggestions.extend(llm_suggestions)
        
        # Add context-specific suggestions
        context_suggestions = self._generate_context_suggestions(failure_context)
        suggestions.extend(context_suggestions)
        
        # Add suggestions from context analysis
        if context_analysis:
            for insight in context_analysis.contextual_insights:
                suggestions.append(insight.recommendation)
            suggestions.extend(context_analysis.suggested_alternatives)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_suggestions = []
        for suggestion in suggestions:
            if suggestion not in seen:
                seen.add(suggestion)
                unique_suggestions.append(suggestion)
                
        return unique_suggestions[:10]  # Limit to top 10 suggestions

    def _generate_llm_suggestions(
        self,
        failure_context: FailureContext,
        categorized_error: Optional[CategorizedError],
        root_cause_analysis: str
    ) -> List[str]:
        """Generate improvement suggestions using LLM."""
        
        prompt = f"""Based on this failure analysis, provide 3-5 specific, actionable suggestions for improvement:

Root Cause: {root_cause_analysis}

User Query: {failure_context.user_query}

Failed Code:
```python
{failure_context.generated_code}
```

Error: {failure_context.execution_results.error_message}

Provide suggestions as a numbered list, focusing on:
1. Code corrections
2. Alternative approaches
3. Data handling improvements
4. Best practices

Be specific and actionable."""

        try:
            response = self.llm_client.generate_text(
                prompt=prompt,
                max_tokens=600,
                temperature=0.4,
            )
            
            # Parse numbered list from response
            import re
            suggestions = []
            for line in response.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                    # Remove numbering/bullets and add to suggestions
                    clean_suggestion = re.sub(r'^[\d\-•.\s]+', '', line).strip()
                    if clean_suggestion:
                        suggestions.append(clean_suggestion)
                        
            return suggestions[:5]  # Limit to 5 suggestions
            
        except Exception as e:
            self.logger.error(f"LLM suggestion generation failed: {e}")
            return []

    def _generate_context_suggestions(self, failure_context: FailureContext) -> List[str]:
        """Generate suggestions based on context analysis."""
        suggestions = []
        
        # Analyze query complexity
        query_words = len(failure_context.user_query.split())
        if query_words > 20:
            suggestions.append("Consider breaking down complex queries into simpler steps")
            
        # Check for repeated failures
        if len(failure_context.previous_attempts) > 2:
            suggestions.append("Multiple failures detected - consider alternative analysis approach")
            
        # Check code length
        code_lines = len(failure_context.generated_code.split('\n'))
        if code_lines > 30:
            suggestions.append("Generated code is complex - consider simplifying the approach")
            
        return suggestions

    def _generate_corrected_approach(
        self,
        failure_context: FailureContext,
        categorized_error: Optional[CategorizedError],
        improvement_suggestions: List[str]
    ) -> Optional[str]:
        """Generate a corrected approach using LLM."""
        
        if not categorized_error or not categorized_error.analysis.is_recoverable:
            return None
            
        prompt = f"""Generate a corrected approach for this failed data analysis:

Original Query: {failure_context.user_query}

Failed Code:
```python
{failure_context.generated_code}
```

Error: {failure_context.execution_results.error_message}

Improvement Suggestions:
{chr(10).join(f"- {suggestion}" for suggestion in improvement_suggestions[:5])}

Provide a brief description of a corrected approach (not full code) that addresses the root cause.
Focus on the key changes needed in the analysis strategy."""

        try:
            response = self.llm_client.generate_text(
                prompt=prompt,
                max_tokens=400,
                temperature=0.3,
            )
            
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"Corrected approach generation failed: {e}")
            return None

    def _assess_retry_strategy(
        self,
        failure_context: FailureContext,
        categorized_error: Optional[CategorizedError]
    ) -> Tuple[bool, Optional[str]]:
        """Assess whether retry is recommended and with what strategy."""
        
        # Don't retry if too many previous attempts
        if len(failure_context.previous_attempts) >= 3:
            return False, "Maximum retry attempts reached"
            
        # Don't retry security errors
        if categorized_error and categorized_error.analysis.category.name == "SECURITY":
            return False, "Security errors require manual intervention"
            
        # Don't retry if not recoverable
        if categorized_error and not categorized_error.analysis.is_recoverable:
            return False, "Error assessed as non-recoverable"
            
        # Assess retry strategy based on error type
        if categorized_error:
            category = categorized_error.analysis.category.name
            
            if category == "SYNTAX":
                return True, "Retry with corrected syntax"
            elif category == "DATA":
                return True, "Retry with improved data handling"
            elif category == "RUNTIME":
                return True, "Retry with fixed runtime issues"
            elif category == "PERFORMANCE":
                return True, "Retry with optimized approach"
            elif category == "DEPENDENCY":
                return False, "Dependency issues require manual resolution"
                
        # Default: retry with general improvements
        return True, "Retry with general improvements"

    def _extract_learning_insights(
        self,
        failure_context: FailureContext,
        categorized_error: Optional[CategorizedError],
        root_cause_analysis: str
    ) -> List[str]:
        """Extract insights for learning system."""
        insights = []
        
        # Add insights from error category
        if categorized_error:
            category = categorized_error.analysis.category.value
            insights.append(f"Common {category} error pattern identified")
            
            if categorized_error.analysis.contributing_factors:
                insights.append(f"Contributing factors: {', '.join(categorized_error.analysis.contributing_factors)}")
                
        # Add query-specific insights
        query_lower = failure_context.user_query.lower()
        if "visualization" in query_lower or "plot" in query_lower or "chart" in query_lower:
            insights.append("Visualization-related query failure")
        elif "correlation" in query_lower or "relationship" in query_lower:
            insights.append("Statistical analysis query failure")
        elif "prediction" in query_lower or "forecast" in query_lower:
            insights.append("Predictive analysis query failure")
            
        # Add code pattern insights
        code_lower = failure_context.generated_code.lower()
        if "groupby" in code_lower:
            insights.append("GroupBy operation involved in failure")
        if "merge" in code_lower or "join" in code_lower:
            insights.append("Data joining operation involved in failure")
            
        return insights

    def _generate_context_insights(
        self, 
        failure_context: FailureContext, 
        categorized_error: Optional[CategorizedError]
    ) -> Dict[str, Any]:
        """Generate insights about the failure context."""
        insights = {}
        
        # Query analysis
        insights["query_length"] = len(failure_context.user_query)
        insights["query_complexity"] = len(failure_context.user_query.split())
        
        # Code analysis
        insights["code_length"] = len(failure_context.generated_code)
        insights["code_lines"] = len(failure_context.generated_code.split('\n'))
        
        # Error analysis
        if categorized_error:
            insights["error_category"] = categorized_error.analysis.category.value
            insights["error_severity"] = categorized_error.analysis.severity
            insights["error_confidence"] = categorized_error.analysis.confidence
            
        # Execution analysis
        insights["execution_time"] = failure_context.execution_results.execution_time
        insights["has_stderr"] = bool(failure_context.execution_results.stderr)
        
        # Context analysis
        insights["has_previous_attempts"] = len(failure_context.previous_attempts) > 0
        insights["attempt_count"] = len(failure_context.previous_attempts) + 1
        
        return insights

    def _calculate_confidence_score(
        self, 
        categorized_error: Optional[CategorizedError], 
        failure_context: FailureContext
    ) -> float:
        """Calculate confidence score for the reflection analysis."""
        base_confidence = 0.5
        
        # Boost confidence if we have good error classification
        if categorized_error:
            base_confidence += categorized_error.analysis.confidence * 0.3
            
        # Boost confidence if we have good context
        if failure_context.data_schema:
            base_confidence += 0.1
            
        # Reduce confidence for complex queries
        if len(failure_context.user_query.split()) > 20:
            base_confidence -= 0.1
            
        # Reduce confidence for repeated failures
        if len(failure_context.previous_attempts) > 2:
            base_confidence -= 0.2
            
        return max(0.1, min(1.0, base_confidence))