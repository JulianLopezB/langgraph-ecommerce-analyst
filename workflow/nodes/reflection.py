"""Enhanced reflection node for LangGraph workflow."""

import time
from typing import Dict, Any, List

from infrastructure.logging import get_logger
from infrastructure.llm.client import get_llm_client
from domain.entities import ExecutionStatus
from domain.reflection import (
    ReflectionEngine,
    PatternDetector, 
    LearningSystem,
    EmbeddingPatternMatcher,
    FailureContext,
)
from workflow.state import AnalysisState

logger = get_logger(__name__)

# Global instances (initialized lazily)
_reflection_engine = None
_pattern_detector = None
_learning_system = None
_embedding_matcher = None


def get_reflection_engine() -> ReflectionEngine:
    """Get or create the reflection engine instance."""
    global _reflection_engine
    if _reflection_engine is None:
        llm_client = get_llm_client()
        _reflection_engine = ReflectionEngine(llm_client)
    return _reflection_engine


def get_pattern_detector() -> PatternDetector:
    """Get or create the pattern detector instance."""
    global _pattern_detector
    if _pattern_detector is None:
        _pattern_detector = PatternDetector()
    return _pattern_detector


def get_learning_system() -> LearningSystem:
    """Get or create the learning system instance."""
    global _learning_system
    if _learning_system is None:
        llm_client = get_llm_client()
        _learning_system = LearningSystem(llm_client)
    return _learning_system


def get_embedding_matcher() -> EmbeddingPatternMatcher:
    """Get or create the embedding matcher instance."""
    global _embedding_matcher
    if _embedding_matcher is None:
        llm_client = get_llm_client()
        _embedding_matcher = EmbeddingPatternMatcher(llm_client)
    return _embedding_matcher


def reflect_on_failure(state: AnalysisState) -> AnalysisState:
    """
    Enhanced reflection node that analyzes failures and learns from them.
    
    This node implements the LangGraph reflection pattern with:
    - Comprehensive error categorization
    - Pattern detection and matching
    - Context-aware analysis
    - Learning and improvement suggestions
    - Embeddings-based similarity matching
    """
    logger.info("Starting enhanced failure reflection")
    
    try:
        # Check if we have a failure to reflect on
        execution_results = state.get("execution_results")
        if not execution_results or execution_results.status != ExecutionStatus.FAILED:
            logger.info("No failure to reflect on, skipping reflection")
            state["reflection_complete"] = True
            return state
            
        # Get reflection components
        reflection_engine = get_reflection_engine()
        pattern_detector = get_pattern_detector()
        learning_system = get_learning_system()
        embedding_matcher = get_embedding_matcher()
        
        # Build failure context
        failure_context = FailureContext(
            user_query=state["user_query"],
            generated_code=state.get("generated_code", {}).get("code_content", "") if state.get("generated_code") else "",
            execution_results=execution_results,
            data_schema=state.get("data_schema", {}),
            conversation_history=state.get("conversation_history", []),
            previous_attempts=state.get("error_context", {}).get("previous_attempts", []),
            session_context={
                "raw_dataset": state.get("raw_dataset"),
                "session_id": state["session_id"],
            }
        )
        
        # Step 1: Perform comprehensive reflection analysis
        logger.info("Performing comprehensive failure analysis")
        reflection_result = reflection_engine.analyze_failure(failure_context)
        
        # Step 2: Detect and match patterns
        logger.info("Detecting failure patterns")
        pattern_matches = []
        if reflection_result.categorized_error:
            # Traditional pattern matching
            traditional_matches = pattern_detector.detect_patterns(
                reflection_result.categorized_error,
                failure_context.user_query,
                failure_context.generated_code,
                failure_context.session_context
            )
            
            # Enhance with embeddings
            failure_text = _build_failure_description(failure_context, reflection_result)
            pattern_matches = embedding_matcher.enhance_pattern_matching(
                failure_text, 
                [(match.pattern, match.similarity_score) for match in traditional_matches]
            )
            
        # Step 3: Record learning event
        logger.info("Recording failure for learning")
        if reflection_result.categorized_error:
            learning_system.record_failure_event(
                reflection_result.categorized_error,
                failure_context.user_query,
                failure_context.generated_code,
                [match for match in traditional_matches] if 'traditional_matches' in locals() else [],
                failure_context.session_context
            )
            
        # Step 4: Learn from the failure
        if reflection_result.categorized_error:
            pattern_detector.learn_from_failure(
                reflection_result.categorized_error,
                failure_context.user_query,
                failure_context.generated_code,
                resolution_success=False,  # This is a failure event
                context=failure_context.session_context
            )
            
        # Step 5: Generate comprehensive reflection output
        reflection_output = _build_reflection_output(
            reflection_result, 
            pattern_matches, 
            failure_context
        )
        
        # Step 6: Update state with reflection results
        state["reflection_result"] = reflection_output
        state["reflection_complete"] = True
        
        # Determine next action based on reflection
        if reflection_result.should_retry and len(failure_context.previous_attempts) < 3:
            state["next_step"] = "retry_with_reflection"
            state["retry_strategy"] = reflection_result.retry_strategy
            
            # Add corrected approach to context
            if reflection_result.corrected_approach:
                if "error_context" not in state:
                    state["error_context"] = {}
                state["error_context"]["corrected_approach"] = reflection_result.corrected_approach
                state["error_context"]["reflection_suggestions"] = reflection_result.improvement_suggestions
                
        else:
            state["next_step"] = "handle_error"
            
        # Add reflection insights to conversation
        _add_reflection_to_conversation(state, reflection_output)
        
        logger.info("Failure reflection completed successfully")
        
    except Exception as e:
        logger.error(f"Reflection analysis failed: {e}", exc_info=True)
        
        # Fallback: basic error handling
        state["reflection_result"] = {
            "error": f"Reflection system error: {str(e)}",
            "fallback": True
        }
        state["reflection_complete"] = True
        state["next_step"] = "handle_error"
        
    return state


def reflect_on_success(state: AnalysisState) -> AnalysisState:
    """
    Reflect on successful executions to learn positive patterns.
    
    This function is called when code execution succeeds to:
    - Record successful patterns
    - Update learning system with positive examples
    - Validate previous improvement suggestions
    """
    logger.info("Reflecting on successful execution")
    
    try:
        execution_results = state.get("execution_results")
        if not execution_results or execution_results.status != ExecutionStatus.SUCCESS:
            return state
            
        learning_system = get_learning_system()
        
        # Record success event
        learning_system.record_success_event(
            query=state["user_query"],
            code=state.get("generated_code", {}).get("code_content", "") if state.get("generated_code") else "",
            execution_time=execution_results.execution_time,
            resolution_method="successful_generation",
            context={
                "raw_dataset": state.get("raw_dataset"),
                "session_id": state["session_id"],
            }
        )
        
        # Check if this success validates any previous improvement suggestions
        error_context = state.get("error_context", {})
        if "reflection_suggestions" in error_context:
            # This success came after applying reflection suggestions
            # TODO: Implement validation of specific suggestions
            logger.info("Success after reflection suggestions - potential validation opportunity")
            
        state["success_reflection_complete"] = True
        logger.info("Success reflection completed")
        
    except Exception as e:
        logger.error(f"Success reflection failed: {e}", exc_info=True)
        
    return state


def get_reflection_insights(state: AnalysisState) -> Dict[str, Any]:
    """
    Get insights from the reflection system for debugging/monitoring.
    
    Args:
        state: Current analysis state
        
    Returns:
        Dictionary with reflection system insights
    """
    try:
        pattern_detector = get_pattern_detector()
        learning_system = get_learning_system()
        embedding_matcher = get_embedding_matcher()
        
        insights = {
            "pattern_statistics": pattern_detector.get_pattern_statistics(),
            "learning_insights": learning_system.get_learning_insights(),
            "embedding_cache_stats": embedding_matcher.get_cache_stats(),
            "reflection_system_status": "operational"
        }
        
        return insights
        
    except Exception as e:
        logger.error(f"Failed to get reflection insights: {e}")
        return {
            "reflection_system_status": "error",
            "error": str(e)
        }


def _build_failure_description(failure_context: FailureContext, reflection_result) -> str:
    """Build a comprehensive text description of the failure for embeddings."""
    parts = [
        f"Query: {failure_context.user_query}",
        f"Error: {failure_context.execution_results.error_message}",
    ]
    
    if reflection_result.categorized_error:
        parts.extend([
            f"Category: {reflection_result.categorized_error.analysis.category.value}",
            f"Primary Cause: {reflection_result.categorized_error.analysis.primary_cause}",
        ])
        
    if reflection_result.root_cause_analysis:
        parts.append(f"Root Cause: {reflection_result.root_cause_analysis}")
        
    return " | ".join(parts)


def _build_reflection_output(
    reflection_result, 
    pattern_matches: List, 
    failure_context: FailureContext
) -> Dict[str, Any]:
    """Build comprehensive reflection output."""
    output = {
        "timestamp": time.time(),
        "confidence_score": reflection_result.confidence_score,
        "should_retry": reflection_result.should_retry,
        "retry_strategy": reflection_result.retry_strategy,
    }
    
    # Add error analysis
    if reflection_result.categorized_error:
        output["error_analysis"] = {
            "category": reflection_result.categorized_error.analysis.category.value,
            "severity": reflection_result.categorized_error.analysis.severity,
            "confidence": reflection_result.categorized_error.analysis.confidence,
            "primary_cause": reflection_result.categorized_error.analysis.primary_cause,
            "contributing_factors": reflection_result.categorized_error.analysis.contributing_factors,
            "is_recoverable": reflection_result.categorized_error.analysis.is_recoverable,
        }
        
    # Add root cause analysis
    output["root_cause_analysis"] = reflection_result.root_cause_analysis
    
    # Add improvement suggestions
    output["improvement_suggestions"] = reflection_result.improvement_suggestions
    
    # Add corrected approach
    if reflection_result.corrected_approach:
        output["corrected_approach"] = reflection_result.corrected_approach
        
    # Add pattern matches
    if pattern_matches:
        output["pattern_matches"] = [
            {
                "pattern_name": pattern.name,
                "similarity_score": score,
                "pattern_description": pattern.description,
                "common_fixes": pattern.common_fixes,
            }
            for pattern, score in pattern_matches[:3]  # Top 3 matches
        ]
        
    # Add learning insights
    output["learning_insights"] = reflection_result.learning_insights
    
    # Add context insights
    output["context_insights"] = reflection_result.context_insights
    
    return output


def _add_reflection_to_conversation(state: AnalysisState, reflection_output: Dict[str, Any]) -> None:
    """Add reflection insights to the conversation history."""
    from domain.entities import ConversationMessage
    from datetime import datetime
    
    # Create a user-friendly reflection message
    reflection_summary = _create_reflection_summary(reflection_output)
    
    message = ConversationMessage(
        timestamp=datetime.now(),
        role="assistant",
        content=reflection_summary,
        message_type="reflection"
    )
    
    if "conversation_history" not in state:
        state["conversation_history"] = []
        
    state["conversation_history"].append(message)


def _create_reflection_summary(reflection_output: Dict[str, Any]) -> str:
    """Create a user-friendly summary of the reflection analysis."""
    parts = ["I've analyzed the failure and here's what I found:"]
    
    # Add error analysis
    if "error_analysis" in reflection_output:
        error_analysis = reflection_output["error_analysis"]
        parts.append(f"**Error Type**: {error_analysis['category']} ({error_analysis['severity']} severity)")
        parts.append(f"**Root Cause**: {error_analysis['primary_cause']}")
        
    # Add root cause analysis
    if reflection_output.get("root_cause_analysis"):
        parts.append(f"**Analysis**: {reflection_output['root_cause_analysis']}")
        
    # Add improvement suggestions
    if reflection_output.get("improvement_suggestions"):
        parts.append("**Suggestions for improvement**:")
        for i, suggestion in enumerate(reflection_output["improvement_suggestions"][:3], 1):
            parts.append(f"{i}. {suggestion}")
            
    # Add pattern matches
    if reflection_output.get("pattern_matches"):
        parts.append("**Similar issues found**:")
        for match in reflection_output["pattern_matches"][:2]:
            parts.append(f"- {match['pattern_name']} (confidence: {match['similarity_score']:.1%})")
            
    # Add retry information
    if reflection_output.get("should_retry"):
        parts.append(f"**Next step**: {reflection_output.get('retry_strategy', 'Retry with improvements')}")
    else:
        parts.append("**Recommendation**: Manual review may be needed")
        
    return "\n\n".join(parts)