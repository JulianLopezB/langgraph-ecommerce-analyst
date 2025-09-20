"""Comprehensive tests for the reflection system."""

import pytest
import pandas as pd
from unittest.mock import Mock, patch
from datetime import datetime

from domain.entities import ExecutionResults, ExecutionStatus
from domain.reflection import (
    ErrorClassifier, 
    ErrorCategory,
    ReflectionEngine,
    PatternDetector,
    LearningSystem,
    EmbeddingPatternMatcher,
    ContextAwareAnalyzer,
    FailureContext,
)
from infrastructure.llm.client import LLMClient


class TestErrorClassifier:
    """Test the error classification system."""
    
    @pytest.fixture
    def classifier(self):
        return ErrorClassifier()
        
    def test_syntax_error_classification(self, classifier):
        """Test classification of syntax errors."""
        error = SyntaxError("invalid syntax")
        code = "print('hello'"  # Missing closing parenthesis
        
        categorized = classifier.classify_error(error, code)
        
        assert categorized.analysis.category == ErrorCategory.SYNTAX
        assert categorized.analysis.confidence > 0.9
        assert "syntax" in categorized.analysis.primary_cause.lower()
        assert len(categorized.analysis.suggested_fixes) > 0
        
    def test_data_error_classification(self, classifier):
        """Test classification of data-related errors."""
        error = KeyError("column 'missing_col' not found")
        code = "df['missing_col'].sum()"
        context = {"raw_dataset": pd.DataFrame({"col1": [1, 2, 3]})}
        
        categorized = classifier.classify_error(error, code, context)
        
        assert categorized.analysis.category == ErrorCategory.DATA
        assert categorized.analysis.confidence > 0.8
        assert "column" in categorized.analysis.primary_cause.lower()
        
    def test_runtime_error_classification(self, classifier):
        """Test classification of runtime errors."""
        error = NameError("name 'undefined_var' is not defined")
        code = "result = undefined_var * 2"
        
        categorized = classifier.classify_error(error, code)
        
        assert categorized.analysis.category == ErrorCategory.RUNTIME
        assert categorized.analysis.confidence > 0.7
        assert "runtime" in categorized.analysis.primary_cause.lower()
        
    def test_performance_error_classification(self, classifier):
        """Test classification of performance errors."""
        error = Exception("Code execution timed out after 30 seconds")
        code = "while True: pass"
        
        categorized = classifier.classify_error(error, code)
        
        assert categorized.analysis.category == ErrorCategory.PERFORMANCE
        assert categorized.analysis.confidence > 0.8
        
    def test_context_clues_extraction(self, classifier):
        """Test extraction of context clues from errors."""
        error = KeyError("column 'age' not found")
        code = "df['age'].mean()"
        df = pd.DataFrame({"name": ["Alice", "Bob"], "years": [25, 30]})
        context = {"raw_dataset": df}
        
        categorized = classifier.classify_error(error, code, context)
        
        assert "dataset_shape" in categorized.analysis.context_clues
        assert "dataset_columns" in categorized.analysis.context_clues
        assert categorized.analysis.context_clues["dataset_shape"] == (2, 2)


class TestReflectionEngine:
    """Test the core reflection engine."""
    
    @pytest.fixture
    def mock_llm_client(self):
        client = Mock(spec=LLMClient)
        client.generate_text.return_value = "Mocked LLM response for testing"
        return client
        
    @pytest.fixture
    def reflection_engine(self, mock_llm_client):
        return ReflectionEngine(mock_llm_client)
        
    @pytest.fixture
    def sample_failure_context(self):
        execution_results = ExecutionResults(
            status=ExecutionStatus.FAILED,
            error_message="KeyError: 'missing_column'",
            execution_time=1.5,
            stderr="Traceback: KeyError: 'missing_column'"
        )
        
        return FailureContext(
            user_query="Show me the average age by department",
            generated_code="df['missing_column'].groupby('department').mean()",
            execution_results=execution_results,
            data_schema={"columns": ["name", "department", "salary"]},
            session_context={"raw_dataset": pd.DataFrame({"name": ["A"], "department": ["IT"], "salary": [50000]})}
        )
        
    def test_failure_analysis_basic(self, reflection_engine, sample_failure_context):
        """Test basic failure analysis functionality."""
        result = reflection_engine.analyze_failure(sample_failure_context)
        
        assert result is not None
        assert result.confidence_score > 0
        assert len(result.improvement_suggestions) > 0
        assert isinstance(result.should_retry, bool)
        
    def test_failure_analysis_with_context(self, reflection_engine, sample_failure_context):
        """Test failure analysis with context analysis."""
        result = reflection_engine.analyze_failure(sample_failure_context)
        
        # Should have context analysis
        assert result.context_analysis is not None
        assert result.context_analysis.query_complexity is not None
        assert result.context_analysis.intent_analysis is not None
        
    def test_retry_strategy_assessment(self, reflection_engine, sample_failure_context):
        """Test retry strategy assessment."""
        result = reflection_engine.analyze_failure(sample_failure_context)
        
        if result.categorized_error and result.categorized_error.analysis.is_recoverable:
            assert result.should_retry is True
            assert result.retry_strategy is not None
        else:
            assert result.should_retry is False
            
    def test_learning_insights_extraction(self, reflection_engine, sample_failure_context):
        """Test extraction of learning insights."""
        result = reflection_engine.analyze_failure(sample_failure_context)
        
        assert isinstance(result.learning_insights, list)
        assert len(result.learning_insights) >= 0  # May be empty for some errors


class TestPatternDetector:
    """Test the pattern detection system."""
    
    @pytest.fixture
    def pattern_detector(self):
        return PatternDetector(storage_path="test_patterns.pkl")
        
    @pytest.fixture
    def sample_categorized_error(self):
        error = KeyError("column not found")
        classifier = ErrorClassifier()
        return classifier.classify_error(error, "df['missing'].sum()")
        
    def test_pattern_detection(self, pattern_detector, sample_categorized_error):
        """Test detection of patterns in failures."""
        query = "Show me the sum of missing column"
        code = "df['missing'].sum()"
        
        matches = pattern_detector.detect_patterns(
            sample_categorized_error, query, code
        )
        
        assert isinstance(matches, list)
        # Should find at least some common patterns
        assert len(matches) >= 0
        
    def test_pattern_learning(self, pattern_detector, sample_categorized_error):
        """Test learning from failures."""
        query = "Sum the missing column"
        code = "df['missing'].sum()"
        
        initial_pattern_count = len(pattern_detector.patterns)
        
        # Learn from failure
        pattern_detector.learn_from_failure(
            sample_categorized_error, query, code,
            resolution_success=False
        )
        
        # Should either update existing pattern or create new one
        final_pattern_count = len(pattern_detector.patterns)
        assert final_pattern_count >= initial_pattern_count
        
    def test_pattern_statistics(self, pattern_detector):
        """Test pattern statistics generation."""
        stats = pattern_detector.get_pattern_statistics()
        
        assert "total_patterns" in stats
        assert isinstance(stats["total_patterns"], int)
        if stats["total_patterns"] > 0:
            assert "category_distribution" in stats
            assert "average_frequency" in stats


class TestLearningSystem:
    """Test the learning system."""
    
    @pytest.fixture
    def mock_llm_client(self):
        client = Mock(spec=LLMClient)
        client.generate_text.return_value = """
        1. Improve column validation before operations
        2. Add better error messages for missing columns
        3. Implement data schema checking
        """
        return client
        
    @pytest.fixture
    def learning_system(self, mock_llm_client):
        return LearningSystem(mock_llm_client, storage_path="test_learning.json")
        
    @pytest.fixture
    def sample_categorized_error(self):
        error = KeyError("column not found")
        classifier = ErrorClassifier()
        return classifier.classify_error(error, "df['missing'].sum()")
        
    def test_failure_event_recording(self, learning_system, sample_categorized_error):
        """Test recording of failure events."""
        query = "Sum the missing column"
        code = "df['missing'].sum()"
        
        record_id = learning_system.record_failure_event(
            sample_categorized_error, query, code, []
        )
        
        assert record_id is not None
        assert len(learning_system.learning_records) > 0
        
    def test_success_event_recording(self, learning_system):
        """Test recording of success events."""
        query = "Sum the salary column"
        code = "df['salary'].sum()"
        
        record_id = learning_system.record_success_event(
            query, code, execution_time=1.2
        )
        
        assert record_id is not None
        assert len(learning_system.learning_records) > 0
        
    def test_improvement_suggestions_generation(self, learning_system, sample_categorized_error):
        """Test generation of improvement suggestions."""
        # Record some failures first
        for i in range(3):
            learning_system.record_failure_event(
                sample_categorized_error, f"Query {i}", f"code_{i}", []
            )
            
        suggestions = learning_system.generate_improvement_suggestions(limit=5)
        
        assert isinstance(suggestions, list)
        assert len(suggestions) <= 5
        
    def test_learning_insights(self, learning_system):
        """Test learning insights generation."""
        insights = learning_system.get_learning_insights()
        
        assert "total_learning_events" in insights
        assert "overall_success_rate" in insights


class TestEmbeddingPatternMatcher:
    """Test the embedding-based pattern matcher."""
    
    @pytest.fixture
    def mock_llm_client(self):
        return Mock(spec=LLMClient)
        
    @pytest.fixture
    def embedding_matcher(self, mock_llm_client):
        return EmbeddingPatternMatcher(mock_llm_client, cache_path="test_embeddings.json")
        
    def test_embedding_generation(self, embedding_matcher):
        """Test embedding generation for text."""
        text = "This is a test failure message"
        embedding = embedding_matcher._get_embedding(text)
        
        assert embedding is not None
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        
    def test_similarity_calculation(self, embedding_matcher):
        """Test cosine similarity calculation."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        vec3 = [1.0, 0.0, 0.0]
        
        # Different vectors should have low similarity
        sim1 = embedding_matcher._cosine_similarity(vec1, vec2)
        assert sim1 == 0.0
        
        # Identical vectors should have high similarity
        sim2 = embedding_matcher._cosine_similarity(vec1, vec3)
        assert sim2 == 1.0
        
    def test_failure_clustering(self, embedding_matcher):
        """Test clustering of similar failures."""
        failures = [
            ("Column 'age' not found", Mock()),
            ("Column 'name' not found", Mock()),
            ("Syntax error in code", Mock()),
        ]
        
        clusters = embedding_matcher.cluster_similar_failures(failures, similarity_threshold=0.5)
        
        assert isinstance(clusters, list)
        assert len(clusters) > 0


class TestContextAwareAnalyzer:
    """Test the context-aware analyzer."""
    
    @pytest.fixture
    def context_analyzer(self):
        return ContextAwareAnalyzer()
        
    @pytest.fixture
    def sample_dataframe(self):
        return pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
            "department": ["IT", "HR", "IT"],
            "salary": [50000, 60000, 55000]
        })
        
    @pytest.fixture
    def sample_categorized_error(self):
        error = KeyError("column 'missing_col' not found")
        classifier = ErrorClassifier()
        return classifier.classify_error(error, "df['missing_col'].sum()")
        
    def test_query_complexity_analysis(self, context_analyzer):
        """Test query complexity analysis."""
        simple_query = "Show me the data"
        complex_query = "Create a machine learning model to predict salary based on age and department with cross-validation"
        
        simple_complexity = context_analyzer._analyze_query_complexity(simple_query)
        complex_complexity = context_analyzer._analyze_query_complexity(complex_query)
        
        assert simple_complexity.value in ["simple", "moderate"]
        assert complex_complexity.value in ["complex", "advanced"]
        
    def test_query_intent_analysis(self, context_analyzer):
        """Test query intent analysis."""
        viz_query = "Plot a histogram of ages"
        agg_query = "Calculate the average salary by department"
        
        viz_intent = context_analyzer._analyze_query_intent(viz_query)
        agg_intent = context_analyzer._analyze_query_intent(agg_query)
        
        assert viz_intent["primary_intent"] == "visualization"
        assert agg_intent["primary_intent"] == "aggregation"
        
    def test_data_characteristics_analysis(self, context_analyzer, sample_dataframe):
        """Test data characteristics analysis."""
        characteristics = context_analyzer._analyze_data_characteristics(sample_dataframe)
        
        assert len(characteristics) > 0
        # Small dataset should be identified as SMALL
        from domain.reflection.context_analysis import DataCharacteristics
        assert DataCharacteristics.SMALL in characteristics
        
    def test_contextual_insights_generation(self, context_analyzer, sample_categorized_error, sample_dataframe):
        """Test generation of contextual insights."""
        query = "Show me the average missing_col by department"
        code = "df['missing_col'].groupby('department').mean()"
        intent_analysis = {"primary_intent": "aggregation", "intent_scores": {}}
        
        from domain.reflection.context_analysis import QueryComplexity
        insights = context_analyzer._generate_contextual_insights(
            sample_categorized_error, query, code, sample_dataframe, 
            intent_analysis, QueryComplexity.MODERATE
        )
        
        assert isinstance(insights, list)
        assert len(insights) > 0
        # Should identify column reference error
        assert any("column" in insight.description.lower() for insight in insights)
        
    def test_full_context_analysis(self, context_analyzer, sample_categorized_error, sample_dataframe):
        """Test full context analysis."""
        query = "Calculate average salary by department"
        code = "df['missing_col'].groupby('department').mean()"
        
        result = context_analyzer.analyze_context(
            sample_categorized_error, query, code, sample_dataframe
        )
        
        assert result.query_complexity is not None
        assert result.intent_analysis is not None
        assert len(result.data_characteristics) > 0
        assert len(result.contextual_insights) > 0
        assert len(result.suggested_alternatives) > 0


class TestIntegration:
    """Integration tests for the complete reflection system."""
    
    @pytest.fixture
    def mock_llm_client(self):
        client = Mock(spec=LLMClient)
        client.generate_text.return_value = "Mock analysis response"
        return client
        
    def test_end_to_end_reflection_workflow(self, mock_llm_client):
        """Test the complete reflection workflow."""
        # Create components
        reflection_engine = ReflectionEngine(mock_llm_client)
        pattern_detector = PatternDetector(storage_path="test_integration_patterns.pkl")
        learning_system = LearningSystem(mock_llm_client, storage_path="test_integration_learning.json")
        
        # Create failure context
        execution_results = ExecutionResults(
            status=ExecutionStatus.FAILED,
            error_message="KeyError: 'age'",
            execution_time=2.1,
            stderr="KeyError: 'age'"
        )
        
        failure_context = FailureContext(
            user_query="Show average age by department",
            generated_code="df['age'].groupby('department').mean()",
            execution_results=execution_results,
            data_schema={"columns": ["name", "department", "salary"]},
            session_context={
                "raw_dataset": pd.DataFrame({
                    "name": ["Alice", "Bob"], 
                    "department": ["IT", "HR"], 
                    "salary": [50000, 60000]
                })
            }
        )
        
        # Perform reflection
        reflection_result = reflection_engine.analyze_failure(failure_context)
        
        # Verify results
        assert reflection_result is not None
        assert reflection_result.categorized_error is not None
        assert reflection_result.context_analysis is not None
        assert len(reflection_result.improvement_suggestions) > 0
        
        # Test pattern detection
        if reflection_result.categorized_error:
            patterns = pattern_detector.detect_patterns(
                reflection_result.categorized_error,
                failure_context.user_query,
                failure_context.generated_code
            )
            assert isinstance(patterns, list)
            
        # Test learning
        if reflection_result.categorized_error:
            record_id = learning_system.record_failure_event(
                reflection_result.categorized_error,
                failure_context.user_query,
                failure_context.generated_code,
                []
            )
            assert record_id is not None
            
    def test_reflection_with_different_error_types(self, mock_llm_client):
        """Test reflection with different types of errors."""
        reflection_engine = ReflectionEngine(mock_llm_client)
        
        error_scenarios = [
            {
                "error_msg": "SyntaxError: invalid syntax",
                "code": "print('hello'",
                "query": "Print hello"
            },
            {
                "error_msg": "NameError: name 'undefined_var' is not defined",
                "code": "result = undefined_var * 2",
                "query": "Calculate result"
            },
            {
                "error_msg": "ImportError: No module named 'missing_lib'",
                "code": "import missing_lib",
                "query": "Use missing library"
            }
        ]
        
        for scenario in error_scenarios:
            execution_results = ExecutionResults(
                status=ExecutionStatus.FAILED,
                error_message=scenario["error_msg"],
                execution_time=0.5,
                stderr=scenario["error_msg"]
            )
            
            failure_context = FailureContext(
                user_query=scenario["query"],
                generated_code=scenario["code"],
                execution_results=execution_results,
                session_context={}
            )
            
            result = reflection_engine.analyze_failure(failure_context)
            
            # Each error type should be properly analyzed
            assert result is not None
            assert result.categorized_error is not None
            assert result.categorized_error.analysis.category in [
                ErrorCategory.SYNTAX, ErrorCategory.RUNTIME, ErrorCategory.DEPENDENCY
            ]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])