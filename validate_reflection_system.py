#!/usr/bin/env python3
"""Simple validation script for the reflection system without external dependencies."""

import sys
import os

# Add workspace to Python path
sys.path.insert(0, '/workspace')

def test_basic_imports():
    """Test that core reflection components can be imported."""
    print("Testing basic imports...")
    
    try:
        # Test error categorization
        from domain.reflection.error_categorization import ErrorCategory, ErrorClassifier
        print("✓ Error categorization imported successfully")
        
        # Test pattern detection
        from domain.reflection.pattern_detection import PatternDetector, FailurePattern
        print("✓ Pattern detection imported successfully")
        
        # Test context analysis
        from domain.reflection.context_analysis import ContextAwareAnalyzer, QueryComplexity
        print("✓ Context analysis imported successfully")
        
        # Test embeddings
        from domain.reflection.embeddings import EmbeddingPatternMatcher
        print("✓ Embeddings imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_error_classification():
    """Test basic error classification functionality."""
    print("\nTesting error classification...")
    
    try:
        # Mock the logger to avoid dependency issues
        import logging
        logging.basicConfig(level=logging.INFO)
        
        # Patch the get_logger function
        import domain.reflection.error_categorization as ec_module
        ec_module.logger = logging.getLogger("test")
        
        from domain.reflection.error_categorization import ErrorClassifier, ErrorCategory
        
        classifier = ErrorClassifier()
        
        # Test syntax error
        syntax_error = SyntaxError("invalid syntax")
        categorized = classifier.classify_error(syntax_error, "print('hello'")
        
        assert categorized.analysis.category == ErrorCategory.SYNTAX
        print("✓ Syntax error classification works")
        
        # Test runtime error
        runtime_error = NameError("name 'x' is not defined")
        categorized = classifier.classify_error(runtime_error, "print(x)")
        
        assert categorized.analysis.category == ErrorCategory.RUNTIME
        print("✓ Runtime error classification works")
        
        return True
        
    except Exception as e:
        print(f"✗ Error classification test failed: {e}")
        return False

def test_context_analysis():
    """Test context analysis functionality."""
    print("\nTesting context analysis...")
    
    try:
        # Mock dependencies
        import logging
        import domain.reflection.context_analysis as ca_module
        ca_module.logger = logging.getLogger("test")
        
        from domain.reflection.context_analysis import ContextAwareAnalyzer, QueryComplexity
        
        analyzer = ContextAwareAnalyzer()
        
        # Test query complexity analysis
        simple_query = "Show me the data"
        complexity = analyzer._analyze_query_complexity(simple_query)
        assert complexity in [QueryComplexity.SIMPLE, QueryComplexity.MODERATE]
        print("✓ Query complexity analysis works")
        
        # Test intent analysis
        viz_query = "Plot a histogram of ages"
        intent = analyzer._analyze_query_intent(viz_query)
        assert intent["primary_intent"] == "visualization"
        print("✓ Query intent analysis works")
        
        return True
        
    except Exception as e:
        print(f"✗ Context analysis test failed: {e}")
        return False

def test_pattern_detection():
    """Test pattern detection functionality."""
    print("\nTesting pattern detection...")
    
    try:
        # Mock dependencies
        import logging
        import domain.reflection.pattern_detection as pd_module
        pd_module.logger = logging.getLogger("test")
        
        from domain.reflection.pattern_detection import PatternDetector
        
        # Use a temporary file for testing
        detector = PatternDetector(storage_path="/tmp/test_patterns.pkl")
        
        # Test pattern statistics
        stats = detector.get_pattern_statistics()
        assert "total_patterns" in stats
        print("✓ Pattern statistics generation works")
        
        return True
        
    except Exception as e:
        print(f"✗ Pattern detection test failed: {e}")
        return False

def test_embeddings():
    """Test embeddings functionality.""" 
    print("\nTesting embeddings...")
    
    try:
        # Mock dependencies
        import logging
        import domain.reflection.embeddings as emb_module
        emb_module.logger = logging.getLogger("test")
        
        from domain.reflection.embeddings import EmbeddingPatternMatcher
        
        # Mock LLM client
        class MockLLMClient:
            def generate_text(self, **kwargs):
                return "mock response"
        
        matcher = EmbeddingPatternMatcher(MockLLMClient(), cache_path="/tmp/test_embeddings.json")
        
        # Test embedding generation
        embedding = matcher._get_embedding("test text")
        assert embedding is not None
        assert len(embedding) > 0
        print("✓ Embedding generation works")
        
        # Test similarity calculation
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        similarity = matcher._cosine_similarity(vec1, vec2)
        assert similarity == 1.0
        print("✓ Cosine similarity calculation works")
        
        return True
        
    except Exception as e:
        print(f"✗ Embeddings test failed: {e}")
        return False

def main():
    """Run all validation tests."""
    print("🔍 Validating Reflection System Implementation")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_error_classification,
        test_context_analysis,
        test_pattern_detection,
        test_embeddings,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Reflection system is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    exit(main())