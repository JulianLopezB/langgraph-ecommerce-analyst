# Code Reflection System Implementation

## Overview

This implementation provides a comprehensive code reflection system that analyzes execution failures and learns from them, following LangGraph patterns as specified in the requirements (DAT-22).

## 🎯 Acceptance Criteria - COMPLETED

- ✅ **Automatic error categorization and analysis**
- ✅ **Context-aware failure understanding** 
- ✅ **Pattern detection for common failure modes**
- ✅ **Learning mechanism for continuous improvement**

## 🏗️ Architecture

The reflection system is built with a modular architecture consisting of:

### 1. Error Categorization (`domain/reflection/error_categorization.py`)
- **ErrorCategory Enum**: Categorizes errors into SYNTAX, RUNTIME, LOGIC, DATA, PERFORMANCE, SECURITY, DEPENDENCY, UNKNOWN
- **ErrorClassifier**: Analyzes exceptions and categorizes them with confidence scores
- **ErrorAnalysis**: Detailed analysis including primary cause, contributing factors, suggested fixes, and severity assessment
- **CategorizedError**: Complete error information with categorization and analysis

**Key Features:**
- Pattern-based classification using regex patterns
- Context-aware analysis considering code and data
- Severity assessment and recoverability analysis
- Extraction of context clues (line numbers, variables, data characteristics)

### 2. Reflection Engine (`domain/reflection/reflection_engine.py`)
- **ReflectionEngine**: Core engine for comprehensive failure analysis
- **FailureContext**: Complete context information for analysis
- **ReflectionResult**: Comprehensive reflection output with suggestions

**Key Features:**
- LLM-powered root cause analysis
- Context-aware improvement suggestions
- Retry strategy assessment
- Learning insights extraction
- Confidence scoring

### 3. Pattern Detection (`domain/reflection/pattern_detection.py`)
- **PatternDetector**: Detects and learns from failure patterns
- **FailurePattern**: Represents common failure patterns with metadata
- **PatternMatch**: Matches between failures and known patterns

**Key Features:**
- Pattern similarity calculation
- Learning from failures (success/failure tracking)
- Pattern statistics and analytics
- Persistent storage of learned patterns

### 4. Context-Aware Analysis (`domain/reflection/context_analysis.py`)
- **ContextAwareAnalyzer**: Analyzes failures in context of data and query intent
- **QueryComplexity**: Classifies query complexity (SIMPLE, MODERATE, COMPLEX, ADVANCED)
- **DataCharacteristics**: Identifies data characteristics (size, sparsity, mixed types, etc.)
- **ContextualInsight**: Provides contextual insights about failures

**Key Features:**
- Query intent classification (exploration, filtering, aggregation, visualization, etc.)
- Data characteristics analysis
- Intent-implementation mismatch detection
- Alternative approach suggestions

### 5. Embeddings-Based Pattern Matching (`domain/reflection/embeddings.py`)
- **EmbeddingPatternMatcher**: Uses embeddings for semantic similarity
- **Pattern clustering and similarity matching**
- **Embedding caching for performance**

**Key Features:**
- Semantic similarity between failures and patterns
- Failure clustering based on similarity
- Enhancement of traditional pattern matching with embeddings
- Efficient caching system

### 6. Learning System (`domain/reflection/learning_system.py`)
- **LearningSystem**: Continuous learning from failures and successes
- **LearningRecord**: Records learning events for analysis
- **ImprovementSuggestion**: Structured improvement suggestions

**Key Features:**
- Recording of failure and success events
- Pattern-based improvement generation
- LLM-powered suggestion generation
- Learning analytics and insights
- Suggestion validation tracking

### 7. LangGraph Integration (`workflow/nodes/reflection.py`)
- **reflect_on_failure**: LangGraph node for failure reflection
- **reflect_on_success**: LangGraph node for success reflection
- **Integration with main workflow**: Updated routing logic

**Key Features:**
- Seamless integration with existing LangGraph workflow
- Automatic reflection triggering based on execution results
- State management for reflection results
- Conversation history integration

## 🔄 Workflow Integration

The reflection system is integrated into the LangGraph workflow as follows:

1. **Execution Node** → Checks execution results
2. **Success Path** → `reflect_on_success` → `synthesize_results`
3. **Failure Path** → `reflect_on_failure` → `retry_with_reflection` OR `handle_error`

### Updated Graph Routing
- Modified `_route_after_execution` to automatically trigger reflection
- Added `_route_after_reflection` for post-reflection routing
- Integrated reflection results into conversation history

## 🧠 Key Capabilities

### Automatic Error Categorization
```python
# Example: KeyError gets categorized as DATA error
error = KeyError("column 'age' not found")
categorized = classifier.classify_error(error, code, context)
# Result: categorized.analysis.category = ErrorCategory.DATA
```

### Context-Aware Analysis
```python
# Analyzes query intent, data characteristics, and failure context
context_result = analyzer.analyze_context(
    categorized_error, query, code, dataframe
)
# Provides insights like "Visualization intent but missing plotting library"
```

### Pattern Detection and Learning
```python
# Detects similar failure patterns
patterns = detector.detect_patterns(error, query, code)
# Learns from resolution success/failure
detector.learn_from_failure(error, query, code, success=True)
```

### Embeddings-Based Matching
```python
# Finds semantically similar failures
similar = matcher.find_similar_patterns(failure_text, patterns)
# Clusters related failures
clusters = matcher.cluster_similar_failures(failures)
```

## 📊 Learning and Improvement

### Continuous Learning
- Records all failure and success events
- Builds pattern database over time
- Generates improvement suggestions based on patterns
- Validates suggestions with real-world results

### Analytics and Insights
- Pattern frequency and success rates
- Error category distributions
- Query intent analysis
- Performance metrics tracking

## 🔧 Configuration and Storage

### Persistent Storage
- Pattern data: `data/reflection_patterns.pkl`
- Learning records: `data/learning_records.json`
- Embedding cache: `data/embedding_cache.json`

### Configurable Components
- Pattern similarity thresholds
- Learning rate parameters
- LLM temperature settings
- Storage paths

## 🚀 Usage Example

```python
from domain.reflection import ReflectionEngine, FailureContext
from infrastructure.llm.client import get_llm_client

# Initialize reflection engine
engine = ReflectionEngine(get_llm_client())

# Create failure context
context = FailureContext(
    user_query="Show average salary by department",
    generated_code="df['salary'].groupby('dept').mean()",
    execution_results=failed_results,
    data_schema=schema_info
)

# Perform reflection
result = engine.analyze_failure(context)

# Get insights
print(f"Error Category: {result.categorized_error.analysis.category}")
print(f"Root Cause: {result.root_cause_analysis}")
print(f"Suggestions: {result.improvement_suggestions}")
print(f"Should Retry: {result.should_retry}")
```

## 🧪 Testing

Comprehensive test suite provided in `tests/test_reflection_system.py` covering:
- Error classification accuracy
- Context analysis functionality
- Pattern detection and learning
- Embeddings-based matching
- Integration testing
- End-to-end workflow validation

## 🔮 Future Enhancements

1. **Advanced ML Models**: Replace simple embeddings with specialized models
2. **Real-time Learning**: Online learning from user feedback
3. **Multi-modal Analysis**: Include execution traces and profiling data
4. **Federated Learning**: Learn from multiple system instances
5. **Advanced Visualization**: Pattern and learning visualization dashboards

## 📝 Technical Notes

### LangGraph Pattern Compliance
- Follows LangGraph reflection patterns from code assistant tutorials
- Implements structured error analysis with actionable feedback
- Uses embeddings for error pattern matching as suggested
- Provides continuous learning mechanism

### Dependencies
- Built on existing infrastructure (LLM client, logging, entities)
- Integrates with current pipeline and workflow systems
- Minimal external dependencies (uses existing pandas, numpy)
- Modular design allows independent component usage

### Performance Considerations
- Embedding caching for efficiency
- Configurable pattern storage
- Lazy initialization of components
- Efficient similarity calculations

## 🎉 Conclusion

This reflection system provides a comprehensive solution for analyzing execution failures and learning from them. It successfully implements all required acceptance criteria while following LangGraph patterns and providing extensive capabilities for continuous improvement.

The system is production-ready and can be immediately integrated into the existing data analysis workflow to provide intelligent error analysis, pattern detection, and learning capabilities.