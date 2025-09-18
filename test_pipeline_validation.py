#!/usr/bin/env python3
"""Simple validation script for the code generation pipeline."""

import sys
import traceback
from unittest.mock import Mock
import pandas as pd

# Add the workspace to Python path
sys.path.insert(0, '/workspace')

def test_pipeline_imports():
    """Test that all pipeline components can be imported."""
    try:
        from domain.pipeline import (
            CodeGenerationPipeline, 
            create_code_generation_pipeline,
            PipelineContext,
            PipelineStatus
        )
        from domain.pipeline.stages import (
            CodeGenerationStage,
            CodeCleaningStage, 
            CodeValidationStage,
            CodeExecutionStage
        )
        print("✓ All pipeline imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False

def test_pipeline_context():
    """Test pipeline context creation."""
    try:
        from domain.pipeline import PipelineContext
        
        context = PipelineContext(
            user_query="Test query",
            analysis_context={"test": "data"}
        )
        
        assert context.user_query == "Test query"
        assert context.analysis_context == {"test": "data"}
        assert context.code_content is None
        
        print("✓ Pipeline context creation successful")
        return True
    except Exception as e:
        print(f"✗ Pipeline context test failed: {e}")
        traceback.print_exc()
        return False

def test_code_cleaning_stage():
    """Test code cleaning stage."""
    try:
        from domain.pipeline.stages import CodeCleaningStage
        from domain.pipeline import PipelineContext
        
        stage = CodeCleaningStage()
        
        dirty_code = """```python
import pandas as pd
print("Hello World")
```"""
        
        context = PipelineContext(
            user_query="Test",
            analysis_context={},
            code_content=dirty_code
        )
        
        result = stage.execute(context)
        
        assert result.success is True
        assert "```python" not in result.data
        assert "```" not in result.data
        assert "import pandas as pd" in result.data
        
        print("✓ Code cleaning stage test successful")
        return True
    except Exception as e:
        print(f"✗ Code cleaning stage test failed: {e}")
        traceback.print_exc()
        return False

def test_mock_pipeline():
    """Test pipeline with mocked components."""
    try:
        from domain.pipeline import CodeGenerationPipeline
        from infrastructure.llm.base import LLMClient
        from infrastructure.execution.validator import CodeValidator, ValidationResult
        from infrastructure.execution.executor import SecureExecutor
        from domain.entities import ExecutionResults, ExecutionStatus
        
        # Create mocks
        mock_llm = Mock(spec=LLMClient)
        mock_llm.generate_adaptive_python_code.return_value = "print('Hello Pipeline')"
        
        mock_validator = Mock(spec=CodeValidator)
        mock_validator.validate.return_value = ValidationResult(
            is_valid=True,
            syntax_errors=[],
            security_warnings=[],
            performance_warnings=[],
            validation_time=0.1,
            security_score=1.0
        )
        
        mock_executor = Mock(spec=SecureExecutor)
        mock_executor.execute_code.return_value = ExecutionResults(
            status=ExecutionStatus.SUCCESS,
            output_data={"result": "success"},
            execution_time=1.0,
            memory_used_mb=50.0,
            stdout="Hello Pipeline",
            stderr=""
        )
        
        # Create pipeline
        pipeline = CodeGenerationPipeline(mock_llm, mock_validator, mock_executor)
        
        # Test pipeline execution
        result = pipeline.generate_and_execute_code(
            user_query="Test pipeline",
            analysis_context={
                "process_data": {"process_type": "python"},
                "data_characteristics": {"shape": (100, 5)},
                "raw_dataset": pd.DataFrame({"test": [1, 2, 3]})
            }
        )
        
        assert result.success is True
        assert len(result.stage_results) == 4
        
        print("✓ Mock pipeline execution test successful")
        return True
    except Exception as e:
        print(f"✗ Mock pipeline test failed: {e}")
        traceback.print_exc()
        return False

def test_pipeline_health():
    """Test pipeline health check."""
    try:
        from domain.pipeline import CodeGenerationPipeline
        from infrastructure.llm.base import LLMClient
        from infrastructure.execution.validator import CodeValidator
        from infrastructure.execution.executor import SecureExecutor
        
        # Create mocks
        mock_llm = Mock(spec=LLMClient)
        mock_validator = Mock(spec=CodeValidator)
        mock_validator.get_allowed_imports.return_value = {"pandas", "numpy"}
        
        mock_executor = Mock(spec=SecureExecutor)
        mock_executor.max_execution_time = 300
        mock_executor.max_memory_mb = 1024
        
        pipeline = CodeGenerationPipeline(mock_llm, mock_validator, mock_executor)
        health = pipeline.get_pipeline_health()
        
        assert health["pipeline_name"] == "code_generation_pipeline"
        assert health["total_stages"] == 4
        assert "stages_info" in health
        
        print("✓ Pipeline health check test successful")
        return True
    except Exception as e:
        print(f"✗ Pipeline health check test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all validation tests."""
    print("Running Code Generation Pipeline Validation Tests...")
    print("=" * 60)
    
    tests = [
        test_pipeline_imports,
        test_pipeline_context,
        test_code_cleaning_stage,
        test_mock_pipeline,
        test_pipeline_health
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        if test():
            passed += 1
        else:
            failed += 1
        print()
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Pipeline implementation is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())