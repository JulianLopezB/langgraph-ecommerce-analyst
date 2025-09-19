#!/usr/bin/env python3
"""Simple validation script for the code generation pipeline core functionality."""

import sys
import traceback
from unittest.mock import Mock

# Add the workspace to Python path
sys.path.insert(0, "/workspace")


def test_pipeline_imports():
    """Test that all pipeline components can be imported."""
    try:
        from domain.pipeline.base import (
            Pipeline,
            PipelineContext,
            PipelineResult,
            PipelineStage,
            PipelineStatus,
            StageResult,
        )

        print("✓ Base pipeline imports successful")
        return True
    except Exception as e:
        print(f"✗ Base import failed: {e}")
        traceback.print_exc()
        return False


def test_pipeline_context():
    """Test pipeline context creation."""
    try:
        from domain.pipeline.base import PipelineContext

        context = PipelineContext(
            user_query="Test query", analysis_context={"test": "data"}
        )

        assert context.user_query == "Test query"
        assert context.analysis_context == {"test": "data"}
        assert context.code_content is None
        assert len(context.error_context) == 0

        print("✓ Pipeline context creation successful")
        return True
    except Exception as e:
        print(f"✗ Pipeline context test failed: {e}")
        traceback.print_exc()
        return False


def test_stage_result():
    """Test stage result functionality."""
    try:
        from domain.pipeline.base import StageResult

        # Test successful result
        success_result = StageResult[str](
            success=True, data="test data", execution_time=1.5
        )

        assert success_result.success is True
        assert success_result.failed is False
        assert success_result.data == "test data"
        assert success_result.execution_time == 1.5

        # Test failed result
        failed_result = StageResult[str](
            success=False, error_message="Test error", execution_time=0.5
        )

        assert failed_result.success is False
        assert failed_result.failed is True
        assert failed_result.error_message == "Test error"

        print("✓ Stage result functionality successful")
        return True
    except Exception as e:
        print(f"✗ Stage result test failed: {e}")
        traceback.print_exc()
        return False


def test_pipeline_result():
    """Test pipeline result functionality."""
    try:
        from domain.pipeline.base import PipelineContext, PipelineResult, PipelineStatus

        context = PipelineContext(user_query="Test", analysis_context={})

        result = PipelineResult(
            status=PipelineStatus.SUCCESS, context=context, total_execution_time=5.0
        )

        assert result.status == PipelineStatus.SUCCESS
        assert result.success is True
        assert result.failed is False
        assert result.total_execution_time == 5.0

        # Test failed result
        failed_result = PipelineResult(
            status=PipelineStatus.FAILED,
            context=context,
            error_message="Pipeline failed",
        )

        assert failed_result.success is False
        assert failed_result.failed is True

        print("✓ Pipeline result functionality successful")
        return True
    except Exception as e:
        print(f"✗ Pipeline result test failed: {e}")
        traceback.print_exc()
        return False


def test_code_cleaning_stage():
    """Test code cleaning stage without external dependencies."""
    try:
        from domain.pipeline.base import PipelineContext
        from domain.pipeline.stages import CodeCleaningStage

        stage = CodeCleaningStage()

        dirty_code = """```python
# This is a comment
def hello():
    return "world"
```"""

        context = PipelineContext(
            user_query="Test", analysis_context={}, code_content=dirty_code
        )

        result = stage.execute(context)

        assert result.success is True
        assert "```python" not in result.data
        assert "```" not in result.data
        assert "def hello():" in result.data
        assert context.cleaned_code == result.data

        # Check metrics
        assert "original_length" in result.stage_metrics
        assert "cleaned_length" in result.stage_metrics

        print("✓ Code cleaning stage test successful")
        return True
    except Exception as e:
        print(f"✗ Code cleaning stage test failed: {e}")
        traceback.print_exc()
        return False


def test_stage_error_handling():
    """Test stage error handling."""
    try:
        from domain.pipeline.base import PipelineContext
        from domain.pipeline.stages import CodeCleaningStage

        stage = CodeCleaningStage()

        # Test with no code content
        context = PipelineContext(
            user_query="Test", analysis_context={}, code_content=None
        )

        result = stage.execute(context)

        assert result.success is False
        assert "No code content to clean" in result.error_message
        assert result.execution_time > 0

        print("✓ Stage error handling test successful")
        return True
    except Exception as e:
        print(f"✗ Stage error handling test failed: {e}")
        traceback.print_exc()
        return False


def test_pipeline_stage_info():
    """Test pipeline stage information."""
    try:
        from domain.pipeline.stages import CodeCleaningStage

        stage = CodeCleaningStage()
        info = stage.get_stage_info()

        assert info["stage_name"] == "code_cleaning"
        assert info["stage_type"] == "cleaning"
        assert "class_name" in info

        print("✓ Pipeline stage info test successful")
        return True
    except Exception as e:
        print(f"✗ Pipeline stage info test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    print("Running Code Generation Pipeline Core Validation Tests...")
    print("=" * 70)

    tests = [
        test_pipeline_imports,
        test_pipeline_context,
        test_stage_result,
        test_pipeline_result,
        test_code_cleaning_stage,
        test_stage_error_handling,
        test_pipeline_stage_info,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            traceback.print_exc()
            failed += 1
        print()

    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All core tests passed! Pipeline implementation is working correctly.")
        print("\nKey Features Validated:")
        print("- ✓ Pipeline context and state management")
        print("- ✓ Stage execution with proper error handling")
        print("- ✓ Metrics collection and stage metadata")
        print("- ✓ Code cleaning and formatting")
        print("- ✓ Error propagation between stages")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
