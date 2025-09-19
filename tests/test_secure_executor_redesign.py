"""Tests for the redesigned secure execution environment."""

import pytest
import time
from unittest.mock import patch, MagicMock

from domain.entities import ExecutionStatus
from infrastructure.config import ExecutionLimits
from infrastructure.execution.secure_executor import SecureExecutor
from infrastructure.execution.security import SecurityIsolator, ContextIsolator
from infrastructure.execution.resources import ResourceManager, OutputManager, TimeoutException, MemoryLimitException
from infrastructure.execution.tracing import ExecutionTracer, ExecutionProfiler
from infrastructure.execution.results import EnhancedExecutionResults, ExecutionMetadata


class TestSecurityIsolator:
    """Test security isolation components."""
    
    def test_create_safe_globals(self):
        """Test safe globals creation."""
        isolator = SecurityIsolator()
        safe_globals = isolator.create_safe_globals()
        
        # Check that builtins are properly filtered
        assert "__builtins__" in safe_globals
        builtins = safe_globals["__builtins__"]
        
        # Safe functions should be present
        assert "print" in builtins
        assert "len" in builtins
        assert "str" in builtins
        
        # Dangerous functions should not be present
        assert "eval" not in builtins
        assert "exec" not in builtins
        assert "__import__" not in builtins
    
    def test_safe_modules_import(self):
        """Test that safe modules are imported correctly."""
        isolator = SecurityIsolator()
        safe_globals = isolator.create_safe_globals()
        
        # Common data science modules should be available
        expected_modules = ["pd", "np", "plt", "math", "json"]
        for module_name in expected_modules:
            if module_name in safe_globals:  # May not be available in test environment
                assert safe_globals[module_name] is not None


class TestContextIsolator:
    """Test context isolation components."""
    
    def test_prepare_execution_context(self):
        """Test execution context preparation."""
        isolator = ContextIsolator()
        
        user_context = {
            "data": [1, 2, 3],
            "config": {"param": "value"},
            "_private": "should_be_filtered",
            "function": lambda x: x  # Should be filtered
        }
        
        safe_globals, safe_locals = isolator.prepare_execution_context(user_context)
        
        # Check globals
        assert "__builtins__" in safe_globals
        
        # Check locals filtering
        assert "data" in safe_locals
        assert "config" in safe_locals
        assert "_private" not in safe_locals  # Private attribute filtered
        assert "function" not in safe_locals  # Callable filtered
    
    def test_empty_context(self):
        """Test handling of empty context."""
        isolator = ContextIsolator()
        safe_globals, safe_locals = isolator.prepare_execution_context(None)
        
        assert isinstance(safe_globals, dict)
        assert isinstance(safe_locals, dict)
        assert len(safe_locals) == 0


class TestResourceManager:
    """Test resource management components."""
    
    def test_resource_manager_initialization(self):
        """Test resource manager initialization."""
        limits = ExecutionLimits(max_execution_time=5, max_memory_mb=256, max_output_size_mb=10)
        manager = ResourceManager(limits)
        
        assert manager.limits == limits
        assert manager._start_time is None
        assert manager._peak_memory == 0.0
    
    @patch('infrastructure.execution.resources.resource.setrlimit')
    def test_managed_execution_success(self, mock_setrlimit):
        """Test successful managed execution."""
        limits = ExecutionLimits(max_execution_time=5, max_memory_mb=256, max_output_size_mb=10)
        manager = ResourceManager(limits)
        
        with manager.managed_execution() as usage:
            time.sleep(0.01)  # Small delay to test timing
            assert usage.execution_time == 0.0  # Not set until context exit
        
        # After context exit, usage should be populated
        assert usage.execution_time > 0.0
        assert usage.cpu_time >= 0.0
        assert usage.memory_used_mb >= 0.0
    
    def test_timeout_exception(self):
        """Test timeout handling."""
        limits = ExecutionLimits(max_execution_time=1, max_memory_mb=256, max_output_size_mb=10)
        manager = ResourceManager(limits)
        
        with pytest.raises(TimeoutException):
            with manager.managed_execution():
                time.sleep(2)  # Exceed timeout


class TestOutputManager:
    """Test output management components."""
    
    def test_output_within_limits(self):
        """Test output within size limits."""
        manager = OutputManager(max_output_size_mb=1)
        
        stdout = "Hello, World!"
        stderr = "Warning: test"
        
        limited_stdout, limited_stderr = manager.limit_output_size(stdout, stderr)
        
        assert limited_stdout == stdout
        assert limited_stderr == stderr
    
    def test_output_size_limiting(self):
        """Test output size limiting when exceeded."""
        manager = OutputManager(max_output_size_mb=1)
        
        # Create large outputs that exceed 1MB
        large_stdout = "A" * (512 * 1024)  # 512KB
        large_stderr = "B" * (600 * 1024)  # 600KB (total > 1MB)
        
        limited_stdout, limited_stderr = manager.limit_output_size(large_stdout, large_stderr)
        
        # Outputs should be truncated
        assert len(limited_stdout) < len(large_stdout)
        assert len(limited_stderr) < len(large_stderr)
        assert "[OUTPUT TRUNCATED]" in limited_stdout
        assert "[OUTPUT TRUNCATED]" in limited_stderr


class TestExecutionTracer:
    """Test execution tracing components."""
    
    def test_tracer_initialization(self):
        """Test tracer initialization."""
        tracer = ExecutionTracer(enable_detailed_tracing=True)
        
        assert tracer.enable_detailed_tracing is True
        assert tracer.current_trace is None
    
    def test_trace_lifecycle(self):
        """Test complete trace lifecycle."""
        tracer = ExecutionTracer()
        
        # Start trace
        code = "x = 1 + 1"
        context = {"data": [1, 2, 3]}
        trace = tracer.start_trace(code, context)
        
        assert trace is not None
        assert trace.trace_id is not None
        assert trace.start_time > 0
        assert trace.context_keys == ["data"]
        
        # Add some steps and snapshots
        tracer.add_execution_step("test_step", {"detail": "test"})
        tracer.add_memory_snapshot(100.5)
        
        # End trace
        final_locals = {"x": 2, "analysis_results": {"result": "success"}}
        completed_trace = tracer.end_trace(success=True, final_locals=final_locals)
        
        assert completed_trace.end_time is not None
        assert completed_trace.duration > 0
        assert len(completed_trace.execution_steps) == 1
        assert len(completed_trace.memory_snapshots) == 1
        assert completed_trace.final_locals is not None


class TestExecutionProfiler:
    """Test execution profiling components."""
    
    def test_profiler_initialization(self):
        """Test profiler initialization."""
        profiler = ExecutionProfiler()
        
        assert isinstance(profiler.profiles, dict)
        assert len(profiler.profiles) == 0
    
    def test_profile_execution(self):
        """Test execution profiling."""
        profiler = ExecutionProfiler()
        tracer = ExecutionTracer()
        
        # Create a trace
        trace = tracer.start_trace("x = 1", {})
        tracer.add_memory_snapshot(50.0)
        tracer.add_memory_snapshot(75.0)
        tracer.add_execution_step("line_trace", {"line_number": 1})
        completed_trace = tracer.end_trace(success=True)
        
        # Profile the trace
        profile = profiler.profile_execution(completed_trace)
        
        assert "trace_id" in profile
        assert "total_duration" in profile
        assert "memory_profile" in profile
        assert "execution_profile" in profile
        
        # Check memory profile
        memory_profile = profile["memory_profile"]
        assert memory_profile["peak_memory"] == 75.0
        assert memory_profile["initial_memory"] == 50.0
        assert memory_profile["memory_growth"] == 25.0


class TestEnhancedExecutionResults:
    """Test enhanced execution results."""
    
    def test_results_initialization(self):
        """Test results initialization."""
        results = EnhancedExecutionResults(status=ExecutionStatus.SUCCESS)
        
        assert results.status == ExecutionStatus.SUCCESS
        assert results.metadata is not None
        assert isinstance(results.metadata, ExecutionMetadata)
    
    def test_legacy_conversion(self):
        """Test conversion to/from legacy results."""
        from domain.entities import ExecutionResults
        
        # Create legacy results
        legacy = ExecutionResults(
            status=ExecutionStatus.SUCCESS,
            output_data={"test": "data"},
            execution_time=1.5,
            memory_used_mb=128.0,
            stdout="Hello",
            stderr="Warning"
        )
        
        # Convert to enhanced
        enhanced = EnhancedExecutionResults.from_legacy_results(legacy)
        
        assert enhanced.status == ExecutionStatus.SUCCESS
        assert enhanced.output_data == {"test": "data"}
        assert enhanced.metadata.execution_time == 1.5
        assert enhanced.metadata.memory_used_mb == 128.0
        
        # Convert back to legacy
        converted_legacy = enhanced.to_legacy_results()
        
        assert converted_legacy.status == ExecutionStatus.SUCCESS
        assert converted_legacy.output_data == {"test": "data"}
        assert converted_legacy.execution_time == 1.5
        assert converted_legacy.memory_used_mb == 128.0
    
    def test_metadata_updates(self):
        """Test metadata update methods."""
        results = EnhancedExecutionResults(status=ExecutionStatus.SUCCESS)
        
        # Add security warning
        results.add_security_warning("Test warning")
        assert "Test warning" in results.metadata.security_warnings
        
        # Add resource limit hit
        results.add_resource_limit_hit("memory")
        assert "memory" in results.metadata.resource_limits_hit


class TestSecureExecutor:
    """Test the redesigned secure executor."""
    
    def test_executor_initialization(self):
        """Test executor initialization."""
        limits = ExecutionLimits(max_execution_time=5, max_memory_mb=256, max_output_size_mb=10)
        executor = SecureExecutor(limits, enable_tracing=True)
        
        assert executor.limits == limits
        assert executor.tracer is not None
        assert executor.profiler is not None
        assert executor.context_isolator is not None
        assert executor.resource_manager is not None
        assert executor.output_manager is not None
    
    def test_simple_code_execution(self):
        """Test simple successful code execution."""
        limits = ExecutionLimits(max_execution_time=5, max_memory_mb=256, max_output_size_mb=10)
        executor = SecureExecutor(limits)
        
        # Mock resource management to avoid system limitations in tests
        with patch.object(executor.resource_manager, '_set_resource_limits'):
            with patch.object(executor.resource_manager, '_get_current_memory_usage', return_value=50.0):
                code = "print('Hello, World!')\nanalysis_results = {'result': 'success'}"
                result = executor.execute_code(code)
        
        assert result.status == ExecutionStatus.SUCCESS
        assert "Hello, World!" in result.stdout
        assert result.output_data == {'result': 'success'}
        assert result.execution_time > 0
    
    def test_code_execution_with_context(self):
        """Test code execution with user context."""
        limits = ExecutionLimits(max_execution_time=5, max_memory_mb=256, max_output_size_mb=10)
        executor = SecureExecutor(limits)
        
        context = {"data": [1, 2, 3, 4, 5]}
        
        with patch.object(executor.resource_manager, '_set_resource_limits'):
            with patch.object(executor.resource_manager, '_get_current_memory_usage', return_value=50.0):
                code = "result = sum(data)\nanalysis_results = {'sum': result}"
                result = executor.execute_code(code, context)
        
        assert result.status == ExecutionStatus.SUCCESS
        assert result.output_data == {'sum': 15}
    
    def test_execution_with_tracing(self):
        """Test execution with tracing enabled."""
        limits = ExecutionLimits(max_execution_time=5, max_memory_mb=256, max_output_size_mb=10)
        executor = SecureExecutor(limits, enable_tracing=True)
        
        with patch.object(executor.resource_manager, '_set_resource_limits'):
            with patch.object(executor.resource_manager, '_get_current_memory_usage', return_value=50.0):
                code = "x = 1 + 1"
                result = executor.execute_code(code)
        
        assert result.status == ExecutionStatus.SUCCESS
        # Tracing creates additional metadata but returns legacy format
        assert result.execution_time > 0
    
    def test_execution_error_handling(self):
        """Test error handling in execution."""
        limits = ExecutionLimits(max_execution_time=5, max_memory_mb=256, max_output_size_mb=10)
        executor = SecureExecutor(limits)
        
        with patch.object(executor.resource_manager, '_set_resource_limits'):
            code = "x = 1 / 0"  # Division by zero
            result = executor.execute_code(code)
        
        assert result.status == ExecutionStatus.FAILED
        assert "division by zero" in result.error_message.lower()
    
    def test_get_execution_stats(self):
        """Test execution statistics retrieval."""
        limits = ExecutionLimits(max_execution_time=5, max_memory_mb=256, max_output_size_mb=10)
        executor = SecureExecutor(limits, enable_tracing=True)
        
        stats = executor.get_execution_stats()
        
        assert "limits" in stats
        assert stats["limits"]["max_execution_time"] == 5
        assert stats["limits"]["max_memory_mb"] == 256
        assert stats["tracing_enabled"] is True
        assert stats["detailed_tracing_enabled"] is False


@pytest.fixture
def sample_limits():
    """Provide sample execution limits for tests."""
    return ExecutionLimits(
        max_execution_time=10,
        max_memory_mb=512, 
        max_output_size_mb=5
    )


def test_backward_compatibility(sample_limits):
    """Test that the new executor maintains backward compatibility."""
    from infrastructure.execution.executor import SecureExecutor as LegacySecureExecutor
    from infrastructure.execution.secure_executor import SecureExecutor as NewSecureExecutor
    
    # Both should have the same interface
    legacy_executor = LegacySecureExecutor(sample_limits)
    new_executor = NewSecureExecutor(sample_limits)
    
    # Mock to avoid system resource issues in tests
    with patch.object(legacy_executor, '_set_resource_limits'):
        with patch.object(legacy_executor, '_get_memory_usage', return_value=50.0):
            with patch.object(new_executor.resource_manager, '_set_resource_limits'):
                with patch.object(new_executor.resource_manager, '_get_current_memory_usage', return_value=50.0):
                    code = "print('test')\nanalysis_results = {'value': 42}"
                    
                    legacy_result = legacy_executor.execute_code(code)
                    new_result = new_executor.execute_code(code)
                    
                    # Results should have the same structure and values
                    assert legacy_result.status == new_result.status
                    assert legacy_result.output_data == new_result.output_data
                    assert "test" in legacy_result.stdout
                    assert "test" in new_result.stdout