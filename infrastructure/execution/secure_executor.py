"""Redesigned secure executor with clean separation of concerns."""

import io
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Dict, Optional

from domain.entities import ExecutionResults, ExecutionStatus
from infrastructure.config import ExecutionLimits
from infrastructure.logging import get_logger

from .base import CodeExecutor
from .resources import MemoryLimitException, OutputManager, ResourceManager, TimeoutException
from .results import EnhancedExecutionResults, ExecutionMetadata
from .security import ContextIsolator
from .tracing import ExecutionProfiler, ExecutionTracer

logger = get_logger(__name__)


class SecureExecutor(CodeExecutor):
    """
    Redesigned secure Python code executor with clean separation of concerns.
    
    This executor separates:
    - Security: Context isolation and safe environment creation
    - Resources: Memory, timeout, and output management  
    - Execution: Core code execution logic
    - Tracing: Debugging and performance monitoring
    """
    
    def __init__(self, execution_limits: ExecutionLimits, enable_tracing: bool = False, 
                 enable_detailed_tracing: bool = False):
        """
        Initialize the secure executor.
        
        Args:
            execution_limits: Resource limits configuration
            enable_tracing: Whether to enable execution tracing
            enable_detailed_tracing: Whether to enable detailed line-by-line tracing
        """
        self.limits = execution_limits
        
        # Initialize components
        self.context_isolator = ContextIsolator()
        self.resource_manager = ResourceManager(execution_limits)
        self.output_manager = OutputManager(execution_limits.max_output_size_mb)
        
        # Optional tracing components
        self.tracer = ExecutionTracer(enable_detailed_tracing) if enable_tracing else None
        self.profiler = ExecutionProfiler() if enable_tracing else None
        
        logger.info(f"SecureExecutor initialized with limits: "
                   f"time={execution_limits.max_execution_time}s, "
                   f"memory={execution_limits.max_memory_mb}MB, "
                   f"output={execution_limits.max_output_size_mb}MB, "
                   f"tracing={'enabled' if enable_tracing else 'disabled'}")
    
    def execute_code(self, code: str, context: Dict[str, Any] = None) -> ExecutionResults:
        """
        Execute Python code in a secure environment.
        
        Args:
            code: Python code to execute
            context: Context variables to inject (e.g., DataFrame)
            
        Returns:
            ExecutionResults with execution details
        """
        # Create enhanced results object
        results = EnhancedExecutionResults(status=ExecutionStatus.PENDING)
        
        # Start tracing if enabled
        trace = None
        if self.tracer:
            trace = self.tracer.start_trace(code, context)
            results.metadata.trace_id = trace.trace_id
        
        try:
            # Prepare execution context
            safe_globals, safe_locals = self.context_isolator.prepare_execution_context(context)
            
            if context:
                results.metadata.context_keys = list(context.keys())
            
            # Execute with resource management
            with self.resource_manager.managed_execution() as usage:
                results = self._execute_with_monitoring(
                    code, safe_globals, safe_locals, results, usage
                )
            
            # Update results with resource usage
            results.update_from_resource_usage(usage)
            
            # Set success status if no errors occurred
            if results.status == ExecutionStatus.PENDING:
                results.status = ExecutionStatus.SUCCESS
                
        except TimeoutException as e:
            results.status = ExecutionStatus.TIMEOUT
            results.error_message = str(e)
            results.add_resource_limit_hit("timeout")
            logger.warning(f"Execution timeout: {e}")
            
        except MemoryLimitException as e:
            results.status = ExecutionStatus.FAILED
            results.error_message = str(e)
            results.add_resource_limit_hit("memory")
            logger.warning(f"Memory limit exceeded: {e}")
            
        except Exception as e:
            results.status = ExecutionStatus.FAILED
            results.error_message = f"Code execution failed: {str(e)}"
            logger.error(f"Execution error: {e}", exc_info=True)
            
        finally:
            # End tracing and create profile
            if self.tracer and trace:
                completed_trace = self.tracer.end_trace(
                    success=results.success,
                    error=None if results.success else Exception(results.error_message),
                    final_locals=safe_locals if 'safe_locals' in locals() else None
                )
                results.update_from_trace(completed_trace)
                
                if self.profiler:
                    profile = self.profiler.profile_execution(completed_trace)
                    results.performance_profile = profile
        
        # Log execution summary
        self._log_execution_summary(results)
        
        # Return legacy format for backward compatibility
        return results.to_legacy_results()
    
    def _execute_with_monitoring(self, code: str, safe_globals: Dict[str, Any], 
                                safe_locals: Dict[str, Any], results: EnhancedExecutionResults,
                                usage) -> EnhancedExecutionResults:
        """
        Execute code with output capture and monitoring.
        
        Args:
            code: Code to execute
            safe_globals: Safe global environment
            safe_locals: Safe local environment  
            results: Results object to update
            usage: Resource usage tracker
            
        Returns:
            Updated results object
        """
        # Capture output
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Add memory snapshot before execution
                if self.tracer:
                    initial_memory = self.resource_manager._get_current_memory_usage()
                    self.tracer.add_memory_snapshot(initial_memory)
                    self.tracer.add_execution_step("execution_start", {"code_length": len(code)})
                
                # Execute the code
                exec(code, safe_globals, safe_locals)
                
                # Add memory snapshot after execution
                if self.tracer:
                    final_memory = self.resource_manager._get_current_memory_usage()
                    self.tracer.add_memory_snapshot(final_memory)
                    self.tracer.add_execution_step("execution_complete", {
                        "locals_count": len(safe_locals),
                        "memory_delta": final_memory - initial_memory
                    })
            
            # Process outputs
            stdout_content = stdout_capture.getvalue()
            stderr_content = stderr_capture.getvalue()
            
            # Apply output size limits
            limited_stdout, limited_stderr = self.output_manager.limit_output_size(
                stdout_content, stderr_content
            )
            
            # Check if output was truncated
            output_truncated = (len(limited_stdout) < len(stdout_content) or 
                              len(limited_stderr) < len(stderr_content))
            
            # Update results
            results.stdout = limited_stdout
            results.stderr = limited_stderr
            results.metadata.stdout_length = len(stdout_content)
            results.metadata.stderr_length = len(stderr_content)
            results.metadata.output_truncated = output_truncated
            
            # Extract analysis results if available
            results.output_data = safe_locals.get("analysis_results", {})
            
            # Update result keys in metadata
            results.metadata.result_keys = [k for k in safe_locals.keys() if not k.startswith('_')]
            
            if output_truncated:
                results.add_security_warning("Output size exceeded limit and was truncated")
            
        except Exception as e:
            # Capture any output generated before the error
            results.stdout = stdout_capture.getvalue()
            results.stderr = stderr_capture.getvalue()
            raise  # Re-raise to be handled by calling method
        
        return results
    
    def _log_execution_summary(self, results: EnhancedExecutionResults):
        """Log execution summary."""
        status_emoji = "✅" if results.success else "❌"
        
        log_msg = (f"{status_emoji} Execution {results.metadata.trace_id or 'unknown'}: "
                  f"status={results.status.value}, "
                  f"time={results.metadata.execution_time:.3f}s, "
                  f"memory={results.metadata.memory_used_mb:.1f}MB")
        
        if results.metadata.peak_memory_mb > results.metadata.memory_used_mb:
            log_msg += f" (peak: {results.metadata.peak_memory_mb:.1f}MB)"
        
        if results.metadata.output_truncated:
            log_msg += ", output_truncated=True"
        
        if results.metadata.resource_limits_hit:
            log_msg += f", limits_hit={results.metadata.resource_limits_hit}"
        
        if results.success:
            logger.info(log_msg)
        else:
            logger.warning(f"{log_msg}, error={results.error_message}")
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """
        Get execution statistics.
        
        Returns:
            Dictionary with execution statistics
        """
        stats = {
            "limits": {
                "max_execution_time": self.limits.max_execution_time,
                "max_memory_mb": self.limits.max_memory_mb,
                "max_output_size_mb": self.limits.max_output_size_mb
            },
            "tracing_enabled": self.tracer is not None,
            "detailed_tracing_enabled": (self.tracer.enable_detailed_tracing 
                                       if self.tracer else False)
        }
        
        if self.profiler:
            stats["profiles_count"] = len(self.profiler.profiles)
            stats["recent_profiles"] = list(self.profiler.profiles.keys())[-5:]  # Last 5
        
        return stats