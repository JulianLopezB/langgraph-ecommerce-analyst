"""Resource management components for code execution."""

import resource
import signal
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

from infrastructure.config import ExecutionLimits
from infrastructure.logging import get_logger

logger = get_logger(__name__)


class ResourceLimitExceededException(Exception):
    """Exception raised when resource limits are exceeded."""
    pass


class TimeoutException(ResourceLimitExceededException):
    """Exception raised when execution times out."""
    pass


class MemoryLimitException(ResourceLimitExceededException):
    """Exception raised when memory limit is exceeded."""
    pass


@dataclass
class ResourceUsage:
    """Resource usage metrics."""
    memory_used_mb: float
    execution_time: float
    cpu_time: float
    peak_memory_mb: float


class ResourceManager:
    """Manages resource limits and monitoring for code execution."""
    
    def __init__(self, limits: ExecutionLimits):
        """
        Initialize resource manager.
        
        Args:
            limits: Execution limits configuration
        """
        self.limits = limits
        self._original_limits = {}
        self._start_time = None
        self._peak_memory = 0.0
        
    @contextmanager
    def managed_execution(self):
        """
        Context manager for resource-controlled execution.
        
        Yields:
            ResourceUsage object that gets populated during execution
        """
        usage = ResourceUsage(0.0, 0.0, 0.0, 0.0)
        
        # Store original limits
        self._store_original_limits()
        
        try:
            # Set resource limits
            self._set_resource_limits()
            
            # Set up timeout handler
            original_handler = signal.signal(signal.SIGALRM, self._timeout_handler)
            signal.alarm(self.limits.max_execution_time)
            
            # Start timing
            self._start_time = time.time()
            start_cpu_time = time.process_time()
            
            yield usage
            
            # Calculate final usage
            usage.execution_time = time.time() - self._start_time
            usage.cpu_time = time.process_time() - start_cpu_time
            usage.memory_used_mb = self._get_current_memory_usage()
            usage.peak_memory_mb = max(self._peak_memory, usage.memory_used_mb)
            
        except TimeoutException:
            if self._start_time:
                usage.execution_time = time.time() - self._start_time
            raise
        except MemoryError:
            if self._start_time:
                usage.execution_time = time.time() - self._start_time
            usage.memory_used_mb = self._get_current_memory_usage()
            raise MemoryLimitException(f"Memory limit of {self.limits.max_memory_mb}MB exceeded")
        finally:
            # Clean up
            signal.alarm(0)  # Cancel timeout
            if 'original_handler' in locals():
                signal.signal(signal.SIGALRM, original_handler)
            self._restore_original_limits()
    
    def _timeout_handler(self, signum, frame):
        """Signal handler for execution timeout."""
        raise TimeoutException(f"Code execution timed out after {self.limits.max_execution_time} seconds")
    
    def _set_resource_limits(self):
        """Set system resource limits."""
        try:
            # Set memory limit (virtual memory)
            memory_limit_bytes = self.limits.max_memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))
            
            # Set CPU time limit
            cpu_limit = self.limits.max_execution_time
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit + 5))  # Small buffer for cleanup
            
            logger.debug(f"Set resource limits: memory={self.limits.max_memory_mb}MB, cpu={cpu_limit}s")
            
        except (OSError, ValueError) as e:
            # Log warning but don't fail - limits might not be supported on all systems
            logger.warning(f"Could not set resource limits: {e}")
    
    def _store_original_limits(self):
        """Store original resource limits for restoration."""
        try:
            self._original_limits = {
                'memory': resource.getrlimit(resource.RLIMIT_AS),
                'cpu': resource.getrlimit(resource.RLIMIT_CPU)
            }
        except (OSError, ValueError) as e:
            logger.debug(f"Could not store original limits: {e}")
    
    def _restore_original_limits(self):
        """Restore original resource limits."""
        try:
            if 'memory' in self._original_limits:
                resource.setrlimit(resource.RLIMIT_AS, self._original_limits['memory'])
            if 'cpu' in self._original_limits:
                resource.setrlimit(resource.RLIMIT_CPU, self._original_limits['cpu'])
        except (OSError, ValueError) as e:
            logger.debug(f"Could not restore original limits: {e}")
    
    def _get_current_memory_usage(self) -> float:
        """
        Get current memory usage in MB.
        
        Returns:
            Current memory usage in MB
        """
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            
            # Track peak memory
            self._peak_memory = max(self._peak_memory, memory_mb)
            
            return memory_mb
        except ImportError:
            logger.debug("psutil not available for memory monitoring")
            return 0.0
        except Exception as e:
            logger.warning(f"Could not get memory usage: {e}")
            return 0.0


class OutputManager:
    """Manages output capture and size limits."""
    
    def __init__(self, max_output_size_mb: int):
        """
        Initialize output manager.
        
        Args:
            max_output_size_mb: Maximum output size in MB
        """
        self.max_output_size_mb = max_output_size_mb
        self.max_output_bytes = max_output_size_mb * 1024 * 1024
    
    def limit_output_size(self, stdout: str, stderr: str) -> tuple[str, str]:
        """
        Limit output size and truncate if necessary.
        
        Args:
            stdout: Standard output content
            stderr: Standard error content
            
        Returns:
            Tuple of (limited_stdout, limited_stderr)
        """
        total_size = len(stdout) + len(stderr)
        
        if total_size <= self.max_output_bytes:
            return stdout, stderr
        
        # Truncate proportionally
        stdout_ratio = len(stdout) / total_size if total_size > 0 else 0.5
        stderr_ratio = len(stderr) / total_size if total_size > 0 else 0.5
        
        stdout_limit = int(self.max_output_bytes * stdout_ratio)
        stderr_limit = int(self.max_output_bytes * stderr_ratio)
        
        # Ensure we don't exceed the limit
        if stdout_limit + stderr_limit > self.max_output_bytes:
            stderr_limit = self.max_output_bytes - stdout_limit
        
        limited_stdout = stdout[:stdout_limit] + "\n[OUTPUT TRUNCATED]" if len(stdout) > stdout_limit else stdout
        limited_stderr = stderr[:stderr_limit] + "\n[OUTPUT TRUNCATED]" if len(stderr) > stderr_limit else stderr
        
        logger.warning(f"Output truncated: {total_size} bytes -> {len(limited_stdout) + len(limited_stderr)} bytes")
        
        return limited_stdout, limited_stderr