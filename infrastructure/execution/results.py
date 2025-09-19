"""Enhanced execution results with detailed metadata."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from domain.entities import ExecutionStatus
from .tracing import ExecutionTrace
from .resources import ResourceUsage


@dataclass
class ExecutionMetadata:
    """Detailed metadata about code execution."""
    
    # Timing information
    execution_time: float = 0.0
    cpu_time: float = 0.0
    
    # Memory information
    memory_used_mb: float = 0.0
    peak_memory_mb: float = 0.0
    
    # Output information
    stdout_length: int = 0
    stderr_length: int = 0
    output_truncated: bool = False
    
    # Execution context
    context_keys: list[str] = field(default_factory=list)
    result_keys: list[str] = field(default_factory=list)
    
    # Security and safety
    resource_limits_hit: list[str] = field(default_factory=list)
    security_warnings: list[str] = field(default_factory=list)
    
    # Tracing information
    trace_id: Optional[str] = None
    execution_steps_count: int = 0
    memory_snapshots_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "timing": {
                "execution_time": self.execution_time,
                "cpu_time": self.cpu_time
            },
            "memory": {
                "used_mb": self.memory_used_mb,
                "peak_mb": self.peak_memory_mb
            },
            "output": {
                "stdout_length": self.stdout_length,
                "stderr_length": self.stderr_length,
                "truncated": self.output_truncated
            },
            "context": {
                "input_keys": self.context_keys,
                "result_keys": self.result_keys
            },
            "limits": {
                "limits_hit": self.resource_limits_hit,
                "security_warnings": self.security_warnings
            },
            "tracing": {
                "trace_id": self.trace_id,
                "steps_count": self.execution_steps_count,
                "snapshots_count": self.memory_snapshots_count
            }
        }


@dataclass
class EnhancedExecutionResults:
    """Enhanced execution results with comprehensive metadata."""
    
    # Core execution results
    status: ExecutionStatus
    output_data: Optional[Any] = None
    error_message: Optional[str] = None
    stdout: str = ""
    stderr: str = ""
    
    # Enhanced metadata
    metadata: ExecutionMetadata = field(default_factory=ExecutionMetadata)
    
    # Optional detailed information
    execution_trace: Optional[ExecutionTrace] = None
    performance_profile: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_legacy_results(cls, legacy_results, metadata: ExecutionMetadata = None) -> 'EnhancedExecutionResults':
        """
        Create enhanced results from legacy ExecutionResults.
        
        Args:
            legacy_results: Legacy ExecutionResults object
            metadata: Optional metadata to include
            
        Returns:
            EnhancedExecutionResults instance
        """
        if metadata is None:
            metadata = ExecutionMetadata(
                execution_time=getattr(legacy_results, 'execution_time', 0.0),
                memory_used_mb=getattr(legacy_results, 'memory_used_mb', 0.0),
                stdout_length=len(getattr(legacy_results, 'stdout', '')),
                stderr_length=len(getattr(legacy_results, 'stderr', ''))
            )
        
        return cls(
            status=legacy_results.status,
            output_data=legacy_results.output_data,
            error_message=legacy_results.error_message,
            stdout=legacy_results.stdout,
            stderr=legacy_results.stderr,
            metadata=metadata
        )
    
    def to_legacy_results(self):
        """Convert to legacy ExecutionResults format for backward compatibility."""
        from domain.entities import ExecutionResults
        
        return ExecutionResults(
            status=self.status,
            output_data=self.output_data,
            execution_time=self.metadata.execution_time,
            memory_used_mb=self.metadata.memory_used_mb,
            error_message=self.error_message,
            stdout=self.stdout,
            stderr=self.stderr
        )
    
    @property
    def success(self) -> bool:
        """Check if execution was successful."""
        return self.status == ExecutionStatus.SUCCESS
    
    @property
    def failed(self) -> bool:
        """Check if execution failed."""
        return self.status in (ExecutionStatus.FAILED, ExecutionStatus.TIMEOUT)
    
    def add_security_warning(self, warning: str):
        """Add security warning to metadata."""
        self.metadata.security_warnings.append(warning)
    
    def add_resource_limit_hit(self, limit_type: str):
        """Add resource limit hit to metadata."""
        self.metadata.resource_limits_hit.append(limit_type)
    
    def update_from_resource_usage(self, usage: ResourceUsage):
        """Update metadata from resource usage information."""
        self.metadata.execution_time = usage.execution_time
        self.metadata.cpu_time = usage.cpu_time
        self.metadata.memory_used_mb = usage.memory_used_mb
        self.metadata.peak_memory_mb = usage.peak_memory_mb
    
    def update_from_trace(self, trace: ExecutionTrace):
        """Update metadata from execution trace."""
        self.execution_trace = trace
        self.metadata.trace_id = trace.trace_id
        self.metadata.execution_steps_count = len(trace.execution_steps)
        self.metadata.memory_snapshots_count = len(trace.memory_snapshots)
        
        if trace.context_keys:
            self.metadata.context_keys = trace.context_keys
        
        if trace.final_locals:
            self.metadata.result_keys = list(trace.final_locals.keys())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "status": self.status.value,
            "output_data": self.output_data,
            "error_message": self.error_message,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "metadata": self.metadata.to_dict()
        }
        
        if self.execution_trace:
            result["trace"] = self.execution_trace.to_dict()
        
        if self.performance_profile:
            result["performance"] = self.performance_profile
            
        return result