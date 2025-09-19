"""Execution tracing and debugging components."""

import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from infrastructure.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ExecutionTrace:
    """Detailed execution trace information."""
    trace_id: str
    start_time: float
    end_time: Optional[float] = None
    code_hash: Optional[str] = None
    context_keys: List[str] = field(default_factory=list)
    execution_steps: List[Dict[str, Any]] = field(default_factory=list)
    memory_snapshots: List[Tuple[float, float]] = field(default_factory=list)  # (timestamp, memory_mb)
    error_details: Optional[Dict[str, Any]] = None
    final_locals: Optional[Dict[str, Any]] = None
    
    @property
    def duration(self) -> float:
        """Get execution duration."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trace to dictionary for serialization."""
        return {
            "trace_id": self.trace_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "code_hash": self.code_hash,
            "context_keys": self.context_keys,
            "execution_steps": self.execution_steps,
            "memory_snapshots": self.memory_snapshots,
            "error_details": self.error_details,
            "final_locals_keys": list(self.final_locals.keys()) if self.final_locals else []
        }


class ExecutionTracer:
    """Traces code execution for debugging and analysis."""
    
    def __init__(self, enable_detailed_tracing: bool = False):
        """
        Initialize execution tracer.
        
        Args:
            enable_detailed_tracing: Whether to enable detailed line-by-line tracing
        """
        self.enable_detailed_tracing = enable_detailed_tracing
        self.current_trace: Optional[ExecutionTrace] = None
        self._original_trace_function = None
        
    def start_trace(self, code: str, context: Dict[str, Any] = None) -> ExecutionTrace:
        """
        Start execution tracing.
        
        Args:
            code: Code being executed
            context: Execution context
            
        Returns:
            ExecutionTrace object
        """
        trace_id = str(uuid4())[:8]
        self.current_trace = ExecutionTrace(
            trace_id=trace_id,
            start_time=time.time(),
            code_hash=str(hash(code)),
            context_keys=list(context.keys()) if context else []
        )
        
        if self.enable_detailed_tracing:
            self._setup_line_tracing()
            
        logger.debug(f"Started execution trace {trace_id}")
        return self.current_trace
    
    def end_trace(self, success: bool = True, error: Optional[Exception] = None, 
                  final_locals: Optional[Dict[str, Any]] = None) -> ExecutionTrace:
        """
        End execution tracing.
        
        Args:
            success: Whether execution was successful
            error: Exception if execution failed
            final_locals: Final local variables
            
        Returns:
            Completed ExecutionTrace object
        """
        if not self.current_trace:
            logger.warning("No active trace to end")
            return None
            
        self.current_trace.end_time = time.time()
        
        if error:
            self.current_trace.error_details = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc()
            }
        
        if final_locals:
            # Store keys and types only for safety
            safe_locals = {}
            for key, value in final_locals.items():
                try:
                    safe_locals[key] = {
                        "type": type(value).__name__,
                        "repr": repr(value)[:100] if not callable(value) else "callable"
                    }
                except Exception:
                    safe_locals[key] = {"type": "unknown", "repr": "error"}
            self.current_trace.final_locals = safe_locals
        
        self._cleanup_line_tracing()
        
        logger.debug(f"Ended execution trace {self.current_trace.trace_id}, "
                    f"duration: {self.current_trace.duration:.3f}s")
        
        completed_trace = self.current_trace
        self.current_trace = None
        return completed_trace
    
    def add_memory_snapshot(self, memory_mb: float):
        """Add memory usage snapshot to current trace."""
        if self.current_trace:
            timestamp = time.time()
            self.current_trace.memory_snapshots.append((timestamp, memory_mb))
    
    def add_execution_step(self, step_type: str, details: Dict[str, Any]):
        """Add execution step to current trace."""
        if self.current_trace:
            step = {
                "timestamp": time.time(),
                "type": step_type,
                "details": details
            }
            self.current_trace.execution_steps.append(step)
    
    def _setup_line_tracing(self):
        """Set up line-by-line execution tracing."""
        if not self.enable_detailed_tracing:
            return
            
        self._original_trace_function = sys.gettrace()
        sys.settrace(self._trace_line)
    
    def _cleanup_line_tracing(self):
        """Clean up line-by-line tracing."""
        if self._original_trace_function is not None:
            sys.settrace(self._original_trace_function)
            self._original_trace_function = None
        else:
            sys.settrace(None)
    
    def _trace_line(self, frame, event, arg):
        """Line tracing function."""
        if not self.current_trace or event not in ('line', 'call', 'return', 'exception'):
            return self._trace_line
        
        try:
            filename = frame.f_code.co_filename
            # Only trace our executed code, not library code
            if '<string>' in filename or 'exec' in filename:
                details = {
                    "event": event,
                    "filename": filename,
                    "line_number": frame.f_lineno,
                    "function_name": frame.f_code.co_name
                }
                
                if event == 'exception' and arg:
                    details["exception"] = {
                        "type": type(arg[1]).__name__ if arg[1] else "unknown",
                        "message": str(arg[1]) if arg[1] else "unknown"
                    }
                
                self.add_execution_step("line_trace", details)
        
        except Exception as e:
            # Don't let tracing errors break execution
            logger.debug(f"Tracing error: {e}")
        
        return self._trace_line


class ExecutionProfiler:
    """Profiles code execution performance."""
    
    def __init__(self):
        """Initialize execution profiler."""
        self.profiles: Dict[str, Dict[str, Any]] = {}
    
    def profile_execution(self, trace: ExecutionTrace) -> Dict[str, Any]:
        """
        Create performance profile from execution trace.
        
        Args:
            trace: Execution trace to profile
            
        Returns:
            Performance profile dictionary
        """
        profile = {
            "trace_id": trace.trace_id,
            "total_duration": trace.duration,
            "memory_profile": self._analyze_memory_usage(trace.memory_snapshots),
            "execution_profile": self._analyze_execution_steps(trace.execution_steps),
            "error_analysis": trace.error_details
        }
        
        self.profiles[trace.trace_id] = profile
        return profile
    
    def _analyze_memory_usage(self, snapshots: List[Tuple[float, float]]) -> Dict[str, Any]:
        """Analyze memory usage patterns."""
        if not snapshots:
            return {"peak_memory": 0, "memory_growth": 0, "snapshots_count": 0}
        
        memories = [mem for _, mem in snapshots]
        return {
            "peak_memory": max(memories),
            "initial_memory": memories[0],
            "final_memory": memories[-1],
            "memory_growth": memories[-1] - memories[0],
            "snapshots_count": len(snapshots),
            "average_memory": sum(memories) / len(memories)
        }
    
    def _analyze_execution_steps(self, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze execution step patterns."""
        if not steps:
            return {"total_steps": 0, "step_types": {}}
        
        step_types = {}
        for step in steps:
            step_type = step.get("type", "unknown")
            step_types[step_type] = step_types.get(step_type, 0) + 1
        
        return {
            "total_steps": len(steps),
            "step_types": step_types,
            "first_step_time": steps[0].get("timestamp") if steps else None,
            "last_step_time": steps[-1].get("timestamp") if steps else None
        }