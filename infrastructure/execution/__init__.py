"""Execution infrastructure bindings."""

from .base import CodeExecutor
from .executor import SecureExecutor as LegacySecureExecutor
from .secure_executor import SecureExecutor
from .validator import CodeValidator, ValidationResult

# New components
from .resources import ResourceManager, OutputManager, ResourceUsage
from .security import SecurityIsolator, ContextIsolator
from .tracing import ExecutionTracer, ExecutionProfiler, ExecutionTrace
from .results import EnhancedExecutionResults, ExecutionMetadata

# Default bindings used by application; injected at runtime
secure_executor: CodeExecutor | None = None
validator: CodeValidator | None = None

# For backward compatibility, keep the legacy executor available
__all__ = [
    "CodeExecutor",
    "secure_executor",
    "validator",
    "SecureExecutor",
    "LegacySecureExecutor",
    "CodeValidator",
    "ValidationResult",
    # New components
    "ResourceManager",
    "OutputManager", 
    "ResourceUsage",
    "SecurityIsolator",
    "ContextIsolator",
    "ExecutionTracer",
    "ExecutionProfiler", 
    "ExecutionTrace",
    "EnhancedExecutionResults",
    "ExecutionMetadata",
]
