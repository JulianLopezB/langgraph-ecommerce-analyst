from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any


class ProcessType(Enum):
    """Simplified process types for data analysis."""

    SQL = "sql"  # All data retrieval, aggregation, and business analytics
    PYTHON = "python"  # Complex analytics, ML, statistical analysis
    VISUALIZATION = "visualization"  # Charts, plots, and visual representations


class ExecutionStatus(Enum):
    """Status of code execution."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class GeneratedCode:
    """Information about generated Python code."""

    code_content: str
    language: str = "python"
    template_used: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    validation_passed: bool = False
    security_score: float = 0.0


@dataclass
class ValidationResults:
    """Results from code validation pipeline."""

    is_valid: bool
    syntax_errors: List[str] = field(default_factory=list)
    security_warnings: List[str] = field(default_factory=list)
    performance_warnings: List[str] = field(default_factory=list)
    validation_time: float = 0.0


@dataclass
class ExecutionResults:
    """Results from code execution."""

    status: ExecutionStatus
    output_data: Optional[Any] = None
    execution_time: float = 0.0
    memory_used_mb: float = 0.0
    error_message: Optional[str] = None
    stdout: str = ""
    stderr: str = ""


@dataclass
class ConversationMessage:
    """Individual message in conversation history."""

    timestamp: datetime
    role: str  # 'user' or 'assistant'
    content: str
    message_type: str = "text"  # 'text', 'query', 'result', 'error'


@dataclass
class AnalysisSession:
    """Represents an analysis conversation session."""

    session_id: str
    created_at: datetime
    conversation_history: List[ConversationMessage] = field(default_factory=list)
    analysis_count: int = 0
    artifacts: Dict[str, Any] = field(default_factory=dict)
