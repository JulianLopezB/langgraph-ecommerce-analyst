"""State management for the LangGraph Data Analysis Agent."""
from typing import Optional, List, Dict, Any, TypedDict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import pandas as pd


class ProcessType(Enum):
    """Simplified process types for data analysis."""
    SQL = "sql"              # All data retrieval, aggregation, and business analytics
    PYTHON = "python"        # Complex analytics, ML, statistical analysis
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


class AnalysisState(TypedDict, total=False):
    """Main state object for the LangGraph agent."""
    # Input Processing
    user_query: str
    process_type: ProcessType
    confidence_score: float

    # Data Layer
    sql_query: str
    raw_dataset: Optional[pd.DataFrame]
    data_schema: Dict[str, Any]

    # Code Generation
    generated_code: Optional[GeneratedCode]
    validation_results: Optional[ValidationResults]
    needs_python_analysis: bool

    # Execution Results
    execution_results: Optional[ExecutionResults]
    analysis_outputs: Dict[str, Any]
    insights: str

    # Error Handling
    error_context: Dict[str, Any]

    # Session Management
    session_id: str
    conversation_history: List[ConversationMessage]

    # Control Flow
    next_step: str
    workflow_complete: bool


def create_initial_state(user_query: str, session_id: str) -> AnalysisState:
    """Create initial state for a new analysis request."""
    return AnalysisState(
        user_query=user_query,
        process_type=ProcessType.SQL,
        confidence_score=0.0,
        sql_query="",
        raw_dataset=None,
        data_schema={},
        generated_code=None,
        validation_results=None,
        needs_python_analysis=False,
        execution_results=None,
        analysis_outputs={},
        insights="",
        error_context={},
        session_id=session_id,
        conversation_history=[],
        next_step="understand_query",
        workflow_complete=False
    )
