"""State management for the LangGraph Data Analysis Agent."""
from typing import Optional, List, Dict, Any, TypedDict

import pandas as pd

from domain.entities import (
    ProcessType,
    GeneratedCode,
    ValidationResults,
    ExecutionResults,
    ConversationMessage,
)


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


def create_initial_state(
    user_query: str,
    session_id: str,
    conversation_history: Optional[List[ConversationMessage]] = None,
) -> AnalysisState:
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
        conversation_history=list(conversation_history) if conversation_history else [],
        next_step="understand_query",
        workflow_complete=False
    )
