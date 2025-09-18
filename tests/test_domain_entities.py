import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from domain.entities import (
    ConversationMessage,
    ExecutionResults,
    ExecutionStatus,
    GeneratedCode,
    ValidationResults,
)


def test_generated_code_defaults():
    code = GeneratedCode("print('hi')")
    assert code.language == "python"
    assert code.parameters == {}
    assert code.validation_passed is False
    assert code.security_score == 0.0


def test_validation_results_defaults():
    results = ValidationResults(is_valid=True)
    assert results.syntax_errors == []
    assert results.security_warnings == []
    assert results.performance_warnings == []
    assert results.validation_time == 0.0


def test_execution_results_defaults():
    result = ExecutionResults(status=ExecutionStatus.SUCCESS)
    assert result.output_data is None
    assert result.execution_time == 0.0
    assert result.memory_used_mb == 0.0
    assert result.stdout == ""
    assert result.stderr == ""


def test_conversation_message_defaults():
    msg = ConversationMessage(timestamp=datetime.now(), role="user", content="hi")
    assert msg.message_type == "text"
