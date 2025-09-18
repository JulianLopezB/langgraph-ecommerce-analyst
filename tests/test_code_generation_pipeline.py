"""Tests for the structured code generation pipeline."""
import pytest
from unittest.mock import Mock, MagicMock, patch
import pandas as pd
from typing import Dict, Any

from domain.pipeline import (
    CodeGenerationPipeline,
    create_code_generation_pipeline,
    PipelineContext,
    PipelineStatus
)
from domain.pipeline.stages import (
    CodeGenerationStage,
    CodeCleaningStage,
    CodeValidationStage,
    CodeExecutionStage
)
from domain.entities import GeneratedCode, ExecutionStatus, ExecutionResults
from infrastructure.execution.validator import ValidationResult
from infrastructure.llm.base import LLMClient
from infrastructure.execution.validator import CodeValidator
from infrastructure.execution.executor import SecureExecutor


class TestPipelineContext:
    """Test pipeline context functionality."""

    def test_context_creation(self):
        """Test creating pipeline context with required fields."""
        context = PipelineContext(
            user_query="Test query",
            analysis_context={"test": "data"}
        )
     assert context.user_query == "Test query"
        assert context.analysis_context == {"test": "data"}
        assert context.code_content is None
        assert context.current_stage is None
        assert len(context.error_context) == 0
        assert len(context.metrics) == 0

    def test_context_metadata_updates(self):
        """Test that context metadata is properly updated."""
        context = PipelineContext(
            user_query="Test query",
            analysis_context={"test": "data"}
        )
     # Simulate stage metadata updates
        context.stage_metadata["generation"] = {
            "execution_time": 1.5,
            "success": True,
            "code_length": 100
        }
     context.metrics["generation_time"] = 1.5
        context.error_context["test_error"] = "Test error message"
     assert context.stage_metadata["generation"]["execution_time"] == 1.5
        assert context.metrics["generation_time"] == 1.5
        assert context.error_context["test_error"] == "Test error message"


class TestCodeGenerationStage:
    """Test code generation stage."""

    @pytest.fixture
    def mock_llm_client(self):
        """Mock LLM client."""
        client = Mock(spec=LLMClient)
        client.generate_adaptive_python_code.return_value = "print('Hello World')"
        return client

    @pytest.fixture
    def generation_stage(self, mock_llm_client):
        """Create code generation stage."""
        return CodeGenerationStage(mock_llm_client)

    def test_successful_code_generation(self, generation_stage, mock_llm_client):
        """Test successful code generation."""
        context = PipelineContext(
            user_query="Generate hello world",
            analysis_context={
                "process_data": {"process_type": "python"},
                "data_characteristics": {"shape": (100, 5)}
            }
        )
     result = generation_stage.execute(context)
     assert result.success is True
        assert result.data is not None
        assert isinstance(result.data, GeneratedCode)
        assert result.data.code_content == "print('Hello World')"
        assert result.data.template_used == "python"
        assert context.code_content == "print('Hello World')"
     # Check metrics
        assert "code_length" in result.stage_metrics
        assert result.stage_metrics["code_length"] == len("print('Hello World')")
     mock_llm_client.generate_adaptive_python_code.assert_called_once()

    def test_llm_failure_handling(self, generation_stage, mock_llm_client):
        """Test handling of LLM failures."""
        mock_llm_client.generate_adaptive_python_code.side_effect = Exception("LLM API Error")
     context = PipelineContext(
            user_query="Generate code",
            analysis_context={"process_data": {"process_type": "python"}}
        )
     result = generation_stage.execute(context)
     assert result.success is False
        assert "Code generation failed" in result.error_message
        assert "LLM API Error" in result.error_context["llm_error"]

    def test_input_validation(self, generation_stage):
        """Test input validation."""
        # Missing user_query
        context = PipelineContext(
            user_query="",
            analysis_context={"test": "data"}
        )
     result = generation_stage.execute(context)
        assert result.success is False
        assert "Missing user_query" in result.error_message
     # Missing analysis_context
        context = PipelineContext(
            user_query="Test query",
            analysis_context={}
        )
     result = generation_stage.execute(context)
        assert result.success is False
        assert "Missing analysis_context" in result.error_message


class TestCodeCleaningStage:
    """Test code cleaning stage."""

    @pytest.fixture
    def cleaning_stage(self):
        """Create code cleaning stage."""
        return CodeCleaningStage()

    def test_successful_code_cleaning(self, cleaning_stage):
        """Test successful code cleaning."""
        dirty_code = """```python
# This is a comment
import pandas as pd

def analyze_data():
    return "result"
```"""
     context = PipelineContext(
            user_query="Test",
            analysis_context={},
            code_content=dirty_code
        )
     result = cleaning_stage.execute(context)
     assert result.success is True
        assert "```python" not in result.data
        assert "```" not in result.data
        assert "import pandas as pd" in result.data
        assert context.cleaned_code == result.data
     # Check metrics
        assert "original_length" in result.stage_metrics
        assert "cleaned_length" in result.stage_metrics
        assert "markdown_blocks_removed" in result.stage_metrics

    def test_input_validation(self, cleaning_stage):
        """Test input validation for cleaning stage."""
        context = PipelineContext(
            user_query="Test",
            analysis_context={},
            code_content=None
        )
     result = cleaning_stage.execute(context)
        assert result.success is False
        assert "No code content to clean" in result.error_message


class TestCodeValidationStage:
    """Test code validation stage."""

    @pytest.fixture
    def mock_validator(self):
        """Mock code validator."""
        validator = Mock(spec=CodeValidator)
        validator.validate.return_value = ValidationResult(
            is_valid=True,
            syntax_errors=[],
            security_warnings=[],
            performance_warnings=[],
            validation_time=0.1,
            security_score=1.0
        )
        return validator

    @pytest.fixture
    def validation_stage(self, mock_validator):
        """Create validation stage."""
        return CodeValidationStage(mock_validator)

    def test_successful_validation(self, validation_stage, mock_validator):
        """Test successful code validation."""
        context = PipelineContext(
            user_query="Test",
            analysis_context={},
            cleaned_code="print('Hello')"
        )
     result = validation_stage.execute(context)
     assert result.success is True
        assert result.data.is_valid is True
        assert context.validation_results == result.data
     # Check metrics
        assert result.stage_metrics["is_valid"] is True
        assert result.stage_metrics["security_score"] == 1.0
        assert result.stage_metrics["syntax_errors_count"] == 0
     mock_validator.validate.assert_called_once_with("print('Hello')")

    def test_validation_failure(self, validation_stage, mock_validator):
        """Test validation failure handling."""
        mock_validator.validate.return_value = ValidationResult(
            is_valid=False,
            syntax_errors=["Syntax error at line 1"],
            security_warnings=["Dangerous function detected"],
            performance_warnings=[],
            validation_time=0.1,
            security_score=0.3
        )
     context = PipelineContext(
            user_query="Test",
            analysis_context={},
            cleaned_code="eval('malicious code')"
        )
     result = validation_stage.execute(context)
     assert result.success is False
        assert "Code validation failed" in result.error_message
        assert "Syntax error at line 1" in result.error_context["syntax_errors"]
        assert "Dangerous function detected" in result.error_context["security_warnings"]


class TestCodeExecutionStage:
    """Test code execution stage."""

    @pytest.fixture
    def mock_executor(self):
        """Mock secure executor."""
        executor = Mock(spec=SecureExecutor)
        executor.execute_code.return_value = ExecutionResults(
            status=ExecutionStatus.SUCCESS,
            output_data={"result": "success"},
            execution_time=1.0,
            memory_used_mb=50.0,
            stdout="Output message",
            stderr=""
        )
        return executor

    @pytest.fixture
    def execution_stage(self, mock_executor):
        """Create execution stage."""
        return CodeExecutionStage(mock_executor)

    def test_successful_execution(self, execution_stage, mock_executor):
        """Test successful code execution."""
        validation_result = ValidationResult(
            is_valid=True,
            syntax_errors=[],
            security_warnings=[],
            performance_warnings=[],
            validation_time=0.1,
            security_score=1.0
        )
     context = PipelineContext(
            user_query="Test",
            analysis_context={"raw_dataset": pd.DataFrame({"a": [1, 2, 3]})},
            cleaned_code="print('Hello')",
            validation_results=validation_result
        )
     result = execution_stage.execute(context)
     assert result.success is True
        assert result.data.status == ExecutionStatus.SUCCESS
        assert result.data.output_data == {"result": "success"}
        assert context.execution_results == result.data
     # Check metrics
        assert result.stage_metrics["execution_status"] == "success"
        assert result.stage_metrics["execution_time"] == 1.0
        assert result.stage_metrics["memory_used_mb"] == 50.0
        assert result.stage_metrics["has_output_data"] is True

    def test_execution_failure(self, execution_stage, mock_executor):
        """Test execution failure handling."""
        mock_executor.execute_code.return_value = ExecutionResults(
            status=ExecutionStatus.FAILED,
            error_message="Runtime error occurred",
            execution_time=0.5,
            memory_used_mb=25.0,
            stdout="",
            stderr="Error details"
        )
     validation_result = ValidationResult(
            is_valid=True,
            syntax_errors=[],
            security_warnings=[],
            performance_warnings=[],
            validation_time=0.1,
            security_score=1.0
        )
     context = PipelineContext(
            user_query="Test",
            analysis_context={"raw_dataset": pd.DataFrame()},
            cleaned_code="raise Exception('test')",
            validation_results=validation_result
        )
     result = execution_stage.execute(context)
     assert result.success is False
        assert "Runtime error occurred" in result.error_message
        assert result.error_context["execution_status"] == "failed"

    def test_input_validation(self, execution_stage):
        """Test input validation for execution stage."""
        # No cleaned code
        context = PipelineContext(
            user_query="Test",
            analysis_context={},
            cleaned_code=None
        )
     result = execution_stage.execute(context)
        assert result.success is False
        assert "No cleaned code to execute" in result.error_message
     # No validation results
        context = PipelineContext(
            user_query="Test",
            analysis_context={},
            cleaned_code="print('test')",
            validation_results=None
        )
     result = execution_stage.execute(context)
        assert result.success is False
        assert "Code has not been validated" in result.error_message


class TestCodeGenerationPipeline:
    """Test the complete code generation pipeline."""

    @pytest.fixture
    def mock_llm_client(self):
        """Mock LLM client."""
        client = Mock(spec=LLMClient)
        client.generate_adaptive_python_code.return_value = "import pandas as pd\nprint('Analysis complete')"
        return client

    @pytest.fixture
    def mock_validator(self):
        """Mock validator."""
        validator = Mock(spec=CodeValidator)
        validator.validate.return_value = ValidationResult(
            is_valid=True,
            syntax_errors=[],
            security_warnings=[],
            performance_warnings=[],
            validation_time=0.1,
            security_score=1.0
        )
        return validator

    @pytest.fixture
    def mock_executor(self):
        """Mock executor."""
        executor = Mock(spec=SecureExecutor)
        executor.execute_code.return_value = ExecutionResults(
            status=ExecutionStatus.SUCCESS,
            output_data={"insights": "Data analysis completed"},
            execution_time=2.0,
            memory_used_mb=75.0,
            stdout="Analysis results printed",
            stderr=""
        )
        return executor

    @pytest.fixture
    def pipeline(self, mock_llm_client, mock_validator, mock_executor):
        """Create complete pipeline."""
        return CodeGenerationPipeline(mock_llm_client, mock_validator, mock_executor)

    def test_successful_pipeline_execution(self, pipeline):
        """Test successful end-to-end pipeline execution."""
        result = pipeline.generate_and_execute_code(
            user_query="Analyze the sales data",
            analysis_context={
                "process_data": {"process_type": "python"},
                "data_characteristics": {"shape": (1000, 10)},
                "raw_dataset": pd.DataFrame({"sales": [100, 200, 300]})
            }
        )
     assert result.success is True
        assert result.status == PipelineStatus.SUCCESS
        assert result.final_output is not None
     # Check that all stages completed
        assert len(result.stage_results) == 4
        assert "code_generation" in result.stage_results
        assert "code_cleaning" in result.stage_results
        assert "code_validation" in result.stage_results
        assert "code_execution" in result.stage_results
     # Check all stages succeeded
        for stage_name, stage_result in result.stage_results.items():
            assert stage_result.success, f"Stage {stage_name} failed: {stage_result.error_message}"
     # Check final output structure
        final_output = result.final_output
        assert "execution_results" in final_output
        assert "generated_code" in final_output
        assert "validation_results" in final_output
        assert "pipeline_metrics" in final_output
        assert "stage_metadata" in final_output

    def test_pipeline_failure_at_validation(self, mock_llm_client, mock_validator, mock_executor):
        """Test pipeline failure at validation stage."""
        # Make validation fail
        mock_validator.validate.return_value = ValidationResult(
            is_valid=False,
            syntax_errors=["Invalid syntax"],
            security_warnings=["Security issue detected"],
            performance_warnings=[],
            validation_time=0.1,
            security_score=0.2
        )
     pipeline = CodeGenerationPipeline(mock_llm_client, mock_validator, mock_executor)
     result = pipeline.generate_and_execute_code(
            user_query="Malicious query",
            analysis_context={
                "process_data": {"process_type": "python"},
                "raw_dataset": pd.DataFrame()
            }
        )
     assert result.success is False
        assert result.status == PipelineStatus.FAILED
        assert "Code validation failed" in result.error_message
     # Check that execution stage was not reached
        assert "code_execution" not in result.stage_results
     # Check that validation stage failed
        assert result.stage_results["code_validation"].failed

    def test_pipeline_health_check(self, pipeline):
        """Test pipeline health information."""
        health = pipeline.get_pipeline_health()
     assert health["pipeline_name"] == "code_generation_pipeline"
        assert health["total_stages"] == 4
        assert "llm_client_type" in health
        assert "validator_allowed_imports" in health
        assert "executor_limits" in health
        assert len(health["stages_info"]) == 4

    def test_create_pipeline_factory(self, mock_llm_client, mock_validator, mock_executor):
        """Test pipeline factory function."""
        pipeline = create_code_generation_pipeline(
            mock_llm_client, mock_validator, mock_executor
        )
     assert isinstance(pipeline, CodeGenerationPipeline)
        assert len(pipeline.stages) == 4
        assert pipeline.pipeline_name == "code_generation_pipeline"


class TestPipelineIntegration:
    """Integration tests for pipeline with real components."""

    @pytest.mark.integration
    def test_pipeline_with_simple_code(self):
        """Test pipeline with simple, safe code generation."""
        # This would require actual LLM, validator, and executor instances
        # For now, we'll skip this test in regular runs
        pytest.skip("Integration test - requires real components")

    @pytest.mark.integration 
    def test_pipeline_error_recovery(self):
        """Test pipeline error recovery mechanisms."""
        pytest.skip("Integration test - requires real components")


class TestReflectionStage:
    """Test the reflection stage functionality."""

    @pytest.fixture
    def mock_llm_client(self):
        """Mock LLM client for reflection."""
        return Mock(spec=LLMClient)

    @pytest.fixture
    def reflection_stage(self, mock_llm_client):
        """Create reflection stage."""
        from domain.pipeline.stages import ReflectionStage
        return ReflectionStage(mock_llm_client)

    def test_successful_reflection(self, reflection_stage):
        """Test successful reflection on execution results."""
        execution_results = ExecutionResults(
            status=ExecutionStatus.SUCCESS,
            output_data={"result": "analysis complete"},
            execution_time=2.5,
            memory_used_mb=150.0,
            stdout="Analysis completed successfully",
            stderr=""
        )
     context = PipelineContext(
            user_query="Test query",
            analysis_context={},
            execution_results=execution_results
        )
     result = reflection_stage.execute(context)
     assert result.success is True
        assert "execution_status" in result.data
        assert "suggestions" in result.data
        assert result.data["execution_status"] == "success"
        assert result.stage_metrics["successful_execution"] is True
        assert result.stage_metrics["suggestions_count"] >= 0

    def test_reflection_with_performance_issues(self, reflection_stage):
        """Test reflection identifies performance issues."""
        execution_results = ExecutionResults(
            status=ExecutionStatus.SUCCESS,
            output_data={"result": "slow analysis"},
            execution_time=15.0,  # Slow execution
            memory_used_mb=600.0,  # High memory usage
            stdout="Analysis completed",
            stderr="Warning: Large dataset processed"
        )
     context = PipelineContext(
            user_query="Test query",
            analysis_context={},
            execution_results=execution_results
        )
     result = reflection_stage.execute(context)
     assert result.success is True
        assert result.stage_metrics["performance_issues"] is True
        assert result.stage_metrics["has_warnings"] is True
        assert len(result.data["suggestions"]) > 0
     # Check specific suggestions
        suggestions_text = " ".join(result.data["suggestions"])
        assert "performance" in suggestions_text.lower() or "memory" in suggestions_text.lower()


class TestPipelineExtensibility:
    """Test pipeline extensibility and reflection capabilities."""

    @pytest.fixture
    def pipeline_with_reflection(self, mock_llm_client, mock_validator, mock_executor):
        """Create pipeline with reflection stage."""
        pipeline = CodeGenerationPipeline(mock_llm_client, mock_validator, mock_executor)
        pipeline.add_reflection_stage(enable_reflection=True)
        return pipeline

    def test_pipeline_with_reflection_stage(self, pipeline_with_reflection):
        """Test pipeline execution with reflection stage."""
        result = pipeline_with_reflection.generate_and_execute_code(
            user_query="Analyze data with reflection",
            analysis_context={
                "process_data": {"process_type": "python"},
                "raw_dataset": pd.DataFrame({"test": [1, 2, 3]})
            }
        )
     assert result.success is True
        assert len(result.stage_results) == 5  # Including reflection stage
        assert "reflection" in result.stage_results
     # Check reflection stage executed
        reflection_result = result.stage_results["reflection"]
        assert reflection_result.success is True

    def test_reflection_stage_optional(self, mock_llm_client, mock_validator, mock_executor):
        """Test that reflection stage is optional."""
        pipeline = CodeGenerationPipeline(mock_llm_client, mock_validator, mock_executor)
     # Default pipeline should have 4 stages
        assert len(pipeline.stages) == 4
     # Add reflection
        pipeline.add_reflection_stage(enable_reflection=True)
        assert len(pipeline.stages) == 5
     # Test disabled reflection
        pipeline2 = CodeGenerationPipeline(mock_llm_client, mock_validator, mock_executor)
        pipeline2.add_reflection_stage(enable_reflection=False)
        assert len(pipeline2.stages) == 4  # Should still be 4


class TestErrorPropagationImprovements:
    """Test improved error propagation and context handling."""

    def test_comprehensive_error_context(self, mock_llm_client, mock_validator, mock_executor):
        """Test that error context is properly propagated between stages."""
        # Make validation fail with detailed context
        mock_validator.validate.return_value = ValidationResult(
            is_valid=False,
            syntax_errors=["Syntax error at line 5: unexpected token"],
            security_warnings=["Use of eval() function detected"],
            performance_warnings=["Inefficient loop detected"],
            validation_time=0.2,
            security_score=0.1
        )
     pipeline = CodeGenerationPipeline(mock_llm_client, mock_validator, mock_executor)
     result = pipeline.generate_and_execute_code(
            user_query="Generate malicious code",
            analysis_context={
                "process_data": {"process_type": "python"},
                "raw_dataset": pd.DataFrame()
            }
        )
     assert result.failed is True
        assert result.error_message is not None
     # Check that error context contains detailed information
        validation_stage_result = result.stage_results["code_validation"]
        assert validation_stage_result.failed is True
        assert "syntax_errors" in validation_stage_result.error_context
        assert "security_warnings" in validation_stage_result.error_context
        assert len(validation_stage_result.error_context["syntax_errors"]) > 0
        assert len(validation_stage_result.error_context["security_warnings"]) > 0

    def test_stage_metrics_collection(self, mock_llm_client, mock_validator, mock_executor):
        """Test that comprehensive metrics are collected from all stages."""
        pipeline = CodeGenerationPipeline(mock_llm_client, mock_validator, mock_executor)
     result = pipeline.generate_and_execute_code(
            user_query="Generate test code",
            analysis_context={
                "process_data": {"process_type": "python"},
                "raw_dataset": pd.DataFrame({"test": [1, 2, 3]})
            }
        )
     assert result.success is True
     # Check that all stages have metrics
        for stage_name, stage_result in result.stage_results.items():
            assert stage_result.execution_time > 0
            assert len(stage_result.stage_metrics) > 0
     # Check specific metrics exist
        gen_metrics = result.stage_results["code_generation"].stage_metrics
        assert "code_length" in gen_metrics
        assert "template_used" in gen_metrics
     clean_metrics = result.stage_results["code_cleaning"].stage_metrics
        assert "original_length" in clean_metrics
        assert "cleaned_length" in clean_metrics
     val_metrics = result.stage_results["code_validation"].stage_metrics
        assert "is_valid" in val_metrics
        assert "security_score" in val_metrics
     exec_metrics = result.stage_results["code_execution"].stage_metrics
        assert "execution_status" in exec_metrics
        assert "execution_time" in exec_metrics


class TestPipelineHealthAndIntrospection:
    """Test pipeline health monitoring and introspection capabilities."""

    def test_pipeline_health_check(self, mock_llm_client, mock_validator, mock_executor):
        """Test pipeline health information."""
        # Mock validator to return allowed imports count
        mock_validator.get_allowed_imports = Mock(return_value=["pandas", "numpy", "matplotlib"])
     pipeline = CodeGenerationPipeline(mock_llm_client, mock_validator, mock_executor)
        health = pipeline.get_pipeline_health()
     assert health["pipeline_name"] == "code_generation_pipeline"
        assert health["total_stages"] == 4
        assert "llm_client_type" in health
        assert "validator_allowed_imports" in health
        assert "executor_limits" in health
        assert len(health["stages_info"]) == 4
     # Check stage info structure
        for stage_info in health["stages_info"]:
            assert "stage_name" in stage_info
            assert "stage_type" in stage_info
            assert "class_name" in stage_info


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
