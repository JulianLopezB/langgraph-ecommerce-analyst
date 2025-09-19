"""Pipeline stages for code generation."""

import re
import time
from typing import Any, Dict, Optional

try:
    from domain.entities import ExecutionResults, GeneratedCode
    from infrastructure.code_cleaning import create_ast_cleaner
    from infrastructure.execution.executor import SecureExecutor
    from infrastructure.execution.validator import CodeValidator, ValidationResult
    from infrastructure.llm.base import LLMClient
except ImportError:
    # Fallback classes for testing
    class GeneratedCode:
        def __init__(self, code_content, template_used=None, parameters=None):
            self.code_content = code_content
            self.template_used = template_used
            self.parameters = parameters or {}

    class ExecutionResults:
        def __init__(
            self,
            status,
            output_data=None,
            execution_time=0,
            memory_used_mb=0,
            error_message=None,
            stdout="",
            stderr="",
        ):
            self.status = status
            self.output_data = output_data
            self.execution_time = execution_time
            self.memory_used_mb = memory_used_mb
            self.error_message = error_message
            self.stdout = stdout
            self.stderr = stderr

    class ValidationResult:
        def __init__(
            self,
            is_valid,
            syntax_errors=None,
            security_warnings=None,
            performance_warnings=None,
            validation_time=0,
            security_score=1.0,
        ):
            self.is_valid = is_valid
            self.syntax_errors = syntax_errors or []
            self.security_warnings = security_warnings or []
            self.performance_warnings = performance_warnings or []
            self.validation_time = validation_time
            self.security_score = security_score

    # Mock interfaces
    CodeValidator = object
    SecureExecutor = object
    LLMClient = object

from domain.pipeline.base import (
    PipelineContext,
    PipelineStage,
    PipelineStageType,
    StageResult,
)


class CodeGenerationStage(PipelineStage[GeneratedCode]):
    """Stage for generating code using LLM."""

    def __init__(self, llm_client: LLMClient):
        """Initialize code generation stage."""
        super().__init__("code_generation", PipelineStageType.GENERATION)
        self.llm_client = llm_client

    def _validate_input(self, context: PipelineContext) -> Optional[str]:
        """Validate that we have the necessary context for code generation."""
        if not context.user_query:
            return "Missing user_query in context"

        if not context.analysis_context:
            return "Missing analysis_context in context"

        return None

    def _execute_stage(self, context: PipelineContext) -> StageResult[GeneratedCode]:
        """Generate code using the LLM client."""
        try:
            # Generate code using the LLM
            generated_code_content = self.llm_client.generate_adaptive_python_code(
                context.analysis_context
            )

            # Create GeneratedCode entity
            generated_code = GeneratedCode(
                code_content=generated_code_content,
                template_used=context.analysis_context.get("process_data", {}).get(
                    "process_type", "unknown"
                ),
                parameters={
                    "analysis_context": context.analysis_context,
                    "original_query": context.user_query,
                    "generation_timestamp": time.time(),
                },
            )

            # Update context
            context.code_content = generated_code_content

            # Collect metrics
            stage_metrics = {
                "code_length": (
                    len(generated_code_content)
                    if isinstance(generated_code_content, str)
                    else 0
                ),
                "template_used": generated_code.template_used,
                "has_imports": (
                    "import " in generated_code_content
                    if isinstance(generated_code_content, str)
                    else False
                ),
                "has_plotting": (
                    any(
                        lib in generated_code_content.lower()
                        for lib in ["matplotlib", "seaborn", "plotly"]
                    )
                    if isinstance(generated_code_content, str)
                    else False
                ),
            }

            self.logger.info(
                f"Generated {len(generated_code_content) if isinstance(generated_code_content, str) else 0} characters of code"
            )

            return StageResult[GeneratedCode](
                success=True, data=generated_code, stage_metrics=stage_metrics
            )

        except Exception as e:
            error_msg = f"Code generation failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)

            return StageResult[GeneratedCode](
                success=False,
                error_message=error_msg,
                error_context={"llm_error": str(e)},
            )


class CodeCleaningStage(PipelineStage[str]):
    """Stage for cleaning and formatting generated code."""

    def __init__(self):
        """Initialize code cleaning stage."""
        super().__init__("code_cleaning", PipelineStageType.CLEANING)
        self.code_cleaner = create_ast_cleaner()

    def _validate_input(self, context: PipelineContext) -> Optional[str]:
        """Validate that we have code to clean."""
        if not context.code_content:
            return "No code content to clean"
        return None

    def _execute_stage(self, context: PipelineContext) -> StageResult[str]:
        """Clean and format the generated code using AST-based processing."""
        try:
            original_code = context.code_content
            cleaned_code, cleaning_metadata = self.code_cleaner.clean_code(
                original_code
            )

            # Update context
            context.cleaned_code = cleaned_code

            # Collect comprehensive metrics from AST cleaner
            stage_metrics = {
                "original_length": len(original_code),
                "cleaned_length": len(cleaned_code),
                "original_lines": cleaning_metadata.get("original_lines", 0),
                "cleaned_lines": cleaning_metadata.get("cleaned_lines", 0),
                "syntax_valid": cleaning_metadata.get("syntax_valid", False),
                "imports_removed": len(cleaning_metadata.get("imports_removed", [])),
                "formatting_applied": cleaning_metadata.get(
                    "formatting_applied", False
                ),
                "cleaning_success": cleaning_metadata.get("success", False),
            }

            if cleaning_metadata.get("success"):
                self.logger.info(
                    f"AST-based cleaning successful: {stage_metrics['original_lines']} -> {stage_metrics['cleaned_lines']} lines"
                )
            else:
                self.logger.warning(
                    f"AST-based cleaning had issues: {cleaning_metadata.get('errors', [])}"
                )

            return StageResult[str](
                success=True,
                data=cleaned_code,
                stage_metrics=stage_metrics,
            )

        except Exception as e:
            error_msg = f"Code cleaning failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)

            return StageResult[str](
                success=False,
                error_message=error_msg,
                error_context={"cleaning_error": str(e)},
            )


class CodeValidationStage(PipelineStage[ValidationResult]):
    """Stage for validating generated code."""

    def __init__(self, validator: CodeValidator):
        """Initialize code validation stage."""
        super().__init__("code_validation", PipelineStageType.VALIDATION)
        self.validator = validator

    def _validate_input(self, context: PipelineContext) -> Optional[str]:
        """Validate that we have cleaned code to validate."""
        if not context.cleaned_code:
            return "No cleaned code to validate"
        return None

    def _execute_stage(self, context: PipelineContext) -> StageResult[ValidationResult]:
        """Validate the cleaned code."""
        try:
            validation_result = self.validator.validate(context.cleaned_code)

            # Update context
            context.validation_results = validation_result

            # Collect metrics
            stage_metrics = {
                "is_valid": validation_result.is_valid,
                "security_score": validation_result.security_score,
                "syntax_errors_count": len(validation_result.syntax_errors),
                "security_warnings_count": len(validation_result.security_warnings),
                "performance_warnings_count": len(
                    validation_result.performance_warnings
                ),
                "validation_time": validation_result.validation_time,
            }

            if validation_result.is_valid:
                self.logger.info(
                    f"Code validation passed with security score: {validation_result.security_score:.2f}"
                )
            else:
                self.logger.warning(
                    f"Code validation failed: {len(validation_result.syntax_errors)} syntax errors, "
                    f"{len(validation_result.security_warnings)} security warnings"
                )

            return StageResult[ValidationResult](
                success=validation_result.is_valid,
                data=validation_result,
                error_message=(
                    None if validation_result.is_valid else "Code validation failed"
                ),
                error_context={
                    "syntax_errors": validation_result.syntax_errors,
                    "security_warnings": validation_result.security_warnings,
                    "performance_warnings": validation_result.performance_warnings,
                },
                stage_metrics=stage_metrics,
            )

        except Exception as e:
            error_msg = f"Code validation error: {str(e)}"
            self.logger.error(error_msg, exc_info=True)

            return StageResult[ValidationResult](
                success=False,
                error_message=error_msg,
                error_context={"validation_error": str(e)},
            )


class CodeExecutionStage(PipelineStage[ExecutionResults]):
    """Stage for executing validated code."""

    def __init__(self, executor: SecureExecutor):
        """Initialize code execution stage."""
        super().__init__("code_execution", PipelineStageType.EXECUTION)
        self.executor = executor

    def _validate_input(self, context: PipelineContext) -> Optional[str]:
        """Validate that we have validated code to execute."""
        if not context.cleaned_code:
            return "No cleaned code to execute"

        if not context.validation_results:
            return "Code has not been validated"

        if not context.validation_results.is_valid:
            return "Code failed validation and cannot be executed"

        return None

    def _execute_stage(self, context: PipelineContext) -> StageResult[ExecutionResults]:
        """Execute the validated code."""
        try:
            # Prepare execution context
            execution_context = self._prepare_execution_context(context)

            # Execute the code
            execution_results = self.executor.execute_code(
                context.cleaned_code, execution_context
            )

            # Update context
            context.execution_results = execution_results

            # Collect metrics
            stage_metrics = {
                "execution_status": execution_results.status.value,
                "execution_time": execution_results.execution_time,
                "memory_used_mb": execution_results.memory_used_mb,
                "has_output_data": execution_results.output_data is not None,
                "stdout_length": len(execution_results.stdout),
                "stderr_length": len(execution_results.stderr),
            }

            success = execution_results.status.value == "success"

            if success:
                self.logger.info(
                    f"Code executed successfully in {execution_results.execution_time:.2f}s, "
                    f"memory: {execution_results.memory_used_mb:.1f}MB"
                )
            else:
                self.logger.warning(
                    f"Code execution failed: {execution_results.error_message}"
                )

            return StageResult[ExecutionResults](
                success=success,
                data=execution_results,
                error_message=execution_results.error_message if not success else None,
                error_context=(
                    {
                        "execution_status": execution_results.status.value,
                        "stderr": execution_results.stderr,
                    }
                    if not success
                    else {}
                ),
                stage_metrics=stage_metrics,
            )

        except Exception as e:
            error_msg = f"Code execution error: {str(e)}"
            self.logger.error(error_msg, exc_info=True)

            return StageResult[ExecutionResults](
                success=False,
                error_message=error_msg,
                error_context={"execution_error": str(e)},
            )

    def _prepare_execution_context(self, context: PipelineContext) -> Dict[str, Any]:
        """Prepare the execution context with necessary variables."""
        execution_context = {}
        # Add DataFrame if available in analysis context
        analysis_context = context.analysis_context
        if "raw_dataset" in analysis_context:
            df_name = analysis_context.get("dataframe_name", "df")
            execution_context[df_name] = analysis_context["raw_dataset"]

        return execution_context


class ReflectionStage(PipelineStage[Dict[str, Any]]):
    """
    Stage for reflecting on execution results and suggesting improvements.

    This stage demonstrates the extensibility of the pipeline pattern
    and can be easily added when reflection capabilities are needed.
    """

    def __init__(self, llm_client: LLMClient):
        """Initialize reflection stage."""
        super().__init__("reflection", PipelineStageType.REFLECTION)
        self.llm_client = llm_client

    def _validate_input(self, context: PipelineContext) -> Optional[str]:
        """Validate that we have execution results to reflect on."""
        if not context.execution_results:
            return "No execution results to reflect on"
        return None

    def _execute_stage(self, context: PipelineContext) -> StageResult[Dict[str, Any]]:
        """Reflect on execution results and suggest improvements."""
        try:
            execution_results = context.execution_results

            # Analyze execution results
            reflection_data = {
                "execution_status": execution_results.status.value,
                "execution_time": execution_results.execution_time,
                "memory_used": execution_results.memory_used_mb,
                "has_output": execution_results.output_data is not None,
                "stdout_length": len(execution_results.stdout),
                "stderr_length": len(execution_results.stderr),
            }

            # Add suggestions based on results
            suggestions = []

            if execution_results.execution_time > 10.0:
                suggestions.append("Consider optimizing code for better performance")

            if execution_results.memory_used_mb > 500:
                suggestions.append(
                    "Consider reducing memory usage or processing data in chunks"
                )

            if execution_results.stderr:
                suggestions.append(
                    "Review stderr output for potential warnings or issues"
                )

            if not execution_results.output_data:
                suggestions.append(
                    "Code executed but produced no output data - verify analysis logic"
                )

            reflection_data["suggestions"] = suggestions
            reflection_data["reflection_summary"] = (
                f"Execution completed in {execution_results.execution_time:.2f}s with {len(suggestions)} suggestions"
            )

            stage_metrics = {
                "suggestions_count": len(suggestions),
                "performance_issues": execution_results.execution_time > 10.0
                or execution_results.memory_used_mb > 500,
                "has_warnings": bool(execution_results.stderr),
                "successful_execution": execution_results.status.value == "success",
            }

            self.logger.info(
                f"Reflection completed with {len(suggestions)} suggestions"
            )

            return StageResult[Dict[str, Any]](
                success=True, data=reflection_data, stage_metrics=stage_metrics
            )

        except Exception as e:
            error_msg = f"Reflection stage error: {str(e)}"
            self.logger.error(error_msg, exc_info=True)

            return StageResult[Dict[str, Any]](
                success=False,
                error_message=error_msg,
                error_context={"reflection_error": str(e)},
            )
