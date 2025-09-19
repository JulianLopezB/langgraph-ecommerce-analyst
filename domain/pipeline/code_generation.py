"""Code generation pipeline implementation."""

from typing import Any, Dict, Optional

from domain.entities import GeneratedCode, ExecutionResults
from domain.pipeline.base import (
    Pipeline,
    PipelineContext,
    PipelineResult,
    PipelineStage,
    StageResult,
)
from domain.pipeline.stages import (
    CodeGenerationStage,
    CodeCleaningStage,
    CodeValidationStage,
    CodeExecutionStage,
    ReflectionStage,
)
from infrastructure.execution.validator import CodeValidator
from infrastructure.execution.executor import SecureExecutor
from infrastructure.llm.base import LLMClient
from infrastructure.logging import get_logger

logger = get_logger(__name__)


class CodeGenerationPipeline(Pipeline):
    """
    Structured pipeline for code generation with clear stages:
    Generation → Cleaning → Validation → Execution

    This pipeline replaces the fragmented approach with a structured pattern
    that enables proper error propagation, logging, and metrics collection.
    """

    def __init__(
        self, llm_client: LLMClient, validator: CodeValidator, executor: SecureExecutor
    ):
        """
           Initialize the code generation pipeline.
        Args:
               llm_client: Client for generating code
               validator: Code validator for security and syntax checking
               executor: Secure code executor
        """
        super().__init__("code_generation_pipeline")

        self.llm_client = llm_client
        self.validator = validator
        self.executor = executor

        # Build the pipeline stages
        self._build_pipeline()

    def _build_pipeline(self) -> None:
        """Build the pipeline with all necessary stages."""
        # Stage 1: Generate code using LLM
        self.add_stage(CodeGenerationStage(self.llm_client))

        # Stage 2: Clean and format the generated code
        self.add_stage(CodeCleaningStage())

        # Stage 3: Validate code for security and syntax
        self.add_stage(CodeValidationStage(self.validator))

        # Stage 4: Execute validated code in secure environment
        self.add_stage(CodeExecutionStage(self.executor))

        logger.info(f"Built pipeline with {len(self.stages)} stages")

    def generate_and_execute_code(
        self, user_query: str, analysis_context: Dict[str, Any]
    ) -> PipelineResult:
        """
           Generate and execute code using the structured pipeline.
        Args:
               user_query: The original user query
               analysis_context: Context for code generation including data characteristics

           Returns:
               PipelineResult with execution results and comprehensive metrics
        """
        # Create pipeline context
        context = PipelineContext(
            user_query=user_query, analysis_context=analysis_context
        )

        logger.info(
            f"Starting code generation pipeline for query: {user_query[:100]}..."
        )

        # Execute the pipeline
        result = self.execute(context)

        # Log final result
        if result.success:
            logger.info(
                f"Pipeline completed successfully in {result.total_execution_time:.2f}s"
            )
        else:
            logger.error(f"Pipeline failed: {result.error_message}")

        return result

    def _update_context_after_stage(
        self, context: PipelineContext, stage: PipelineStage, result: StageResult
    ) -> None:
        """Update context after each stage execution."""
        if stage.stage_name == "code_generation" and result.success:
            # Store the generated code entity
            generated_code: GeneratedCode = result.data
            context.code_content = generated_code.code_content

        elif stage.stage_name == "code_cleaning" and result.success:
            # Store the cleaned code
            context.cleaned_code = result.data

        elif stage.stage_name == "code_validation" and result.success:
            # Store validation results
            context.validation_results = result.data

        elif stage.stage_name == "code_execution" and result.success:
            # Store execution results
            context.execution_results = result.data

        elif stage.stage_name == "reflection" and result.success:
            # Store reflection results in context metadata
            context.stage_metadata["reflection_analysis"] = result.data

    def _extract_final_output(self, context: PipelineContext) -> Dict[str, Any]:
        """Extract comprehensive output from the pipeline execution."""
        output = {
            "execution_results": context.execution_results,
            "generated_code": GeneratedCode(
                code_content=context.cleaned_code or context.code_content or "",
                validation_passed=(
                    context.validation_results.is_valid
                    if context.validation_results
                    else False
                ),
                security_score=(
                    context.validation_results.security_score
                    if context.validation_results
                    else 0.0
                ),
                parameters=context.analysis_context,
            ),
            "validation_results": context.validation_results,
            "pipeline_metrics": context.metrics,
            "stage_metadata": context.stage_metadata,
        }

        return output

    def add_reflection_stage(self, enable_reflection: bool = True) -> None:
        """
           Add a reflection stage for analyzing execution results and suggesting improvements.
           This demonstrates how new stages can be easily added to the pipeline.
        Args:
               enable_reflection: Whether to enable reflection stage
        """
        if enable_reflection:
            reflection_stage = ReflectionStage(self.llm_client)
            self.add_stage(reflection_stage)
            logger.info(
                "Reflection stage added to pipeline - will analyze execution results"
            )
        else:
            logger.info("Reflection stage capability available but not enabled")

    def get_pipeline_health(self) -> Dict[str, Any]:
        """Get health information about the pipeline components."""
        return {
            "pipeline_name": self.pipeline_name,
            "total_stages": len(self.stages),
            "llm_client_type": type(self.llm_client).__name__,
            "validator_allowed_imports": len(self.validator.get_allowed_imports()),
            "executor_limits": {
                "max_execution_time": self.executor.max_execution_time,
                "max_memory_mb": self.executor.max_memory_mb,
            },
            "stages_info": [stage.get_stage_info() for stage in self.stages],
        }


def create_code_generation_pipeline(
    llm_client: LLMClient, validator: CodeValidator, executor: SecureExecutor
) -> CodeGenerationPipeline:
    """
       Factory function to create a properly configured code generation pipeline.

       Args:
           llm_client: LLM client for code generation
           validator: Code validator
           executor: Secure code executor
    Returns:
           Configured CodeGenerationPipeline instance
    """
    pipeline = CodeGenerationPipeline(
        llm_client=llm_client, validator=validator, executor=executor
    )

    logger.info("Created code generation pipeline with all stages configured")
    return pipeline
