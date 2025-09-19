"""Base pipeline interfaces and contracts."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic
from enum import Enum
import time
from datetime import datetime

try:
    from infrastructure.logging import get_logger

    logger = get_logger(__name__)
except ImportError:
    # Fallback for testing without dependencies
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

T = TypeVar("T")


class PipelineStageType(Enum):
    """Types of pipeline stages."""

    GENERATION = "generation"
    CLEANING = "cleaning"
    VALIDATION = "validation"
    EXECUTION = "execution"
    REFLECTION = "reflection"  # For future extensibility


class PipelineStatus(Enum):
    """Status of pipeline execution."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PipelineContext:
    """Context passed between pipeline stages."""

    # Core data
    user_query: str
    analysis_context: Dict[str, Any]

    # Generated artifacts
    code_content: Optional[str] = None
    cleaned_code: Optional[str] = None
    validation_results: Optional[Any] = None
    execution_results: Optional[Any] = None

    # Metadata
    stage_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    error_context: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)

    # Pipeline state
    current_stage: Optional[str] = None
    pipeline_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class StageResult(Generic[T]):
    """Result from a single pipeline stage."""

    success: bool
    data: Optional[T] = None
    error_message: Optional[str] = None
    error_context: Dict[str, Any] = field(default_factory=dict)
    stage_metrics: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0

    @property
    def failed(self) -> bool:
        """Check if stage failed."""
        return not self.success


@dataclass
class PipelineResult:
    """Final result from pipeline execution."""

    status: PipelineStatus
    context: PipelineContext
    final_output: Optional[Any] = None
    error_message: Optional[str] = None
    total_execution_time: float = 0.0
    stage_results: Dict[str, StageResult] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if pipeline succeeded."""
        return self.status == PipelineStatus.SUCCESS

    @property
    def failed(self) -> bool:
        """Check if pipeline failed."""
        return self.status == PipelineStatus.FAILED


class PipelineStage(ABC, Generic[T]):
    """Abstract base class for pipeline stages."""

    def __init__(self, stage_name: str, stage_type: PipelineStageType):
        """Initialize pipeline stage."""
        self.stage_name = stage_name
        self.stage_type = stage_type
        self.logger = logger.getChild(stage_name)

    def execute(self, context: PipelineContext) -> StageResult[T]:
        """Execute the pipeline stage with proper error handling and metrics."""
        start_time = time.time()

        try:
            self.logger.info(f"Starting {self.stage_name} stage")
            context.current_stage = self.stage_name

            # Pre-execution validation
            validation_error = self._validate_input(context)
            if validation_error:
                return StageResult[T](
                    success=False,
                    error_message=f"Input validation failed: {validation_error}",
                    execution_time=time.time() - start_time,
                )

            # Execute stage logic
            result = self._execute_stage(context)

            # Post-execution processing
            execution_time = time.time() - start_time
            result.execution_time = execution_time

            # Update context with stage metadata
            context.stage_metadata[self.stage_name] = {
                "execution_time": execution_time,
                "success": result.success,
                "timestamp": datetime.now().isoformat(),
                **result.stage_metrics,
            }

            # Update context metrics
            context.metrics[f"{self.stage_name}_execution_time"] = execution_time
            context.metrics[f"{self.stage_name}_success"] = result.success

            if result.success:
                self.logger.info(
                    f"Completed {self.stage_name} stage successfully in {execution_time:.2f}s"
                )
            else:
                self.logger.warning(
                    f"Stage {self.stage_name} failed: {result.error_message}"
                )
                # Propagate error to context
                context.error_context[f"{self.stage_name}_error"] = (
                    result.error_message or "Unknown error"
                )

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Unexpected error in {self.stage_name}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)

            context.error_context[f"{self.stage_name}_error"] = error_msg

            return StageResult[T](
                success=False, error_message=error_msg, execution_time=execution_time
            )

    @abstractmethod
    def _execute_stage(self, context: PipelineContext) -> StageResult[T]:
        """Execute the core stage logic. Must be implemented by subclasses."""
        pass

    def _validate_input(self, context: PipelineContext) -> Optional[str]:
        """Validate input context. Override in subclasses for specific validation."""
        return None

    def get_stage_info(self) -> Dict[str, Any]:
        """Get stage information for introspection."""
        return {
            "stage_name": self.stage_name,
            "stage_type": self.stage_type.value,
            "class_name": self.__class__.__name__,
        }


class Pipeline(ABC):
    """Abstract base class for pipelines."""

    def __init__(self, pipeline_name: str):
        """Initialize pipeline."""
        self.pipeline_name = pipeline_name
        self.stages: List[PipelineStage] = []
        self.logger = get_logger(f"{__name__}.{pipeline_name}")

    def add_stage(self, stage: PipelineStage) -> None:
        """Add a stage to the pipeline."""
        self.stages.append(stage)
        self.logger.debug(f"Added stage: {stage.stage_name}")

    def execute(self, context: PipelineContext) -> PipelineResult:
        """Execute the complete pipeline."""
        start_time = time.time()
        context.pipeline_id = f"{self.pipeline_name}_{int(start_time)}"

        self.logger.info(
            f"Starting pipeline {self.pipeline_name} with {len(self.stages)} stages"
        )

        stage_results = {}

        try:
            for stage in self.stages:
                stage_result = stage.execute(context)
                stage_results[stage.stage_name] = stage_result

                if stage_result.failed:
                    self.logger.error(
                        f"Pipeline {self.pipeline_name} failed at stage {stage.stage_name}: "
                        f"{stage_result.error_message}"
                    )

                    total_time = time.time() - start_time
                    return PipelineResult(
                        status=PipelineStatus.FAILED,
                        context=context,
                        error_message=stage_result.error_message,
                        total_execution_time=total_time,
                        stage_results=stage_results,
                    )

                # Allow stage to update context for next stage
                self._update_context_after_stage(context, stage, stage_result)

            # Pipeline completed successfully
            total_time = time.time() - start_time
            self.logger.info(
                f"Pipeline {self.pipeline_name} completed successfully in {total_time:.2f}s"
            )

            return PipelineResult(
                status=PipelineStatus.SUCCESS,
                context=context,
                final_output=self._extract_final_output(context),
                total_execution_time=total_time,
                stage_results=stage_results,
            )

        except Exception as e:
            total_time = time.time() - start_time
            error_msg = f"Unexpected pipeline error: {str(e)}"
            self.logger.error(error_msg, exc_info=True)

            return PipelineResult(
                status=PipelineStatus.FAILED,
                context=context,
                error_message=error_msg,
                total_execution_time=total_time,
                stage_results=stage_results,
            )

    def _update_context_after_stage(
        self, context: PipelineContext, stage: PipelineStage, result: StageResult
    ) -> None:
        """Update context after stage execution. Override in subclasses."""
        pass

    def _extract_final_output(self, context: PipelineContext) -> Any:
        """Extract final output from context. Override in subclasses."""
        return context.execution_results

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get pipeline information for introspection."""
        return {
            "pipeline_name": self.pipeline_name,
            "stages": [stage.get_stage_info() for stage in self.stages],
            "total_stages": len(self.stages),
        }
