"""High-level orchestration for the data analysis workflow."""

from typing import Any, Callable, Dict

from domain.entities import ProcessType
from domain.pipeline import CodeGenerationPipeline, create_code_generation_pipeline

from application.use_cases import (
    CodeExecutionUseCase,
    CodeValidationUseCase,
    InsightSynthesisUseCase,
    ProcessClassificationUseCase,
    PythonGenerationUseCase,
    SchemaAnalysisUseCase,
    SQLGenerationUseCase,
)

from infrastructure.execution.validator import CodeValidator
from infrastructure.execution.executor import SecureExecutor
from infrastructure.llm.base import LLMClient
from infrastructure.logging import get_logger

logger = get_logger(__name__)


class AnalysisWorkflow:
    """
    Coordinate use case classes to produce final analysis results.

    This class now uses the structured CodeGenerationPipeline to replace
    the previous fragmented approach with proper error propagation,
    logging, and metrics collection.
    """

    def __init__(
        self,
        schema_analysis: SchemaAnalysisUseCase,
        process_classification: ProcessClassificationUseCase,
        sql_generation: SQLGenerationUseCase,
        python_generation: PythonGenerationUseCase,
        validation: CodeValidationUseCase,
        execution: CodeExecutionUseCase,
        synthesis: InsightSynthesisUseCase,
        llm_client: LLMClient,
        validator: CodeValidator,
        executor: SecureExecutor,
    ) -> None:
        """Initialize the workflow with required use cases and pipeline components."""
        self._schema_analysis = schema_analysis
        self._process_classification = process_classification
        self._sql_generation = sql_generation
        self._python_generation = python_generation
        self._validation = validation
        self._execution = execution
        self._synthesis = synthesis

        # Create structured pipeline for code generation
        self._code_pipeline = create_code_generation_pipeline(
            llm_client=llm_client,
            validator=validator,
            executor=executor
        )

        logger.info("AnalysisWorkflow initialized with structured code generation pipeline")

    def run(self, query: str) -> str:
        """
        Execute the complete analysis workflow and return insights.
     Now uses the structured CodeGenerationPipeline for Python code generation
        instead of the fragmented approach, providing better error handling,
        logging, and metrics collection.
        """
        try:
            logger.info(f"Starting analysis workflow for query: {query[:100]}...")

            # Analyze data schema
            schema_info = self._schema_analysis.analyze()
            logger.debug("Schema analysis completed")

            # Determine processing strategy
            process_type_raw = self._process_classification.classify(query, schema_info)
            process_type = next(
                (ptype for ptype in ProcessType if ptype.value in str(process_type_raw).lower()),
                ProcessType.SQL,
            )
            logger.info(f"Process type determined: {process_type}")

            # Always generate SQL and retrieve data
            sql = self._sql_generation.generate(query, schema_info)
            data = self._execution.run_query(sql)
            logger.info(f"SQL execution completed, retrieved {len(data) if hasattr(data, '__len__') else 'data'} rows")

            # For complex analysis, use structured pipeline for Python code generation
            if process_type is ProcessType.PYTHON:
                logger.info("Using structured pipeline for Python code generation")

                # Prepare analysis context for pipeline
                analysis_context = {
                    "original_query": query,
                    "query_intent": query,
                    "process_data": {"process_type": "python"},
                    "sql_query": sql,
                    "sql_explanation": "Data retrieved via SQL for analysis",
                    "data_info": {"shape": getattr(data, 'shape', None), "type": type(data).__name__},
                    "data_characteristics": self._inspect_data_characteristics(data),
                    "data_understanding": {"query_intent": query},
                    "sql_metadata": {"explanation": "SQL data retrieval"},
                    "dataframe_name": "df",
                    "raw_dataset": data
                }

                # Execute structured pipeline
                pipeline_result = self._code_pipeline.generate_and_execute_code(
                    user_query=query,
                    analysis_context=analysis_context
                )

                if pipeline_result.success:
                    logger.info(
                        f"Pipeline completed successfully in {pipeline_result.total_execution_time:.2f}s"
                    )

                    # Extract execution results
                    if pipeline_result.final_output and pipeline_result.final_output.get("execution_results"):
                        execution_results = pipeline_result.final_output["execution_results"]
                        if execution_results.output_data:
                            data = execution_results.output_data
                        else:
                            # Use original data if no output_data
                            logger.warning("No output data from pipeline, using original data")
                else:
                    # Pipeline failed - provide detailed error context
                    error_details = []
                    for stage_name, stage_result in pipeline_result.stage_results.items():
                        if stage_result.failed:
                            error_details.append(f"{stage_name}: {stage_result.error_message}")

                    comprehensive_error = f"Code generation pipeline failed: {'; '.join(error_details)}"
                    logger.error(comprehensive_error)

                    # Instead of raising generic ValueError, provide structured error
                    raise RuntimeError(
                        f"Analysis failed during Python code generation. "
                        f"Pipeline error: {pipeline_result.error_message}. "
                        f"Stage details: {'; '.join(error_details)}"
                    )

            # Convert raw results into insights
            insights = self._synthesis.synthesize(data, query)
            logger.info("Analysis workflow completed successfully")
            return insights

        except Exception as e:
            logger.error(f"Analysis workflow failed: {str(e)}", exc_info=True)
            raise

    def _inspect_data_characteristics(self, data) -> Dict[str, Any]:
        """Inspect data characteristics for pipeline context."""
        try:
            import pandas as pd

            if isinstance(data, pd.DataFrame):
                return {
                    "shape": data.shape,
                    "columns": list(data.columns),
                    "dtypes": {col: str(dtype) for col, dtype in data.dtypes.items()},
                    "memory_usage": data.memory_usage(deep=True).sum(),
                    "has_nulls": data.isnull().any().any(),
                    "numeric_columns": list(data.select_dtypes(include=['number']).columns),
                    "datetime_columns": list(data.select_dtypes(include=['datetime']).columns),
                    "categorical_columns": list(data.select_dtypes(include=['object', 'category']).columns)
                }
            else:
                return {
                    "type": type(data).__name__,
                    "length": len(data) if hasattr(data, '__len__') else None,
                    "shape": getattr(data, 'shape', None)
                }
        except Exception as e:
            logger.warning(f"Failed to inspect data characteristics: {e}")
            return {
                "type": type(data).__name__,
                "inspection_error": str(e)
            }


def create_workflow_adapter(
    workflow: AnalysisWorkflow,
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """Adapter for integrating the workflow with LangGraph nodes.

    The returned callable expects a state dictionary containing a
    ``user_query`` key. The workflow result is stored under ``insights``.
    This keeps LangGraph-specific constructs out of the application code.
    """

    def adapter(state: Dict[str, Any]) -> Dict[str, Any]:
        query = state.get("user_query", "")
        state["insights"] = workflow.run(query)
        return state

    return adapter
