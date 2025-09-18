"""High-level orchestration for the data analysis workflow."""

from typing import Any, Callable, Dict

from domain.entities import ProcessType
from application.use_cases import (
    SchemaAnalysisUseCase,
    ProcessClassificationUseCase,
    SQLGenerationUseCase,
    PythonGenerationUseCase,
    CodeValidationUseCase,
    CodeExecutionUseCase,
    InsightSynthesisUseCase,
)


class AnalysisWorkflow:
    """Coordinate use case classes to produce final analysis results."""

    def __init__(
        self,
        schema_analysis: SchemaAnalysisUseCase,
        process_classification: ProcessClassificationUseCase,
        sql_generation: SQLGenerationUseCase,
        python_generation: PythonGenerationUseCase,
        validation: CodeValidationUseCase,
        execution: CodeExecutionUseCase,
        synthesis: InsightSynthesisUseCase,
    ) -> None:
        """Initialize the workflow with required use cases."""
        self._schema_analysis = schema_analysis
        self._process_classification = process_classification
        self._sql_generation = sql_generation
        self._python_generation = python_generation
        self._validation = validation
        self._execution = execution
        self._synthesis = synthesis

    def run(self, query: str) -> str:
        """Execute the complete analysis workflow and return insights."""
        # Analyze data schema
        schema_info = self._schema_analysis.analyze()

        # Determine processing strategy
        process_type_raw = self._process_classification.classify(query, schema_info)
        process_type = next(
            (
                ptype
                for ptype in ProcessType
                if ptype.value in str(process_type_raw).lower()
            ),
            ProcessType.SQL,
        )

        # Always generate SQL and retrieve data
        sql = self._sql_generation.generate(query, schema_info)
        data = self._execution.run_query(sql)

        # For complex analysis, generate and execute Python code
        if process_type is ProcessType.PYTHON:
            code_prompt = f"Analyze the following data for query: {query}"
            code = self._python_generation.generate(code_prompt)

            if not self._validation.validate(code):
                raise ValueError("Generated code failed validation")

            data = self._execution.execute_code(code, data)

        # Convert raw results into insights
        return self._synthesis.synthesize(data, query)


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
