"""Execution and synthesis nodes."""
from datetime import datetime
from typing import Dict, Any

import pandas as pd

from infrastructure.persistence import data_repository
from infrastructure.execution import validator, secure_executor
from infrastructure.logging import get_logger
from infrastructure.llm import llm_client
from workflow.state import AnalysisState
from domain.entities import ConversationMessage

logger = get_logger(__name__)
data_repo = data_repository
llm_service = llm_client


def execute_sql(state: AnalysisState) -> AnalysisState:
    """Execute SQL query and retrieve data."""
    logger.info("Executing SQL query")

    try:
        df = data_repo.execute_query(state["sql_query"])
        state["raw_dataset"] = df

        data_info = {
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
            "shape": df.shape,
            "head": df.head().to_dict(),
            "null_counts": df.isnull().sum().to_dict(),
        }
        state["analysis_outputs"]["data_info"] = data_info

        message = ConversationMessage(
            timestamp=datetime.now(),
            role="assistant",
            content=f"Retrieved {len(df)} rows and {len(df.columns)} columns from BigQuery",
            message_type="result",
        )
        state["conversation_history"].append(message)

        if state["needs_python_analysis"]:
            state["next_step"] = "generate_python_code"
        else:
            state["next_step"] = "synthesize_results"

        logger.info(
            "SQL executed successfully: %s rows, %s columns",
            df.shape[0],
            df.shape[1],
        )

    except Exception as e:
        logger.error(f"Error executing SQL: {str(e)}")
        state["error_context"]["sql_execution_error"] = str(e)
        state["next_step"] = "handle_error"

    return state


def _inspect_data_characteristics(df) -> Dict[str, Any]:
    """Inspect actual data to understand its characteristics for adaptive code generation."""
    if df is None or df.empty:
        return {"error": "No data available", "shape": (0, 0)}

    try:
        characteristics = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "data_types": df.dtypes.to_dict(),
            "non_null_counts": df.count().to_dict(),
            "numeric_columns": df.select_dtypes(include=["number"]).columns.tolist(),
            "datetime_columns": [],
            "categorical_columns": df.select_dtypes(include=["object", "string"]).columns.tolist(),
            "sample_values": {},
        }

        for col in df.columns:
            if df[col].dtype == "object":
                sample_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                if sample_val and (
                    "date" in col.lower()
                    or "time" in col.lower()
                    or "month" in col.lower()
                ):
                    characteristics["datetime_columns"].append(col)

            if not df[col].dropna().empty:
                characteristics["sample_values"][col] = df[col].dropna().iloc[:3].tolist()

        if len(df) > 0:
            characteristics["time_series_capable"] = len(df) >= 3
            characteristics["seasonal_analysis_capable"] = len(df) >= 24
            characteristics["trend_analysis_capable"] = len(df) >= 6

        if characteristics["datetime_columns"] and characteristics["numeric_columns"]:
            characteristics["forecasting_ready"] = True
            characteristics["time_column"] = characteristics["datetime_columns"][0]
            characteristics["value_column"] = characteristics["numeric_columns"][0]
        else:
            characteristics["forecasting_ready"] = False

        logger.debug(f"Data characteristics: {characteristics}")
        return characteristics

    except Exception as e:
        logger.warning(f"Error inspecting data characteristics: {e}")
        return {
            "error": str(e),
            "shape": df.shape if df is not None else (0, 0),
            "columns": df.columns.tolist() if df is not None else [],
        }


def generate_python_code(state: AnalysisState) -> AnalysisState:
    """Generate adaptive Python code based on actual data inspection."""
    logger.info("Generating data-adaptive Python analysis code")

    try:
        data_characteristics = _inspect_data_characteristics(state["raw_dataset"])

        analysis_context = {
            "original_query": state["user_query"],
            "query_intent": state["analysis_outputs"].get("data_understanding", {}).get("query_intent", state["user_query"]),
            "process_data": state["analysis_outputs"]["process_data"],
            "sql_query": state.get("sql_query", ""),
            "sql_explanation": state["analysis_outputs"].get("sql_metadata", {}).get("explanation", ""),
            "data_info": state["analysis_outputs"]["data_info"],
            "data_characteristics": data_characteristics,
            "data_understanding": state["analysis_outputs"].get("data_understanding", {}),
            "sql_metadata": state["analysis_outputs"].get("sql_metadata", {}),
        }

        python_code = llm_service.generate_adaptive_python_code(analysis_context)

        generated_code = state["generated_code"]
        if generated_code is None:
            from domain.entities import GeneratedCode  # avoid circular
            generated_code = GeneratedCode(
                code_content=python_code,
                template_used=analysis_context["process_data"].get("process_type", "unknown"),
                parameters={
                    "analysis_context": analysis_context,
                    "original_query": state["user_query"],
                    "sql_query": state.get("sql_query", ""),
                    "data_characteristics": data_characteristics,
                },
            )
            state["generated_code"] = generated_code
        else:
            generated_code.code_content = python_code
            generated_code.template_used = analysis_context["process_data"].get("process_type", "unknown")
            generated_code.parameters = {
                "analysis_context": analysis_context,
                "original_query": state["user_query"],
                "sql_query": state.get("sql_query", ""),
                "data_characteristics": data_characteristics,
            }

        state["next_step"] = "validate_code"
        logger.info(
            "Data-adaptive Python code generated for: %s... (Data shape: %s)",
            state["user_query"][:50],
            data_characteristics.get("shape", "unknown"),
        )

    except Exception as e:
        logger.error(f"Error generating Python code: {str(e)}")
        state["error_context"]["code_generation_error"] = str(e)
        state["next_step"] = "handle_error"

    return state


def validate_code(state: AnalysisState) -> AnalysisState:
    """Validate generated Python code for security and syntax."""
    logger.info("Validating generated code")

    try:
        if not state["generated_code"]:
            raise ValueError("No code to validate")

        validation_result = validator.validate(state["generated_code"].code_content)
        state["validation_results"] = validation_result

        state["generated_code"].validation_passed = validation_result.is_valid
        state["generated_code"].security_score = validation_result.security_score

        if validation_result.is_valid:
            state["next_step"] = "execute_code"
            logger.info("Code validation passed")
        else:
            logger.warning(
                f"Code validation failed: {validation_result.security_warnings}"
            )
            state["error_context"]["validation_errors"] = {
                "syntax_errors": validation_result.syntax_errors,
                "security_warnings": validation_result.security_warnings,
                "security_score": validation_result.security_score,
            }
            state["next_step"] = "handle_error"

    except Exception as e:
        logger.error(f"Error validating code: {str(e)}")
        state["error_context"]["validation_error"] = str(e)
        state["next_step"] = "handle_error"

    return state


def execute_code(state: AnalysisState) -> AnalysisState:
    """Execute validated Python code in secure environment."""
    logger.info("Executing Python code")

    try:
        if not state["generated_code"] or not state["generated_code"].validation_passed:
            raise ValueError("Invalid or unvalidated code")

        context = {"df": state["raw_dataset"]}

        execution_results = secure_executor.execute_code(
            state["generated_code"].code_content,
            context,
        )

        state["execution_results"] = execution_results

        if execution_results.status.value == "success":
            if execution_results.output_data:
                state["analysis_outputs"]["python_results"] = execution_results.output_data
            state["next_step"] = "synthesize_results"
            logger.info("Code executed successfully")
        else:
            logger.warning(
                f"Code execution failed: {execution_results.error_message}"
            )
            state["error_context"]["execution_error"] = execution_results.error_message
            state["next_step"] = "handle_error"

    except Exception as e:
        logger.error(f"Error executing code: {str(e)}")
        state["error_context"]["execution_error"] = str(e)
        state["next_step"] = "handle_error"

    return state


def _generate_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate summary statistics for the dataset."""
    try:
        stats: Dict[str, Any] = {}
        numeric_cols = df.select_dtypes(include=["number"]).columns
        if len(numeric_cols) > 0:
            stats["numeric_summary"] = df[numeric_cols].describe().to_dict()

        categorical_cols = df.select_dtypes(include=["object", "string"]).columns
        if len(categorical_cols) > 0:
            stats["categorical_summary"] = {}
            for col in categorical_cols[:5]:
                value_counts = df[col].value_counts().head(10)
                stats["categorical_summary"][col] = value_counts.to_dict()

        stats["dataset_info"] = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "numeric_columns": len(numeric_cols),
            "categorical_columns": len(categorical_cols),
            "missing_values": df.isnull().sum().to_dict(),
        }

        return stats

    except Exception as e:
        logger.warning(f"Error generating summary stats: {e}")
        return {"error": "Could not generate summary statistics"}


def synthesize_results(state: AnalysisState) -> AnalysisState:
    """Synthesize analysis results using AI-driven insights."""
    logger.info("Synthesizing analysis results with AI")

    try:
        analysis_results = {}

        if state["raw_dataset"] is not None:
            df = state["raw_dataset"]
            process_type = state["process_type"]

            logger.debug(
                "Synthesizing results for %s process", process_type.value
            )
            logger.debug(
                "DataFrame shape: %s, columns: %s", df.shape, df.columns.tolist()
            )

            data_overview = {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "column_names": df.columns.tolist(),
                "data_types": df.dtypes.to_dict(),
                "sample_data": df.head(5).to_dict("records") if len(df) > 0 else [],
            }

            sql_metadata = state["analysis_outputs"].get("sql_metadata", {})
            data_understanding = state["analysis_outputs"].get(
                "data_understanding", {}
            )

            analysis_results = {
                "process_type": process_type.value,
                "query_intent": data_understanding.get(
                    "query_intent", state["user_query"]
                ),
                "data_overview": data_overview,
                "sql_explanation": sql_metadata.get(
                    "explanation", "Data retrieved successfully"
                ),
                "complexity_level": sql_metadata.get("complexity", "medium"),
                "tables_used": sql_metadata.get("tables_used", []),
                "metrics_computed": sql_metadata.get("metrics_computed", []),
                "full_dataset": df.to_dict("records")
                if len(df) <= 100
                else df.head(100).to_dict("records"),
                "data_summary_stats": _generate_summary_stats(df)
                if len(df) > 0
                else {},
            }

            logger.debug(
                "Prepared comprehensive analysis results with %s records",
                len(analysis_results["full_dataset"]),
            )
        else:
            logger.warning("No dataset available for analysis")
            analysis_results = {
                "error": "No data available for analysis",
                "process_type": state["process_type"].value
                if state.get("process_type")
                else "unknown",
                "query_intent": state["user_query"],
            }

        if "python_results" in state["analysis_outputs"]:
            analysis_results["python_analysis"] = state["analysis_outputs"][
                "python_results"
            ]

        insights = llm_service.generate_insights(
            analysis_results,
            state["user_query"],
        )

        state["insights"] = insights

        message = ConversationMessage(
            timestamp=datetime.now(),
            role="assistant",
            content=insights,
            message_type="result",
        )
        state["conversation_history"].append(message)

        state["workflow_complete"] = True
        state["next_step"] = "complete"

        logger.info("AI-driven analysis synthesis complete")

    except Exception as e:
        logger.error(f"Error synthesizing results: {str(e)}")
        state["error_context"]["synthesis_error"] = str(e)
        state["next_step"] = "handle_error"

    return state
