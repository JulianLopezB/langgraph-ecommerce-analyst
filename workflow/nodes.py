"""LangGraph workflow nodes for the data analysis agent."""
from datetime import datetime
from typing import Dict, Any

import pandas as pd

from workflow.state import AnalysisState, ProcessType, ConversationMessage, AnalysisLineage, GeneratedCode
from agents.process_classifier import ProcessTypeClassifier, ProcessTypeResult
from agents.schema_agent import SchemaIntelligenceAgent
from agents.sql_agent import SQLGenerationAgent
from bq_client import BigQueryRunner
from code_generation.validators import CodeValidator
from execution.sandbox import SecureExecutor
from logging_config import get_logger
from services.llm_service import GeminiService
from tracing.langsmith_setup import LangSmithTracer, trace_agent_operation
from utils.sql_utils import clean_sql_query, format_error_message

logger = get_logger(__name__)


class WorkflowNodes:
    """Collection of LangGraph workflow nodes."""

    def __init__(
        self,
        llm_service: GeminiService,
        bq_client: BigQueryRunner,
        process_classifier: ProcessTypeClassifier,
        schema_agent: SchemaIntelligenceAgent,
        sql_agent: SQLGenerationAgent,
        validator: CodeValidator,
        secure_executor: SecureExecutor,
        tracer: LangSmithTracer,
    ):
        """Initialize workflow nodes with required services."""
        self.llm_service = llm_service
        self.bq_client = bq_client
        self.process_classifier = process_classifier
        self.schema_agent = schema_agent
        self.sql_agent = sql_agent
        self.validator = validator
        self.secure_executor = secure_executor
        self.tracer = tracer
    
    def understand_query(self, state: AnalysisState) -> AnalysisState:
        """
        Parse and understand user intent using AI agents.
        
        Args:
            state: Current analysis state
            
        Returns:
            Updated state with process type classification
        """
        with trace_agent_operation(
            self.tracer,
            name="understand_query_ai",
            user_query=state["user_query"],
            session_id=state["session_id"]
        ):
            logger.info("Understanding user query with AI agents")
        
            try:
                # Get schema information first
                schema_info = self._get_schema_info()
                state["data_schema"] = schema_info
                
                # Use AI agent to classify process type (replaces intent classification)
                process_result = self.process_classifier.classify(state["user_query"], schema_info)
                
                # Update state with process type information
                state["process_type"] = process_result.process_type
                state["confidence_score"] = process_result.confidence
                state["needs_python_analysis"] = (process_result.process_type == ProcessType.PYTHON)
                
                # Store process classification data
                state["analysis_outputs"]["process_data"] = {
                    "process_type": process_result.process_type.value,
                    "confidence": process_result.confidence,
                    "reasoning": process_result.reasoning,
                    "complexity_level": process_result.complexity_level,
                    "suggested_tables": process_result.suggested_tables
                }
                
                # Log metrics to trace
                self.tracer.log_metrics({
                    "process_type": process_result.process_type.value,
                    "confidence_score": process_result.confidence,
                    "needs_python_analysis": state["needs_python_analysis"],
                    "query_length": len(state["user_query"]),
                    "complexity_level": process_result.complexity_level
                })
                
                # Add conversation message
                message = ConversationMessage(
                    timestamp=datetime.now(),
                    role="assistant",
                    content=f"I'll handle this using {process_result.process_type.value.upper()} processing. {process_result.reasoning}",
                    message_type="query"
                )
                state["conversation_history"].append(message)
                
                # Determine next step based on process type
                if process_result.confidence > 0.7:
                    if process_result.process_type == ProcessType.SQL:
                        state["next_step"] = "generate_sql"
                    elif process_result.process_type == ProcessType.PYTHON:
                        state["next_step"] = "generate_sql"  # Still need data first
                    elif process_result.process_type == ProcessType.VISUALIZATION:
                        state["next_step"] = "generate_sql"  # Need data for visualization
                else:
                    state["next_step"] = "clarify_query"
                
                logger.info(f"Process type classified: {process_result.process_type.value} "
                          f"(confidence: {process_result.confidence:.2f})")
                
            except Exception as e:
                logger.error(f"Error understanding query: {str(e)}")
                state["error_context"]["understanding_error"] = str(e)
                state["next_step"] = "handle_error"
            
            return state
    
    def generate_sql(self, state: AnalysisState) -> AnalysisState:
        """
        Generate SQL query using AI agents for intelligent analysis.
        
        Args:
            state: Current analysis state
            
        Returns:
            Updated state with AI-generated SQL query
        """
        logger.info("Generating SQL query using AI agents")
        
        try:
            # Get process classification data
            process_data = state["analysis_outputs"]["process_data"]
            process_result = ProcessTypeResult(
                process_type=state["process_type"],
                confidence=state["confidence_score"],
                reasoning=process_data["reasoning"],
                complexity_level=process_data["complexity_level"],
                suggested_tables=process_data["suggested_tables"]
            )
            
            # Use AI agent to understand data schema semantically
            data_understanding = self.schema_agent.understand_data(
                state["user_query"],
                state["data_schema"],
                state["process_type"]
            )
            
            # Store data understanding for later use
            state["analysis_outputs"]["data_understanding"] = {
                "query_intent": data_understanding.query_intent,
                "relevant_tables": [table.name for table in data_understanding.relevant_tables],
                "target_metrics": [metric.name for metric in data_understanding.target_metrics],
                "grouping_dimensions": [dim.name for dim in data_understanding.grouping_dimensions],
                "complexity_score": data_understanding.complexity_score
            }
            
            # Use AI agent to generate intelligent SQL
            sql_result = self.sql_agent.generate_sql(
                state["user_query"],
                data_understanding,
                process_result
            )
            
            # Update state with SQL results
            state["sql_query"] = sql_result.sql_query
            state["analysis_outputs"]["sql_metadata"] = {
                "explanation": sql_result.explanation,
                "complexity": sql_result.estimated_complexity,
                "optimizations": sql_result.optimization_applied,
                "tables_used": sql_result.tables_used,
                "metrics_computed": sql_result.metrics_computed,
                "confidence": sql_result.confidence
            }
            
            # Add to conversation history
            message = ConversationMessage(
                timestamp=datetime.now(),
                role="assistant",
                content=f"Generated optimized SQL query: {sql_result.explanation}",
                message_type="query"
            )
            state["conversation_history"].append(message)
            
            state["next_step"] = "execute_sql"
            logger.info(f"AI-generated SQL complete: {sql_result.estimated_complexity} complexity, "
                       f"confidence: {sql_result.confidence:.2f}")
            
        except Exception as e:
            logger.error(f"Error generating SQL with AI agents: {str(e)}")
            state["error_context"]["sql_generation_error"] = str(e)
            state["next_step"] = "handle_error"
        
        return state
    
    def execute_sql(self, state: AnalysisState) -> AnalysisState:
        """
        Execute SQL query and retrieve data.
        
        Args:
            state: Current analysis state
            
        Returns:
            Updated state with retrieved data
        """
        logger.info("Executing SQL query")
        
        try:
            # Execute SQL query
            df = self.bq_client.execute_query(state["sql_query"])
            state["raw_dataset"] = df
            
            # Store data information for code generation
            data_info = {
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.to_dict(),
                'shape': df.shape,
                'head': df.head().to_dict(),
                'null_counts': df.isnull().sum().to_dict()
            }
            state["analysis_outputs"]["data_info"] = data_info
            
            # Add to conversation history
            message = ConversationMessage(
                timestamp=datetime.now(),
                role="assistant",
                content=f"Retrieved {len(df)} rows and {len(df.columns)} columns from BigQuery",
                message_type="result"
            )
            state["conversation_history"].append(message)
            
            # Determine next step based on analysis needs
            if state["needs_python_analysis"]:
                state["next_step"] = "generate_python_code"
            else:
                state["next_step"] = "synthesize_results"
            
            logger.info(f"SQL executed successfully: {df.shape[0]} rows, {df.shape[1]} columns")
            
        except Exception as e:
            logger.error(f"Error executing SQL: {str(e)}")
            state["error_context"]["sql_execution_error"] = str(e)
            state["next_step"] = "handle_error"
        
        return state
    
    def generate_python_code(self, state: AnalysisState) -> AnalysisState:
        """
        Generate adaptive Python code based on actual data inspection.
        
        Args:
            state: Current analysis state
            
        Returns:
            Updated state with generated code
        """
        logger.info("Generating data-adaptive Python analysis code")
        
        try:
            # First, inspect the actual data to understand its characteristics
            data_characteristics = self._inspect_data_characteristics(state["raw_dataset"])
            
            # Prepare comprehensive analysis context with data insights
            analysis_context = {
                # Original user request and intent
                "original_query": state["user_query"],
                "query_intent": state["analysis_outputs"].get("data_understanding", {}).get("query_intent", state["user_query"]),
                
                # Process classification results
                "process_data": state["analysis_outputs"]["process_data"],
                
                # SQL execution context
                "sql_query": state.get("sql_query", ""),
                "sql_explanation": state["analysis_outputs"].get("sql_metadata", {}).get("explanation", ""),
                
                # Enhanced data information with characteristics
                "data_info": state["analysis_outputs"]["data_info"],
                "data_characteristics": data_characteristics,
                
                # Schema understanding context
                "data_understanding": state["analysis_outputs"].get("data_understanding", {}),
                "sql_metadata": state["analysis_outputs"].get("sql_metadata", {})
            }
            
            # Generate Python code using data-adaptive approach
            python_code = self.llm_service.generate_adaptive_python_code(analysis_context)
            
            # Create GeneratedCode object with enhanced parameters
            generated_code = GeneratedCode(
                code_content=python_code,
                template_used=analysis_context["process_data"].get("process_type", "unknown"),
                parameters={
                    "analysis_context": analysis_context,
                    "original_query": state["user_query"],
                    "sql_query": state.get("sql_query", ""),
                    "data_characteristics": data_characteristics
                }
            )
            
            state["generated_code"] = generated_code
            state["next_step"] = "validate_code"
            
            logger.info(f"Data-adaptive Python code generated for: {state['user_query'][:50]}... "
                       f"(Data shape: {data_characteristics.get('shape', 'unknown')})")
            
        except Exception as e:
            logger.error(f"Error generating Python code: {str(e)}")
            state["error_context"]["code_generation_error"] = str(e)
            state["next_step"] = "handle_error"
        
        return state
    
    def _inspect_data_characteristics(self, df) -> Dict[str, Any]:
        """Inspect actual data to understand its characteristics for adaptive code generation."""
        if df is None or df.empty:
            return {"error": "No data available", "shape": (0, 0)}
        
        try:
            characteristics = {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "data_types": df.dtypes.to_dict(),
                "non_null_counts": df.count().to_dict(),
                "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
                "datetime_columns": [],
                "categorical_columns": df.select_dtypes(include=['object', 'string']).columns.tolist(),
                "sample_values": {}
            }
            
            # Detect potential datetime columns
            for col in df.columns:
                if df[col].dtype == 'object':
                    sample_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                    if sample_val and ('date' in col.lower() or 'time' in col.lower() or 'month' in col.lower()):
                        characteristics["datetime_columns"].append(col)
                        
                # Store sample values for understanding
                if not df[col].dropna().empty:
                    characteristics["sample_values"][col] = df[col].dropna().iloc[:3].tolist()
            
            # Assess data quality for time series
            if len(df) > 0:
                characteristics["time_series_capable"] = len(df) >= 3
                characteristics["seasonal_analysis_capable"] = len(df) >= 24
                characteristics["trend_analysis_capable"] = len(df) >= 6
                
            # Identify potential time and value columns for forecasting
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
                "columns": df.columns.tolist() if df is not None else []
            }
    
    def validate_code(self, state: AnalysisState) -> AnalysisState:
        """
        Validate generated Python code for security and syntax.
        
        Args:
            state: Current analysis state
            
        Returns:
            Updated state with validation results
        """
        logger.info("Validating generated code")
        
        try:
            if not state["generated_code"]:
                raise ValueError("No code to validate")
            
            # Validate code using security validator
            validation_result = self.validator.validate(state["generated_code"].code_content)
            state["validation_results"] = validation_result
            
            # Update generated code with validation info
            state["generated_code"].validation_passed = validation_result.is_valid
            state["generated_code"].security_score = validation_result.security_score
            
            if validation_result.is_valid:
                state["next_step"] = "execute_code"
                logger.info("Code validation passed")
            else:
                logger.warning(f"Code validation failed: {validation_result.security_warnings}")
                state["error_context"]["validation_errors"] = {
                    "syntax_errors": validation_result.syntax_errors,
                    "security_warnings": validation_result.security_warnings,
                    "security_score": validation_result.security_score
                }
                state["next_step"] = "handle_error"
            
        except Exception as e:
            logger.error(f"Error validating code: {str(e)}")
            state["error_context"]["validation_error"] = str(e)
            state["next_step"] = "handle_error"
        
        return state
    
    def execute_code(self, state: AnalysisState) -> AnalysisState:
        """
        Execute validated Python code in secure environment.
        
        Args:
            state: Current analysis state
            
        Returns:
            Updated state with execution results
        """
        logger.info("Executing Python code")
        
        try:
            if not state["generated_code"] or not state["generated_code"].validation_passed:
                raise ValueError("Invalid or unvalidated code")
            
            # Prepare execution context with data
            context = {
                'df': state["raw_dataset"]
            }
            
            # Execute code in secure environment
            execution_results = self.secure_executor.execute_code(
                state["generated_code"].code_content,
                context
            )
            
            state["execution_results"] = execution_results
            
            if execution_results.status.value == "success":
                # Store analysis results
                if execution_results.output_data:
                    state["analysis_outputs"]["python_results"] = execution_results.output_data
                
                state["next_step"] = "synthesize_results"
                logger.info("Code executed successfully")
            else:
                logger.warning(f"Code execution failed: {execution_results.error_message}")
                state["error_context"]["execution_error"] = execution_results.error_message
                state["next_step"] = "handle_error"
            
        except Exception as e:
            logger.error(f"Error executing code: {str(e)}")
            state["error_context"]["execution_error"] = str(e)
            state["next_step"] = "handle_error"
        
        return state
    
    def synthesize_results(self, state: AnalysisState) -> AnalysisState:
        """
        Synthesize analysis results using AI-driven insights (no keyword matching).
        
        Args:
            state: Current analysis state
            
        Returns:
            Updated state with AI-generated insights
        """
        logger.info("Synthesizing analysis results with AI")
        
        try:
            # Prepare comprehensive results for AI insight generation
            analysis_results = {}
            
            if state["raw_dataset"] is not None:
                df = state["raw_dataset"]
                process_type = state["process_type"]
                
                logger.debug(f"Synthesizing results for {process_type.value} process")
                logger.debug(f"DataFrame shape: {df.shape}, columns: {df.columns.tolist()}")
                
                # Prepare data for AI analysis (no keyword matching!)
                # Let the AI understand the data context directly
                
                # Get basic data overview
                data_overview = {
                    "total_rows": len(df),
                    "total_columns": len(df.columns),
                    "column_names": df.columns.tolist(),
                    "data_types": df.dtypes.to_dict(),
                    "sample_data": df.head(5).to_dict('records') if len(df) > 0 else []
                }
                
                # Include SQL metadata if available
                sql_metadata = state["analysis_outputs"].get("sql_metadata", {})
                data_understanding = state["analysis_outputs"].get("data_understanding", {})
                
                # Structure comprehensive results for AI insight generation
                analysis_results = {
                    "process_type": process_type.value,
                    "query_intent": data_understanding.get("query_intent", state["user_query"]),
                    "data_overview": data_overview,
                    "sql_explanation": sql_metadata.get("explanation", "Data retrieved successfully"),
                    "complexity_level": sql_metadata.get("complexity", "medium"),
                    "tables_used": sql_metadata.get("tables_used", []),
                    "metrics_computed": sql_metadata.get("metrics_computed", []),
                    "full_dataset": df.to_dict('records') if len(df) <= 100 else df.head(100).to_dict('records'),
                    "data_summary_stats": self._generate_summary_stats(df) if len(df) > 0 else {}
                }
                
                logger.debug(f"Prepared comprehensive analysis results with {len(analysis_results['full_dataset'])} records")
                
            else:
                logger.warning("No dataset available for analysis")
                analysis_results = {
                    "error": "No data available for analysis",
                    "process_type": state["process_type"].value if state.get("process_type") else "unknown",
                    "query_intent": state["user_query"]
                }
            
            # Add Python analysis results if available
            if "python_results" in state["analysis_outputs"]:
                analysis_results["python_analysis"] = state["analysis_outputs"]["python_results"]
            
            # Use AI to generate comprehensive insights
            insights = self.llm_service.generate_insights(
                analysis_results,
                state["user_query"]
            )
            
            state["insights"] = insights
            
            # Add final message to conversation
            message = ConversationMessage(
                timestamp=datetime.now(),
                role="assistant",
                content=insights,
                message_type="result"
            )
            state["conversation_history"].append(message)
            
            # Mark workflow as complete
            state["workflow_complete"] = True
            state["next_step"] = "complete"
            
            logger.info("AI-driven analysis synthesis complete")
            
        except Exception as e:
            logger.error(f"Error synthesizing results: {str(e)}")
            state["error_context"]["synthesis_error"] = str(e)
            state["next_step"] = "handle_error"
        
        return state
    
    def handle_error(self, state: AnalysisState) -> AnalysisState:
        """
        Handle errors and provide recovery options.
        
        Args:
            state: Current analysis state
            
        Returns:
            Updated state with error handling
        """
        logger.info("Handling workflow error")
        
        try:
            error_messages = []
            for error_type, error_msg in state["error_context"].items():
                error_messages.append(f"{error_type}: {error_msg}")
            
            # Generate error explanation and suggestions
            error_summary = "; ".join(error_messages)
            
            # Simple error recovery suggestions
            suggestions = []
            if "sql" in error_summary.lower():
                suggestions.append("Try rephrasing your question with more specific details")
            if "validation" in error_summary.lower():
                suggestions.append("The analysis requires simpler operations")
            if "execution" in error_summary.lower():
                suggestions.append("The analysis may need a smaller dataset")
            
            error_response = f"I encountered an issue: {error_summary}"
            if suggestions:
                error_response += f"\\n\\nSuggestions: {'; '.join(suggestions)}"
            
            # Add error message to conversation
            message = ConversationMessage(
                timestamp=datetime.now(),
                role="assistant",
                content=error_response,
                message_type="error"
            )
            state["conversation_history"].append(message)
            
            # Mark workflow as complete (with error)
            state["workflow_complete"] = True
            state["next_step"] = "complete"
            
        except Exception as e:
            logger.error(f"Error in error handler: {str(e)}")
            # Fallback error message
            message = ConversationMessage(
                timestamp=datetime.now(),
                role="assistant",
                content="I encountered an unexpected error during analysis. Please try again with a simpler request.",
                message_type="error"
            )
            state["conversation_history"].append(message)
            state["workflow_complete"] = True
            state["next_step"] = "complete"
        
        return state
    
    def _get_schema_info(self) -> Dict[str, Any]:
        """Get schema information for core tables.

        Returns:
            Dictionary mapping table names to schema metadata with a ``columns`` key
            containing a list of column definitions.
        """
        try:
            schema_info: Dict[str, Any] = {}

            # Always include core tables
            core_tables = ["orders", "order_items", "products", "users"]

            for table in core_tables:
                try:
                    columns = self.bq_client.get_table_schema(table)
                    schema_info[table] = {"columns": columns}
                except Exception as e:
                    logger.warning(f"Could not get schema for {table}: {e}")

            return schema_info

        except Exception as e:
            logger.error(f"Error getting schema info: {e}")
            return {}
    
    def _generate_summary_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics for the dataset."""
        try:
            stats = {}
            
            # Numeric columns statistics
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                stats["numeric_summary"] = df[numeric_cols].describe().to_dict()
            
            # Categorical columns statistics
            categorical_cols = df.select_dtypes(include=['object', 'string']).columns
            if len(categorical_cols) > 0:
                stats["categorical_summary"] = {}
                for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
                    value_counts = df[col].value_counts().head(10)
                    stats["categorical_summary"][col] = value_counts.to_dict()
            
            # Overall dataset statistics
            stats["dataset_info"] = {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "numeric_columns": len(numeric_cols),
                "categorical_columns": len(categorical_cols),
                "missing_values": df.isnull().sum().to_dict()
            }
            
            return stats
            
        except Exception as e:
            logger.warning(f"Error generating summary stats: {e}")
            return {"error": "Could not generate summary statistics"}
    
