"""LangGraph workflow nodes for the data analysis agent."""
from typing import Dict, Any
import json
import pandas as pd
from datetime import datetime

from agent.state import AnalysisState, ProcessType, ConversationMessage, AnalysisLineage, GeneratedCode
from services.llm_service import GeminiService
from code_generation.validators import validator
from execution.sandbox import secure_executor
from bq_client import BigQueryRunner
from logging_config import get_logger
from tracing.langsmith_setup import tracer, trace_agent_operation
from agents.process_classifier import process_classifier, ProcessTypeResult
from agents.schema_agent import schema_agent
from agents.sql_agent import sql_agent

logger = get_logger(__name__)


class WorkflowNodes:
    """Collection of LangGraph workflow nodes."""
    
    def __init__(self):
        """Initialize workflow nodes with required services."""
        self.llm_service = GeminiService()
        self.bq_client = BigQueryRunner()
    
    def understand_query(self, state: AnalysisState) -> AnalysisState:
        """
        Parse and understand user intent using AI agents.
        
        Args:
            state: Current analysis state
            
        Returns:
            Updated state with process type classification
        """
        with trace_agent_operation(
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
                process_result = process_classifier.classify(state["user_query"], schema_info)
                
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
                tracer.log_metrics({
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
            data_understanding = schema_agent.understand_data(
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
            sql_result = sql_agent.generate_sql(
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
        Generate Python code for advanced analysis.
        
        Args:
            state: Current analysis state
            
        Returns:
            Updated state with generated code
        """
        logger.info("Generating Python analysis code")
        
        try:
            process_data = state["analysis_outputs"]["process_data"]
            data_info = state["analysis_outputs"]["data_info"]
            
            # Generate Python code using LLM
            python_code = self.llm_service.generate_python_code(process_data, data_info)
            
            # Create GeneratedCode object
            generated_code = GeneratedCode(
                code_content=python_code,
                template_used=process_data.get("process_type", "unknown"),
                parameters={"process_data": process_data, "data_info": data_info}
            )
            
            state["generated_code"] = generated_code
            state["next_step"] = "validate_code"
            
            logger.info("Python code generated successfully")
            
        except Exception as e:
            logger.error(f"Error generating Python code: {str(e)}")
            state["error_context"]["code_generation_error"] = str(e)
            state["next_step"] = "handle_error"
        
        return state
    
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
            validation_result = validator.validate(state["generated_code"].code_content)
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
            execution_results = secure_executor.execute_code(
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
        """Get schema information for core tables."""
        try:
            schema_info = {}
            
            # Always include core tables
            core_tables = ["orders", "order_items", "products", "users"]
            
            for table in core_tables:
                try:
                    schema = self.bq_client.get_table_schema(table)
                    schema_info[table] = schema
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
    
    def _clean_sql_query(self, sql_query: str) -> str:
        """Clean and format SQL query."""
        # Remove markdown formatting if present
        sql_query = sql_query.strip()
        if sql_query.startswith("```sql"):
            sql_query = sql_query[6:]
        if sql_query.startswith("```"):
            sql_query = sql_query[3:]
        if sql_query.endswith("```"):
            sql_query = sql_query[:-3]
        
        # Add dataset prefix if missing
        if "bigquery-public-data.thelook_ecommerce" not in sql_query:
            # Simple replacement for common table references
            for table in ["orders", "order_items", "products", "users"]:
                sql_query = sql_query.replace(f" {table} ", f" `bigquery-public-data.thelook_ecommerce.{table}` ")
                sql_query = sql_query.replace(f"FROM {table}", f"FROM `bigquery-public-data.thelook_ecommerce.{table}`")
                sql_query = sql_query.replace(f"JOIN {table}", f"JOIN `bigquery-public-data.thelook_ecommerce.{table}`")
        
        return sql_query.strip()


# Global nodes instance
workflow_nodes = WorkflowNodes()
