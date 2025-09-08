"""LangGraph workflow nodes for the data analysis agent."""
from typing import Dict, Any
import json
import pandas as pd
from datetime import datetime

from agent.state import AnalysisState, IntentType, ConversationMessage, AnalysisLineage, GeneratedCode
from services.llm_service import GeminiService
from code_generation.validators import validator
from execution.sandbox import secure_executor
from bq_client import BigQueryRunner
from logging_config import get_logger

logger = get_logger(__name__)


class WorkflowNodes:
    """Collection of LangGraph workflow nodes."""
    
    def __init__(self):
        """Initialize workflow nodes with required services."""
        self.llm_service = GeminiService()
        self.bq_client = BigQueryRunner()
    
    def understand_query(self, state: AnalysisState) -> AnalysisState:
        """
        Parse and understand user intent from the query.
        
        Args:
            state: Current analysis state
            
        Returns:
            Updated state with intent classification
        """
        logger.info("Understanding user query")
        
        try:
            # Classify user intent using LLM
            intent_data = self.llm_service.classify_intent(state["user_query"])
            
            # Update state with intent information
            state["intent_classification"] = IntentType(intent_data.get("intent", "unknown"))
            state["confidence_score"] = intent_data.get("confidence", 0.0)
            state["needs_python_analysis"] = intent_data.get("needs_python", True)
            
            # Store intent data for later use
            state["analysis_outputs"]["intent_data"] = intent_data
            
            # Add conversation message
            message = ConversationMessage(
                timestamp=datetime.now(),
                role="assistant",
                content=f"I understand you want to perform {intent_data.get('analysis_type', 'analysis')}. Classification confidence: {intent_data.get('confidence', 0.0):.2f}",
                message_type="query"
            )
            state["conversation_history"].append(message)
            
            # Determine next step
            if state["confidence_score"] > 0.7:
                state["next_step"] = "generate_sql"
            else:
                state["next_step"] = "clarify_query"
            
            logger.info(f"Intent classified: {state['intent_classification']} (confidence: {state['confidence_score']:.2f})")
            
        except Exception as e:
            logger.error(f"Error understanding query: {str(e)}")
            state["error_context"]["understanding_error"] = str(e)
            state["next_step"] = "handle_error"
        
        return state
    
    def generate_sql(self, state: AnalysisState) -> AnalysisState:
        """
        Generate SQL query for data retrieval.
        
        Args:
            state: Current analysis state
            
        Returns:
            Updated state with SQL query
        """
        logger.info("Generating SQL query")
        
        try:
            # Get schema information for relevant tables
            schema_info = self._get_schema_info(state["intent_classification"])
            state["data_schema"] = schema_info
            
            # Generate SQL query using LLM
            intent_data = state["analysis_outputs"]["intent_data"]
            sql_query = self.llm_service.generate_sql_query(intent_data, schema_info)
            
            # Clean and validate SQL
            sql_query = self._clean_sql_query(sql_query)
            state["sql_query"] = sql_query
            
            # Add to conversation history
            message = ConversationMessage(
                timestamp=datetime.now(),
                role="assistant",
                content=f"Generated SQL query for data retrieval",
                message_type="query"
            )
            state["conversation_history"].append(message)
            
            state["next_step"] = "execute_sql"
            logger.info("SQL query generated successfully")
            
        except Exception as e:
            logger.error(f"Error generating SQL: {str(e)}")
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
            intent_data = state["analysis_outputs"]["intent_data"]
            data_info = state["analysis_outputs"]["data_info"]
            
            # Generate Python code using LLM
            python_code = self.llm_service.generate_python_code(intent_data, data_info)
            
            # Create GeneratedCode object
            generated_code = GeneratedCode(
                code_content=python_code,
                template_used=intent_data.get("intent", "unknown"),
                parameters={"intent": intent_data, "data_info": data_info}
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
        Synthesize analysis results and generate insights.
        
        Args:
            state: Current analysis state
            
        Returns:
            Updated state with final insights
        """
        logger.info("Synthesizing analysis results")
        
        try:
            # Prepare results for insight generation
            analysis_results = {}
            
            # Add data summary
            if state["raw_dataset"] is not None:
                df = state["raw_dataset"]
                analysis_results["data_summary"] = {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "column_names": df.columns.tolist()
                }
            
            # Add Python analysis results if available
            if "python_results" in state["analysis_outputs"]:
                analysis_results["analysis_results"] = state["analysis_outputs"]["python_results"]
            
            # Generate insights using LLM
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
            
            logger.info("Analysis results synthesized successfully")
            
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
    
    def _get_schema_info(self, intent: IntentType) -> Dict[str, Any]:
        """Get relevant schema information based on intent."""
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
