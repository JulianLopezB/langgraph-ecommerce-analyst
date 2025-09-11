"""AI agent for intelligent SQL query generation."""

from dataclasses import dataclass
from typing import List

from agents.process_classifier import ProcessTypeResult
from agents.schema_agent import DataUnderstanding
from agents.sql_prompting import create_sql_generation_prompt
from agents.sql_parsing import parse_sql_response
from agents.sql_validation import (
    create_fallback_sql,
    init_sql_validator,
    optimize_and_validate,
    validate_sql_with_langchain,
)
from logging_config import get_logger
from services.llm_service import GeminiService
from tracing.langsmith_setup import tracer, trace_agent_operation
from config import config
from utils.config_helpers import DATASET_ID, MAX_RESULTS

logger = get_logger(__name__)


@dataclass
class SQLGenerationResult:
    """Result from SQL generation."""
    sql_query: str
    explanation: str
    estimated_complexity: str  # low, medium, high
    optimization_applied: List[str]  # List of optimizations applied
    tables_used: List[str]
    metrics_computed: List[str]
    confidence: float


class SQLGenerationAgent:
    """AI agent that generates intelligent SQL queries based on data understanding."""
    
    def __init__(self):
        """Initialize the SQL generation agent."""
        self.llm_service = GeminiService()

        # Initialize LangChain SQL validator
        self.sql_validation_chain = init_sql_validator()

        logger.info("SQLGenerationAgent initialized")
    
    def generate_sql(self, query: str, data_understanding: DataUnderstanding, 
                    process_result: ProcessTypeResult) -> SQLGenerationResult:
        """
        Generate an optimized SQL query based on AI understanding of data and intent.
        
        Args:
            query: Original user query
            data_understanding: AI analysis of relevant data schema
            process_result: Process type classification result
            
        Returns:
            SQLGenerationResult with generated query and metadata
        """
        with trace_agent_operation(
            name="generate_intelligent_sql",
            query=query,
            target_tables=len(data_understanding.relevant_tables),
            target_metrics=len(data_understanding.target_metrics)
        ):
            logger.info(f"Generating SQL for: {query[:100]}...")
            
            try:
                # Create SQL generation prompt
                prompt = create_sql_generation_prompt(
                    query, data_understanding, process_result
                )
                
                # Generate SQL using AI
                response = self.llm_service.generate_text(prompt, temperature=0.1)  # Low temp for precision
                
                # Parse the response
                sql_result = parse_sql_response(
                    response.content, data_understanding
                )
                
                # Apply final optimizations and validation
                sql_result = optimize_and_validate(sql_result, data_understanding)
                
                # Use LangChain SQL validation for robust query checking
                sql_result.sql_query = validate_sql_with_langchain(
                    self.sql_validation_chain, sql_result.sql_query
                )
                
                # Log metrics
                tracer.log_metrics({
                    "sql_generation_success": True,
                    "query_length": len(sql_result.sql_query),
                    "tables_used": len(sql_result.tables_used),
                    "metrics_computed": len(sql_result.metrics_computed),
                    "complexity": sql_result.estimated_complexity,
                    "confidence": sql_result.confidence
                })
                
                logger.info(f"SQL generated successfully: {sql_result.estimated_complexity} complexity, "
                          f"{len(sql_result.tables_used)} tables")
                
                return sql_result
                
            except Exception as e:
                logger.error(f"Error generating SQL: {e}")
                # Return fallback SQL
                return create_fallback_sql(query, data_understanding)


# Global SQL agent instance
sql_agent = SQLGenerationAgent()
