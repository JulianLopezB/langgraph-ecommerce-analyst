"""LLM service for Google Gemini integration."""
import os
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from config import config
from logging_config import get_logger
from tracing.langsmith_setup import tracer, trace_llm_operation
from services.intent_models import IntentClassificationResult, CLASSIFY_INTENT_FUNCTION

logger = get_logger(__name__)


@dataclass
class LLMResponse:
    """Response from LLM service."""
    content: str
    tokens_used: int = 0
    response_time: float = 0.0
    model_used: str = ""
    confidence: float = 0.0


class GeminiService:
    """Service for interacting with Google Gemini API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Gemini service."""
        self.api_key = api_key or config.api_configurations.gemini_api_key
        if not self.api_key:
            raise ValueError("Gemini API key not provided")
        
        genai.configure(api_key=self.api_key)
        
        # Configure safety settings for code generation
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }
        
        self.model = genai.GenerativeModel(
            model_name=os.getenv('GEMINI_MODEL', 'gemini-1.5-flash'),
            safety_settings=self.safety_settings
        )
        
        logger.info("Gemini service initialized")
    
    def generate_text(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2048) -> LLMResponse:
        """Generate text response from Gemini."""
        start_time = time.time()
        
        with trace_llm_operation(
            name="gemini_generate_text",
            model="gemini-1.5-flash",
            temperature=temperature,
            max_tokens=max_tokens,
            prompt_length=len(prompt)
        ):
            try:
                generation_config = genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    candidate_count=1
                )
                
                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
            
                if response.candidates and response.candidates[0].content.parts:
                    content = response.candidates[0].content.parts[0].text
                    response_time = time.time() - start_time
                    tokens_used = self._estimate_tokens(prompt + content)
                    
                    # Log metrics to trace
                    tracer.log_metrics({
                        "response_time": response_time,
                        "tokens_used": tokens_used,
                        "content_length": len(content),
                        "prompt_length": len(prompt)
                    })
                    
                    return LLMResponse(
                        content=content,
                        tokens_used=tokens_used,
                        response_time=response_time,
                        model_used="gemini-1.5-flash"
                    )
                else:
                    logger.warning("Empty response from Gemini API")
                    return LLMResponse(content="", response_time=time.time() - start_time)
                    
            except Exception as e:
                logger.error(f"Error generating text with Gemini: {str(e)}")
                raise
    
    def classify_intent(self, user_query: str) -> IntentClassificationResult:
        """Classify user intent for analysis routing using function calling."""
        with trace_llm_operation(
            name="gemini_classify_intent",
            model="gemini-1.5-flash",
            query_length=len(user_query)
        ):
            prompt = f"""
            Analyze the following user query and classify the intent for data analysis:

            Query: "{user_query}"

            Consider these intent categories:
            - data_exploration: Basic data inspection, summaries, data quality checks
            - customer_analysis: Customer segmentation, behavior analysis, churn, CLV  
            - product_analysis: Product performance, recommendations, inventory
            - sales_analysis: Sales trends, forecasting, revenue analysis
            - advanced_analytics: Machine learning, statistical modeling, complex algorithms
            - visualization: Creating charts, graphs, dashboards

            Determine:
            1. The most appropriate intent category
            2. Confidence score (0.0-1.0) 
            3. Whether Python analysis is needed beyond SQL
            4. Key entities mentioned (customers, products, dates, metrics)
            5. Brief description of the analysis type
            6. Complexity level (low/medium/high)

            Use the classify_user_intent function to provide structured output.
            """
            
            try:
                # Configure model for function calling
                function_calling_model = genai.GenerativeModel(
                    model_name=os.getenv('GEMINI_MODEL', 'gemini-1.5-flash'),
                    safety_settings=self.safety_settings,
                    tools=[{"function_declarations": [CLASSIFY_INTENT_FUNCTION]}]
                )
                
                response = function_calling_model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.3,
                        max_output_tokens=1024
                    )
                )
                
                # Extract function call result
                if response.candidates and response.candidates[0].content.parts:
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, 'function_call') and part.function_call:
                            function_call = part.function_call
                            if function_call.name == "classify_user_intent":
                                # Convert function call args to Pydantic model
                                try:
                                    args = dict(function_call.args)
                                    logger.debug(f"Function call args: {args}")
                                    
                                    # Handle IntentType conversion
                                    if 'intent' in args:
                                        from agent.state import IntentType
                                        intent_value = args['intent']
                                        args['intent'] = IntentType(intent_value)
                                    
                                    result = IntentClassificationResult(**args)
                                    
                                    # Log successful function calling metrics
                                    tracer.log_metrics({
                                        "intent_parsing_success": True,
                                        "classified_intent": result.intent.value,
                                        "confidence_score": result.confidence,
                                        "function_calling_used": True
                                    })
                                    
                                    return result
                                except Exception as e:
                                    logger.error(f"Error creating IntentClassificationResult: {e}")
                                    logger.debug(f"Function call args were: {args}")
                                    # Continue to fallback
                
                # Fallback if no function call found
                logger.warning("No function call found in response, using fallback")
                tracer.log_metrics({
                    "intent_parsing_success": False,
                    "function_calling_used": False,
                    "fallback_reason": "no_function_call"
                })
                
                from agent.state import IntentType
                return IntentClassificationResult(
                    intent=IntentType.UNKNOWN,
                    confidence=0.0,
                    needs_python=True,
                    entities=[],
                    analysis_type="unknown",
                    complexity="high"
                )
                
            except Exception as e:
                logger.error(f"Error in function calling intent classification: {e}")
                
                # Log error metrics
                tracer.log_metrics({
                    "intent_parsing_success": False,
                    "function_calling_used": True,
                    "error": str(e)
                })
                
                from agent.state import IntentType
                return IntentClassificationResult(
                    intent=IntentType.UNKNOWN,
                    confidence=0.0,
                    needs_python=True,
                    entities=[],
                    analysis_type="unknown",
                    complexity="high"
                )
    
    def generate_sql_query(self, intent_data: Dict[str, Any], schema_info: Dict[str, Any]) -> str:
        """Generate SQL query based on user intent and schema."""
        with trace_llm_operation(
            name="gemini_generate_sql",
            model="gemini-1.5-flash",
            intent_type=intent_data.get("intent", "unknown"),
            schema_tables=len(schema_info)
        ):
            prompt = f"""
        Generate a BigQuery SQL query for this analysis request:

        Intent: {intent_data['intent']}
        Analysis Type: {intent_data['analysis_type']}
        Entities: {intent_data['entities']}
        Complexity: {intent_data['complexity']}

        Available tables and schema:
        {schema_info}

        Requirements:
        1. Use the dataset `bigquery-public-data.thelook_ecommerce`
        2. Optimize for performance and cost
        3. Limit results to reasonable size (10,000 rows max)
        4. Include meaningful column aliases
        5. Add appropriate filters and constraints

        Generate ONLY the SQL query without explanations or markdown formatting.
        """
            
            response = self.generate_text(prompt, temperature=0.2)
            sql_query = response.content.strip()
            
            # Log SQL generation metrics
            tracer.log_metrics({
                "sql_generation_prompt_length": len(prompt),
                "generated_sql_length": len(sql_query),
                "intent_type": intent_data.get("intent", "unknown"),
                "complexity": intent_data.get("complexity", "unknown")
            })
            
            return sql_query
    
    def generate_python_code(self, intent_data: Dict[str, Any], data_info: Dict[str, Any]) -> str:
        """Generate Python code for advanced analysis."""
        with trace_llm_operation(
            name="gemini_generate_python",
            model="gemini-1.5-flash",
            intent_type=intent_data.get("intent", "unknown"),
            data_columns=len(data_info.get("columns", []))
        ):
            prompt = f"""
        Generate Python code for this data analysis task:

        Intent: {intent_data['intent']}
        Analysis Type: {intent_data['analysis_type']}
        Complexity: {intent_data['complexity']}

        Data Information:
        - DataFrame variable name: df
        - Columns: {data_info.get('columns', [])}
        - Data types: {data_info.get('dtypes', {})}
        - Shape: {data_info.get('shape', 'unknown')}

        Requirements:
        1. Use only allowed libraries: pandas, numpy, matplotlib, seaborn, plotly, sklearn, scipy, statsmodels
        2. Assume DataFrame 'df' is already available
        3. Create meaningful visualizations where appropriate
        4. Generate insights and summary statistics
        5. Handle missing data appropriately
        6. Include print statements for key findings
        7. Use descriptive variable names

        Generate ONLY the Python code without explanations or markdown formatting.
        Store final results in a variable called 'analysis_results'.
        """
            
            response = self.generate_text(prompt, temperature=0.4)
            python_code = response.content.strip()
            
            # Log Python code generation metrics
            tracer.log_metrics({
                "python_generation_prompt_length": len(prompt),
                "generated_code_length": len(python_code),
                "intent_type": intent_data.get("intent", "unknown"),
                "data_shape": str(data_info.get("shape", "unknown"))
            })
            
            return python_code
    
    def generate_insights(self, analysis_results: Dict[str, Any], original_query: str) -> str:
        """Generate business insights from analysis results."""
        with trace_llm_operation(
            name="gemini_generate_insights",
            model="gemini-1.5-flash",
            query_length=len(original_query),
            results_keys=len(analysis_results.keys())
        ):
            prompt = f"""
        Generate actionable business insights based on this data analysis:

        Original Question: "{original_query}"

        Analysis Results:
        {analysis_results}

        Provide:
        1. Key findings (3-5 bullet points)
        2. Business implications
        3. Recommended actions
        4. Any limitations or caveats

        Format as clear, business-friendly language suitable for stakeholders.
        """
            
            response = self.generate_text(prompt, temperature=0.6)
            insights = response.content.strip()
            
            # Log insight generation metrics
            tracer.log_metrics({
                "insights_generation_prompt_length": len(prompt),
                "generated_insights_length": len(insights),
                "original_query_length": len(original_query),
                "analysis_data_points": len(analysis_results.get("processed_data", [])) if "processed_data" in analysis_results else 0
            })
            
            return insights
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        return len(text.split()) * 1.3  # Rough estimate
