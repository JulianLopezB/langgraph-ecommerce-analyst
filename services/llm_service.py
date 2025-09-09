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
    
    # Removed old classify_intent method - now using AI agents for process type classification
    
    # Removed old generate_sql_query method - now using AI agents for intelligent SQL generation
    
    def generate_adaptive_python_code(self, analysis_context: Dict[str, Any]) -> str:
        """Generate adaptive Python code based on actual data characteristics."""
        with trace_llm_operation(
            name="gemini_generate_adaptive_python",
            model="gemini-1.5-flash",
            original_query=analysis_context.get("original_query", "unknown")[:50],
            data_shape=str(analysis_context.get("data_characteristics", {}).get("shape", "unknown"))
        ):
            # Extract context information
            original_query = analysis_context.get("original_query", "")
            process_data = analysis_context.get("process_data", {})
            data_characteristics = analysis_context.get("data_characteristics", {})
            sql_explanation = analysis_context.get("sql_explanation", "")
            query_intent = analysis_context.get("query_intent", "")
            
            prompt = f"""
        Generate Python code that adapts to the actual data characteristics:

        USER REQUEST: "{original_query}"
        INTENT: {query_intent}
        
        ACTUAL DATA CHARACTERISTICS:
        - Shape: {data_characteristics.get('shape', 'unknown')}
        - Columns: {data_characteristics.get('columns', [])}
        - Data Types: {str(data_characteristics.get('data_types', {}))}
        - Numeric Columns: {data_characteristics.get('numeric_columns', [])}
        - DateTime Columns: {data_characteristics.get('datetime_columns', [])}
        - Sample Values: {str(data_characteristics.get('sample_values', {}))}
        - Forecasting Ready: {data_characteristics.get('forecasting_ready', False)}
        
        GENERATE ROBUST CODE THAT:
        1. Works with THIS specific data structure
        2. Uses appropriate algorithms for the data size
        3. Handles the actual column names and types
        4. Creates meaningful visualizations
        5. Provides insights relevant to the user's request
        
        FOR FORECASTING SPECIFICALLY:
        - Use whatever fits the data (linear, polynomial, exponential models)
        - Plot historical data and forecasted trend
        - Don't force complex models if data is limited
        - Generate 3-month projections with confidence bounds
        - Be pragmatic: use what works, not what's theoretically best

        Requirements:
        1. Use allowed libraries: pandas, numpy, matplotlib, seaborn, plotly, sklearn, scipy
        2. DataFrame 'df' is already available with the columns shown above
        3. Use try-except blocks for robustness
        4. Generate insights and summary statistics
        5. Create visualizations that help understand the data and results
        6. Store final results in 'analysis_results' variable
        
        CODE FORMATTING:
        - Generate ONLY valid Python code, no explanations
        - Start with import statements
        - End with: analysis_results = dict with key_findings, forecast, and plots
        - Use print() statements to show progress and findings
        """
            
            response = self.generate_text(prompt, temperature=0.4)
            python_code = response.content.strip()
            
            # Clean up any markdown formatting that might slip through
            python_code = self._clean_python_code(python_code)
            
            # Log Python code generation metrics
            tracer.log_metrics({
                "python_generation_prompt_length": len(prompt),
                "generated_code_length": len(python_code),
                "process_type": process_data.get("process_type", "unknown"),
                "data_shape": str(data_characteristics.get("shape", "unknown"))
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
        Generate clear, actionable business insights based on this data analysis:

        Question: "{original_query}"
        Analysis Results: {analysis_results}

        Provide exactly two sections:

        Key Findings:
        • [List the most important discoveries from the data]
        • [Include as many findings as are truly significant - don't limit to 3]
        • [Focus on what the data actually shows]

        Business Impact:
        [Write a holistic analysis of what these findings mean for the business]
        [Explain the broader implications and strategic considerations]
        [Keep it conversational and insightful]

        Use clear, friendly language. Avoid technical jargon.
        Don't include limitations, caveats, or next steps - focus on insights and impact.
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
    
    def _clean_python_code(self, code: str) -> str:
        """Clean Python code by removing markdown formatting."""
        # Remove markdown code blocks
        if code.startswith("```python"):
            code = code[9:]
        elif code.startswith("```"):
            code = code[3:]
        
        if code.endswith("```"):
            code = code[:-3]
        
        # Remove any leading/trailing whitespace
        code = code.strip()
        
        # Remove any explanation text before the first import or assignment
        lines = code.split('\n')
        code_start_idx = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            if (line.startswith('import ') or 
                line.startswith('from ') or 
                line.startswith('#') or
                '=' in line):
                code_start_idx = i
                break
        
        if code_start_idx > 0:
            code = '\n'.join(lines[code_start_idx:])
        
        return code.strip()
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        return len(text.split()) * 1.3  # Rough estimate
