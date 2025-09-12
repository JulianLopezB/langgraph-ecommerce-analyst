"""Google Gemini implementation of :class:`LLMClient`."""
import os
from infrastructure.secret_manager import get_env_or_secret
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from infrastructure.logging import get_logger
from tracing.langsmith_setup import tracer, trace_llm_operation
from .base import LLMClient

logger = get_logger(__name__)


@dataclass
class LLMResponse:
    """Response from LLM service."""
    content: str
    tokens_used: int = 0
    response_time: float = 0.0
    model_used: str = ""
    confidence: float = 0.0


class GeminiClient(LLMClient):
    """Client for interacting with Google Gemini API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Gemini service."""
        self.api_key = api_key or get_env_or_secret("GEMINI_API_KEY", "GEMINI_SECRET_NAME")
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
    
    def generate_text(self, prompt: str, temperature: float = 0.3, max_tokens: int = 2048) -> LLMResponse:
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
        
        USER REQUEST: "{original_query}"
        
        CRITICAL: Create code that provides EXACTLY what the user asked for with DETAILED results.
        
        WHAT THE USER WANTS VS WHAT TO GENERATE:
        
        "RFM analysis" or "segment customers":
        → Calculate R, F, M scores (1-5) using quintiles for each customer
        → Create named segments: Champions (555), Loyal (444), Potential (543), At Risk (211), etc.
        → Count customers in each segment: "Champions: 850 customers (8.5%), avg $420 spend"
        → Show segment table with counts, percentages, and average metrics per segment
        
        "Top X users" or "show me users":
        → Return actual user IDs, names, and their specific metrics in a detailed table
        → Include total spending, number of purchases, average order value, last purchase date
        → Show exact numbers for each user (purchases, spending, etc.)
        → Format as: "User ID 12345 (John Doe): 7 purchases, $2,850 total, $407 avg order, last: 15 days ago"
        
        "Product performance" or "top products":
        → List actual product names, IDs, and performance metrics
        → Include detailed rankings with specific numbers
        
        "Forecast" or "predict":
        → Generate actual future predictions with dates and values
        → Include confidence intervals and trend analysis
        
        GENERATE COMPREHENSIVE, DETAILED RESULTS - NOT SUMMARY STATISTICS!
        
        Use columns: {data_characteristics.get('columns', [])}
        Data shape: {data_characteristics.get('shape', 'unknown')}

        Requirements:
        1. Use allowed libraries: pandas, numpy, matplotlib, seaborn, plotly, sklearn, scipy
        2. DataFrame 'df' is already available with the columns shown above
        3. Use try-except blocks for robustness
        4. ALWAYS check data types before applying operations (use df.dtypes)
        5. Handle date/datetime conversions carefully (check if column is actually datetime)
        6. Generate insights and summary statistics
        7. Create visualizations that help understand the data and results
        8. Store final results in 'analysis_results' variable
        
        CODE FORMATTING:
        - Generate ONLY valid Python code, no explanations
        - Start with import statements
        - Include print() statements showing key results and insights
        - Focus on business-relevant findings, not calculation details
        - End with: analysis_results = dict with key_findings and detailed_results
        - Avoid explaining basic operations like sorting or percentage calculations
        """
            
            response = self.generate_text(prompt, temperature=0.2)
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
        Generate concise business insights with specific numbers and percentages:

        Question: "{original_query}"
        Analysis Results: {analysis_results}

        Format exactly like this:

        Key Findings:
        • [Specific finding with actual numbers/percentages/names/IDs]
        • [Another finding with concrete data and details]
        • [One more key insight with comprehensive metrics]

        Business Impact:
        [2-3 sentences explaining strategic implications and opportunities]

        Requirements:
        - Include specific numbers, percentages, dollar amounts, names, and IDs from the data
        - Show detailed results that directly answer the user's question
        - For segments: include segment names, counts, and characteristics  
        - For top users/products: include actual IDs, names, and metrics
        - For recommendations: explain the business reasoning (not technical methodology)
        - Use concrete data points with comprehensive details
        - Make findings actionable with specific insights
        - Focus on business insights, not calculation methods
        - Avoid explaining basic mathematical operations or sorting methods
        """
            
            response = self.generate_text(prompt, temperature=0.2)
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
        """Clean Python code by removing markdown formatting and fixing syntax issues."""
        # Remove markdown code blocks
        if code.startswith("```python"):
            code = code[9:]
        elif code.startswith("```"):
            code = code[3:]
        
        if code.endswith("```"):
            code = code[:-3]
        
        # Remove any leading/trailing whitespace
        code = code.strip()
        
        # Fix unterminated string literals by removing incomplete lines
        lines = code.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip lines that might have unterminated strings
            if line.count('"') % 2 != 0 and not line.strip().startswith('#'):
                # Try to fix by removing trailing content
                if '"' in line:
                    quote_pos = line.rfind('"')
                    line = line[:quote_pos + 1]
            
            cleaned_lines.append(line)
        
        code = '\n'.join(cleaned_lines)
        
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