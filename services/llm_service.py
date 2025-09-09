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
    
    def generate_python_code(self, process_data: Dict[str, Any], data_info: Dict[str, Any]) -> str:
        """Generate Python code for advanced analysis."""
        with trace_llm_operation(
            name="gemini_generate_python",
            model="gemini-1.5-flash",
            process_type=process_data.get("process_type", "unknown"),
            data_columns=len(data_info.get("columns", []))
        ):
            prompt = f"""
        Generate Python code for this data analysis task:

        Process Type: {process_data['process_type']}
        Reasoning: {process_data['reasoning']}
        Complexity: {process_data['complexity_level']}

        Data Information:
        - DataFrame variable name: df
        - Columns: {data_info.get('columns', [])}
        - Data types: {data_info.get('dtypes', {})}
        - Shape: {data_info.get('shape', 'unknown')}

        SPECIFIC TASK: If this is RFM analysis, create customer segments using:
        1. Calculate RFM quintiles for each metric
        2. Create RFM segments (e.g., Champions, Potential Loyalists, etc.)
        3. Generate segment analysis and recommendations

        Requirements:
        1. Use only allowed libraries: pandas, numpy, matplotlib, seaborn, plotly, sklearn, scipy, statsmodels
        2. Assume DataFrame 'df' is already available
        3. Create meaningful visualizations where appropriate
        4. Generate insights and summary statistics
        5. Handle missing data appropriately
        6. Include print statements for key findings
        7. Use descriptive variable names

        CRITICAL - Robust Data Handling:
        8. When using pd.qcut() or pd.cut(), always set duplicates='drop' to handle duplicate bin edges
        9. Check for sufficient data variance before binning (use .nunique() > 1)
        10. Handle edge cases where data has limited variation or small sample sizes
        11. Use try-except blocks for potentially problematic operations like binning
        12. For RFM scoring, use simple numeric scoring (1-5) instead of accessing .codes on intervals
        13. Convert categorical results to strings/integers for easier processing
        
        Example robust binning and scoring:
        try:
            if column.nunique() > 5:
                df['binned'] = pd.qcut(column, q=5, duplicates='drop', labels=False) + 1
            else:
                df['binned'] = pd.cut(column, bins=min(3, column.nunique()), duplicates='drop', labels=False) + 1
        except ValueError:
            df['binned'] = 1
            
        CRITICAL - RFM Scoring Pattern:
        # Correct RFM scoring approach
        df['R_Score'] = pd.qcut(df['Recency'], q=5, duplicates='drop', labels=[5,4,3,2,1])
        df['F_Score'] = pd.qcut(df['Frequency'], q=5, duplicates='drop', labels=[1,2,3,4,5]) 
        df['M_Score'] = pd.qcut(df['Monetary'], q=5, duplicates='drop', labels=[1,2,3,4,5])
        # Convert to integers for calculation
        df['RFM_Score'] = df['R_Score'].astype(int) * 100 + df['F_Score'].astype(int) * 10 + df['M_Score'].astype(int)

        CRITICAL FORMATTING REQUIREMENTS:
        1. Generate ONLY valid Python code, no explanations or markdown
        2. Do NOT include markdown code blocks (```python or ```)  
        3. Start directly with import statements
        4. End with storing results in 'analysis_results' variable
        5. Ensure all syntax is valid Python

        EXAMPLE START:
        import pandas as pd
        import numpy as np
        
        # Your analysis code here
        analysis_results = {...}
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
