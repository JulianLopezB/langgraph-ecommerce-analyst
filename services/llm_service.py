"""LLM service for Google Gemini integration."""
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from config import config
from logging_config import get_logger

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
                
                return LLMResponse(
                    content=content,
                    tokens_used=self._estimate_tokens(prompt + content),
                    response_time=response_time,
                    model_used=os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')
                )
            else:
                logger.warning("Empty response from Gemini API")
                return LLMResponse(content="", response_time=time.time() - start_time)
                
        except Exception as e:
            logger.error(f"Error generating text with Gemini: {str(e)}")
            raise
    
    def classify_intent(self, user_query: str) -> Dict[str, Any]:
        """Classify user intent for analysis routing."""
        prompt = f"""
        Analyze the following user query and classify the intent for data analysis:

        Query: "{user_query}"

        Classify into one of these categories:
        - data_exploration: Basic data inspection, summaries, data quality checks
        - customer_analysis: Customer segmentation, behavior analysis, churn, CLV
        - product_analysis: Product performance, recommendations, inventory
        - sales_analysis: Sales trends, forecasting, revenue analysis
        - advanced_analytics: Machine learning, statistical modeling, complex algorithms
        - visualization: Creating charts, graphs, dashboards

        Also determine:
        1. Confidence score (0.0-1.0)
        2. Whether SQL alone is sufficient or if Python analysis is needed
        3. Key entities mentioned (customers, products, dates, metrics)

        Respond in this exact JSON format:
        {{
            "intent": "category_name",
            "confidence": 0.85,
            "needs_python": true/false,
            "entities": ["entity1", "entity2"],
            "analysis_type": "brief description",
            "complexity": "low/medium/high"
        }}
        """
        
        response = self.generate_text(prompt, temperature=0.3)
        
        try:
            import json
            import re
            
            # Extract JSON from response (handle markdown formatting)
            content = response.content.strip()
            
            # Try to extract JSON block if wrapped in markdown
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                content = json_match.group(1)
            elif content.startswith('```') and content.endswith('```'):
                content = content.strip('`').strip()
            
            # Try to find JSON object in the content
            if not content.startswith('{'):
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    content = json_match.group(0)
            
            result = json.loads(content)
            return result
        except (json.JSONDecodeError, AttributeError, TypeError) as e:
            logger.warning(f"Failed to parse intent classification JSON: {e}")
            logger.debug(f"Raw response content: {response.content[:200]}...")
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "needs_python": True,
                "entities": [],
                "analysis_type": "unknown",
                "complexity": "high"
            }
    
    def generate_sql_query(self, intent_data: Dict[str, Any], schema_info: Dict[str, Any]) -> str:
        """Generate SQL query based on user intent and schema."""
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
        return response.content.strip()
    
    def generate_python_code(self, intent_data: Dict[str, Any], data_info: Dict[str, Any]) -> str:
        """Generate Python code for advanced analysis."""
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
        return response.content.strip()
    
    def generate_insights(self, analysis_results: Dict[str, Any], original_query: str) -> str:
        """Generate business insights from analysis results."""
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
        return response.content.strip()
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        return len(text.split()) * 1.3  # Rough estimate
