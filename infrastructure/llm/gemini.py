"""Google Gemini implementation of :class:`LLMClient`."""

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

import google.generativeai as genai
from google.generativeai.types import HarmBlockThreshold, HarmCategory

from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from infrastructure.code_cleaning import create_ast_cleaner
from infrastructure.logging import get_logger
from tracing.langsmith_setup import trace_llm_operation, tracer

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
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
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

        self.model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

        # Initialize AST-based code cleaner
        self.code_cleaner = create_ast_cleaner()

        logger.info("Gemini service initialized")

    def _create_langchain_model(
        self, temperature: float, max_tokens: int
    ) -> ChatGoogleGenerativeAI:
        """Create a LangChain Gemini model configured for this client."""

        return ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key,
            temperature=temperature,
            max_output_tokens=max_tokens,
            safety_settings=self.safety_settings,
        )

    def generate_text(
        self, prompt: str, temperature: float = 0.3, max_tokens: int = 2048
    ) -> LLMResponse:
        """Generate text response from Gemini."""
        start_time = time.time()

        with trace_llm_operation(
            name="gemini_generate_text",
            model="gemini-1.5-flash",
            temperature=temperature,
            max_tokens=max_tokens,
            prompt_length=len(prompt),
        ):
            try:
                langchain_model = self._create_langchain_model(
                    temperature, max_tokens
                )
                chain = langchain_model | StrOutputParser()
                content = chain.invoke(prompt)
                if not content:
                    logger.warning("Empty response from Gemini API")
                    return LLMResponse(
                        content="", response_time=time.time() - start_time
                    )

                response_time = time.time() - start_time
                tokens_used = self._estimate_tokens(prompt + content)

                # Log metrics to trace
                tracer.log_metrics(
                    {
                        "response_time": response_time,
                        "tokens_used": tokens_used,
                        "content_length": len(content),
                        "prompt_length": len(prompt),
                        "langchain_parser": "StrOutputParser",
                    }
                )

                return LLMResponse(
                    content=content,
                    tokens_used=tokens_used,
                    response_time=response_time,
                    model_used=self.model_name,
                )

            except Exception as e:
                logger.error(f"Error generating text with Gemini: {str(e)}")
                raise

    def generate_adaptive_python_code(self, analysis_context: Dict[str, Any]) -> str:
        """Generate adaptive Python code based on actual data characteristics."""
        with trace_llm_operation(
            name="gemini_generate_adaptive_python",
            model="gemini-1.5-flash",
            original_query=analysis_context.get("original_query", "unknown")[:50],
            data_shape=str(
                analysis_context.get("data_characteristics", {}).get("shape", "unknown")
            ),
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
            tracer.log_metrics(
                {
                    "python_generation_prompt_length": len(prompt),
                    "generated_code_length": len(python_code),
                    "process_type": process_data.get("process_type", "unknown"),
                    "data_shape": str(data_characteristics.get("shape", "unknown")),
                }
            )

            return python_code

    def generate_insights(
        self, analysis_results: Dict[str, Any], original_query: str
    ) -> str:
        """Generate business insights from analysis results."""
        with trace_llm_operation(
            name="gemini_generate_insights",
            model="gemini-1.5-flash",
            query_length=len(original_query),
            results_keys=len(analysis_results.keys()),
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
            tracer.log_metrics(
                {
                    "insights_generation_prompt_length": len(prompt),
                    "generated_insights_length": len(insights),
                    "original_query_length": len(original_query),
                    "analysis_data_points": (
                        len(analysis_results.get("processed_data", []))
                        if "processed_data" in analysis_results
                        else 0
                    ),
                }
            )

            return insights

    def _clean_python_code(self, code: str) -> str:
        """Clean Python code using AST-based processing to avoid syntax errors."""
        try:
            cleaned_code, metadata = self.code_cleaner.clean_code(code)

            if metadata["success"]:
                logger.debug(
                    f"AST-based cleaning successful: {metadata['original_lines']} -> {metadata['cleaned_lines']} lines"
                )
                if metadata["imports_removed"]:
                    logger.debug(f"Removed imports: {metadata['imports_removed']}")
                return cleaned_code
            else:
                logger.warning(f"AST-based cleaning failed: {metadata['errors']}")
                # Fallback to basic markdown removal only
                return self._basic_markdown_cleanup(code)

        except Exception as e:
            logger.error(f"Code cleaning failed: {e}", exc_info=True)
            # Fallback to basic cleanup
            return self._basic_markdown_cleanup(code)

    def _basic_markdown_cleanup(self, code: str) -> str:
        """Basic fallback cleanup that only removes markdown formatting."""
        # Remove markdown code blocks
        if code.startswith("```python"):
            code = code[9:]
        elif code.startswith("```"):
            code = code[3:]

        if code.endswith("```"):
            code = code[:-3]

        return code.strip()

    def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> T:
        """Generate structured output using LangChain's Google Gemini integration."""
        start_time = time.time()

        with trace_llm_operation(
            name="gemini_generate_structured",
            model="gemini-1.5-flash",
            temperature=temperature,
            max_tokens=max_tokens,
            prompt_length=len(prompt),
            schema_name=schema.__name__,
        ):
            try:
                langchain_model = self._create_langchain_model(
                    temperature, max_tokens
                )

                # LangChain handles Pydantic schemas natively through function calling
                structured_model = langchain_model.with_structured_output(schema)
                response = structured_model.invoke(prompt)

                response_time = time.time() - start_time
                tokens_used = self._estimate_tokens(prompt + str(response))

                # Log success metrics
                tracer.log_metrics(
                    {
                        "structured_generation_success": True,
                        "response_time": response_time,
                        "tokens_used": tokens_used,
                        "schema_validation_success": True,
                        "langchain_structured_output": True,
                    }
                )

                logger.debug(
                    f"LangChain structured output successful for {schema.__name__}"
                )
                return response

            except Exception as e:
                logger.error(f"LangChain structured output failed: {e}")
                tracer.log_metrics(
                    {
                        "structured_generation_success": False,
                        "error": str(e),
                        "schema_validation_success": False,
                    }
                )
                return self._create_fallback_response(schema)

    def _create_fallback_response(self, schema: Type[T]) -> T:
        """Create a minimal fallback response when structured generation fails."""
        try:
            # Create a minimal valid instance with default values
            return schema()
        except Exception as e:
            logger.error(f"Error creating fallback response for {schema.__name__}: {e}")
            raise RuntimeError(
                f"Unable to create fallback response for {schema.__name__}: {e}"
            )

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        return len(text.split()) * 1.3  # Rough estimate
