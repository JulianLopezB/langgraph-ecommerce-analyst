"""Configuration management for the LangGraph Data Analysis Agent."""
import os
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class ExecutionLimits(BaseModel):
    """Configuration for code execution limits."""
    max_execution_time: int = Field(default=300, description="Maximum execution time in seconds")
    max_memory_mb: int = Field(default=1024, description="Maximum memory usage in MB")
    max_output_size_mb: int = Field(default=100, description="Maximum output size in MB")
    concurrent_executions: int = Field(default=1, description="Maximum concurrent executions")


class SecurityConfig(BaseModel):
    """Security configuration settings."""
    enable_code_scanning: bool = Field(default=True, description="Enable security code scanning")
    allowed_imports: list[str] = Field(
        default=[
            "pandas", "numpy", "matplotlib", "seaborn", "plotly", "scipy", "sklearn",
            "statsmodels", "prophet", "xgboost", "datetime", "math", "statistics",
            "json", "re", "warnings", "typing"
        ],
        description="List of allowed import modules"
    )
    forbidden_patterns: list[str] = Field(
        default=[
            "__import__", "eval", "exec", "compile", "globals", "locals",
            "open", "file", "input", "raw_input", "subprocess", "os.system",
            "os.popen", "os.spawn", "socket", "urllib", "requests"
        ],
        description="List of forbidden code patterns"
    )


class APIConfig(BaseModel):
    """API configuration settings."""
    gemini_api_key: Optional[str] = Field(default=None, description="Google Gemini API key")
    bigquery_project_id: Optional[str] = Field(default=None, description="BigQuery project ID")
    dataset_id: str = Field(default="bigquery-public-data.thelook_ecommerce", description="BigQuery dataset ID")
    max_query_results: int = Field(default=10000, description="Maximum query result rows")
    query_timeout: int = Field(default=300, description="Query timeout in seconds")


class LoggingConfig(BaseModel):
    """Logging configuration settings."""
    level: str = Field(default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR)")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )
    file_path: str = Field(default="logs/agent.log", description="Main log file path")
    console_output: bool = Field(default=True, description="Enable console output")
    file_output: bool = Field(default=True, description="Enable file output")


class SystemConfig(BaseModel):
    """Main system configuration."""
    execution_limits: ExecutionLimits = Field(default_factory=ExecutionLimits)
    security_settings: SecurityConfig = Field(default_factory=SecurityConfig)
    api_configurations: APIConfig = Field(default_factory=APIConfig)
    logging_settings: LoggingConfig = Field(default_factory=LoggingConfig)
    
    def __init__(self, **kwargs):
        # Load API configurations from environment variables
        api_config = kwargs.get('api_configurations', {})
        if not isinstance(api_config, dict):
            api_config = api_config.dict() if hasattr(api_config, 'dict') else {}
        
        api_config.setdefault('gemini_api_key', os.getenv('GEMINI_API_KEY'))
        api_config.setdefault('bigquery_project_id', os.getenv('GOOGLE_CLOUD_PROJECT'))
        
        kwargs['api_configurations'] = APIConfig(**api_config)
        super().__init__(**kwargs)


# Global configuration instance
config = SystemConfig()
