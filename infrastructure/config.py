"""Configuration management for the LangGraph Data Analysis Agent."""
import os
from typing import Optional
from pydantic import BaseModel, Field
from .secret_manager import get_env_or_secret

# Centralized constants
DEFAULT_DATASET_ID = "bigquery-public-data.thelook_ecommerce"


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


class AppConfig(BaseModel):
    """Application configuration settings."""
    gemini_api_key: Optional[str] = Field(default=None, description="Google Gemini API key")
    bigquery_project_id: Optional[str] = Field(default=None, description="BigQuery project ID")
    dataset_id: str = Field(default=DEFAULT_DATASET_ID, description="BigQuery dataset ID")
    max_query_results: int = Field(default=10000, description="Maximum query result rows")
    query_timeout: int = Field(default=300, description="Query timeout in seconds")
    
    # LangSmith configuration
    langsmith_api_key: Optional[str] = Field(default=None, description="LangSmith API key")
    langsmith_project: str = Field(default="data-analysis-agent", description="LangSmith project name")
    langsmith_endpoint: str = Field(default="https://api.smith.langchain.com", description="LangSmith API endpoint")
    enable_tracing: bool = Field(default=True, description="Enable LangSmith tracing")


class LoggingConfig(BaseModel):
    """Logging configuration settings."""
    level: str = Field(default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR)")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )
    file_path: str = Field(default="logs/agent.log", description="Main log file path")
    console_output: bool = Field(default=False, description="Enable console output")
    file_output: bool = Field(default=True, description="Enable file output")


class SystemConfig(BaseModel):
    """Main system configuration."""
    execution_limits: ExecutionLimits = Field(default_factory=ExecutionLimits)
    security_settings: SecurityConfig = Field(default_factory=SecurityConfig)
    api_configurations: AppConfig = Field(default_factory=AppConfig)
    logging_settings: LoggingConfig = Field(default_factory=LoggingConfig)
    environment: str = Field(default=os.getenv("APP_ENV", "development"), description="Deployment environment")
    
    def __init__(self, **kwargs):
        # Load API configurations from environment variables
        api_config = kwargs.get('api_configurations', {})
        if not isinstance(api_config, dict):
            api_config = api_config.dict() if hasattr(api_config, 'dict') else {}
        
        api_config.setdefault('gemini_api_key', get_env_or_secret('GEMINI_API_KEY', 'GEMINI_SECRET_NAME'))
        api_config.setdefault('bigquery_project_id', os.getenv('GOOGLE_CLOUD_PROJECT'))
        api_config.setdefault('dataset_id', os.getenv('BQ_DATASET_ID', DEFAULT_DATASET_ID))
        api_config.setdefault('langsmith_api_key', get_env_or_secret('LANGCHAIN_API_KEY', 'LANGCHAIN_SECRET_NAME'))
        api_config.setdefault('langsmith_project', os.getenv('LANGCHAIN_PROJECT', 'data-analysis-agent'))
        api_config.setdefault('enable_tracing', os.getenv('LANGCHAIN_TRACING_V2', 'true').lower() == 'true')
        
        kwargs['api_configurations'] = AppConfig(**api_config)
        super().__init__(**kwargs)


class DevelopmentConfig(SystemConfig):
    """Configuration for development environment."""


class ProductionConfig(SystemConfig):
    """Configuration for production environment."""


_CONFIG_MAP = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
}


def get_config(environment: Optional[str] = None) -> SystemConfig:
    """Load configuration based on the deployment environment."""
    env = (environment or os.getenv("APP_ENV", "development")).lower()
    config_cls = _CONFIG_MAP.get(env, DevelopmentConfig)
    return config_cls()
