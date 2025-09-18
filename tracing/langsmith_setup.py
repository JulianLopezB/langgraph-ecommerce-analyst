"""LangSmith tracing setup and custom instrumentation."""

import os
from contextlib import contextmanager, nullcontext
from functools import wraps
from typing import Any, Callable, Dict, Optional

from langsmith import Client, traceable
from langsmith.run_helpers import tracing_context

from infrastructure.logging import get_logger

from .opentelemetry_setup import otel_tracer

logger = get_logger(__name__)


class LangSmithTracer:
    """Custom LangSmith tracer for the data analysis agent."""

    def __init__(self):
        """Initialize LangSmith tracing."""
        self.client: Optional[Client] = None
        self.enabled = False
        self.setup_tracing()

    def setup_tracing(self) -> None:
        """Setup LangSmith tracing configuration."""
        try:
            api_key = os.getenv("LANGCHAIN_API_KEY")
            project = os.getenv("LANGCHAIN_PROJECT", "data-analysis-agent")
            endpoint = os.getenv(
                "LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"
            )
            enable = os.getenv("LANGCHAIN_TRACING_V2", "true").lower() == "true"

            if enable and api_key:
                os.environ["LANGCHAIN_TRACING_V2"] = "true"
                os.environ["LANGCHAIN_API_KEY"] = api_key
                os.environ["LANGCHAIN_PROJECT"] = project
                os.environ["LANGCHAIN_ENDPOINT"] = endpoint

                self.client = Client(api_url=endpoint, api_key=api_key)
                self.enabled = True
                logger.info(f"LangSmith tracing enabled for project: {project}")
            else:
                logger.info(
                    "LangSmith tracing disabled (missing API key or disabled in env)"
                )

        except Exception as e:
            logger.warning(f"Failed to setup LangSmith tracing: {e}")
            self.enabled = False

    @contextmanager
    def trace_operation(self, name: str, operation_type: str = "custom", **metadata):
        """Context manager for tracing custom operations."""

        span_ctx = (
            otel_tracer.start_as_current_span(name) if otel_tracer else nullcontext()
        )

        with span_ctx as span:
            if span is not None:
                span.set_attribute("operation_type", operation_type)
                for key, value in metadata.items():
                    span.set_attribute(key, value)

            if not self.enabled:
                yield
                return

            try:
                with tracing_context(
                    name=name,
                    metadata={"operation_type": operation_type, **metadata},
                ):
                    yield
            except Exception as e:
                logger.error(f"Error in traced operation {name}: {e}")
                yield

    def trace_function(
        self, name: Optional[str] = None, operation_type: str = "function"
    ):
        """Decorator for tracing functions."""

        def decorator(func: Callable) -> Callable:
            traced_func = (
                traceable(name=name or func.__name__)(func) if self.enabled else func
            )

            @wraps(func)
            def wrapper(*args, **kwargs):
                span_ctx = (
                    otel_tracer.start_as_current_span(name or func.__name__)
                    if otel_tracer
                    else nullcontext()
                )
                with span_ctx:
                    try:
                        return traced_func(*args, **kwargs)
                    except Exception as e:
                        logger.error(f"Error in traced function {func.__name__}: {e}")
                        raise

            return wrapper

        return decorator

    def trace_llm_call(self, name: str, model: str, **metadata):
        """
        Context manager for tracing LLM calls with specific metadata.

        Args:
            name: Name of the LLM operation
            model: Model being used
            **metadata: Additional metadata
        """
        return self.trace_operation(
            name=name, operation_type="llm", model=model, **metadata
        )

    def trace_bigquery_operation(self, name: str, query: str, **metadata):
        """
        Context manager for tracing BigQuery operations.

        Args:
            name: Name of the BigQuery operation
            query: SQL query being executed
            **metadata: Additional metadata
        """
        return self.trace_operation(
            name=name,
            operation_type="bigquery",
            sql_query=query[:500] + "..." if len(query) > 500 else query,
            **metadata,
        )

    def trace_code_execution(self, name: str, code: str, **metadata):
        """
        Context manager for tracing code execution.

        Args:
            name: Name of the code execution
            code: Code being executed
            **metadata: Additional metadata
        """
        return self.trace_operation(
            name=name,
            operation_type="code_execution",
            code_snippet=code[:200] + "..." if len(code) > 200 else code,
            **metadata,
        )

    def log_metrics(self, metrics: Dict[str, Any], run_id: Optional[str] = None):
        """
        Log metrics to the current trace.

        Args:
            metrics: Dictionary of metrics to log
            run_id: Optional run ID to attach metrics to
        """
        if not self.enabled:
            return

        try:
            # Just log metrics as debug for now to avoid context issues
            # The tracing context API seems to be inconsistent
            for key, value in metrics.items():
                logger.debug(f"LangSmith metric - {key}: {value}")
        except Exception as e:
            logger.debug(f"Error logging metrics: {e}")


# Global tracer instance
tracer = LangSmithTracer()


# Convenience decorators and context managers
def trace_agent_operation(name: str, **metadata):
    """Shorthand for tracing agent operations."""
    return tracer.trace_operation(name=name, operation_type="agent", **metadata)


def trace_llm_operation(name: str, model: str = "gemini", **metadata):
    """Shorthand for tracing LLM operations."""
    return tracer.trace_llm_call(name=name, model=model, **metadata)


def traced_function(name: Optional[str] = None):
    """Shorthand decorator for tracing functions."""
    return tracer.trace_function(name=name)
