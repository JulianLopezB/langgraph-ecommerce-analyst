"""OpenTelemetry tracing setup for the data analysis agent."""

from __future__ import annotations

import os
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - only for type hints
    from opentelemetry import trace as trace_mod


# Global tracer instance used across the application
otel_tracer: Optional["trace_mod.Tracer"] = None


def setup_tracing(service_name: str = "data-analysis-agent") -> Optional["trace_mod.Tracer"]:
    """Configure OpenTelemetry tracing with Jaeger and optional OTLP exporters.

    If the OpenTelemetry SDK is not installed, the function returns ``None`` and
    tracing is effectively disabled. This allows the application to run even in
    environments without the optional dependency.
    """

    global otel_tracer

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.jaeger.thrift import JaegerExporter
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
    except Exception:  # pragma: no cover - missing optional dependency
        otel_tracer = None
        return None

    resource = Resource(attributes={SERVICE_NAME: service_name})
    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)

    # Jaeger exporter configuration
    jaeger_host = os.getenv("OTEL_EXPORTER_JAEGER_AGENT_HOST", "localhost")
    jaeger_port = int(os.getenv("OTEL_EXPORTER_JAEGER_AGENT_PORT", "6831"))
    jaeger_exporter = JaegerExporter(agent_host_name=jaeger_host, agent_port=jaeger_port)
    provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))

    # Optional OTLP exporter (e.g. sending to OpenSearch or an OTEL collector)
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if otlp_endpoint:
        otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

    otel_tracer = trace.get_tracer(service_name)
    return otel_tracer


__all__ = ["setup_tracing", "otel_tracer"]

