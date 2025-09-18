"""LLM infrastructure bindings."""

from .base import LLMClient
from .gemini import GeminiClient

# Default LLM client binding used across the application; injected at runtime
llm_client: LLMClient | None = None

__all__ = ["LLMClient", "llm_client", "GeminiClient"]
