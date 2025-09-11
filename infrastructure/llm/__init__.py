"""LLM infrastructure bindings."""
from .base import LLMClient
from .gemini import GeminiClient

# Default LLM client binding used across the application
llm_client: LLMClient = GeminiClient()

__all__ = ["LLMClient", "llm_client", "GeminiClient"]
