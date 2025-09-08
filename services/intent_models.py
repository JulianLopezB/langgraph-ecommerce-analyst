"""Pydantic models for LLM function calling."""
from typing import List, Optional
from pydantic import BaseModel, Field
from enum import Enum

# Import IntentType enum
from agent.state import IntentType


class IntentClassificationResult(BaseModel):
    """Structured result from intent classification function call."""
    intent: IntentType = Field(description="The classified intent type")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0", ge=0.0, le=1.0)
    needs_python: bool = Field(description="Whether Python analysis is needed beyond SQL")
    entities: List[str] = Field(description="Key entities mentioned in the query", default_factory=list)
    analysis_type: str = Field(description="Brief description of the analysis type")
    complexity: str = Field(description="Complexity level: low, medium, or high")


# Define the function schema for Gemini function calling
CLASSIFY_INTENT_FUNCTION = {
    "name": "classify_user_intent",
    "description": "Classify user query intent for data analysis routing",
    "parameters": {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "enum": [intent.value for intent in IntentType],
                "description": "The classified intent type"
            },
            "confidence": {
                "type": "number",
                "description": "Confidence score between 0.0 and 1.0"
            },
            "needs_python": {
                "type": "boolean",
                "description": "Whether Python analysis is needed beyond SQL"
            },
            "entities": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Key entities mentioned in the query (customers, products, dates, metrics)"
            },
            "analysis_type": {
                "type": "string",
                "description": "Brief description of the analysis type"
            },
            "complexity": {
                "type": "string",
                "enum": ["low", "medium", "high"],
                "description": "Complexity level of the analysis"
            }
        },
        "required": ["intent", "confidence", "needs_python", "entities", "analysis_type", "complexity"]
    }
}
