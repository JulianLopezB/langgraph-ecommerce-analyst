"""Error handling node."""
from datetime import datetime

from logging_config import get_logger
from workflow.state import AnalysisState, ConversationMessage

logger = get_logger(__name__)


def handle_error(state: AnalysisState) -> AnalysisState:
    """Handle errors and provide recovery options."""
    logger.info("Handling workflow error")

    try:
        error_messages = []
        for error_type, error_msg in state["error_context"].items():
            error_messages.append(f"{error_type}: {error_msg}")

        error_summary = "; ".join(error_messages)

        suggestions = []
        if "sql" in error_summary.lower():
            suggestions.append("Try rephrasing your question with more specific details")
        if "validation" in error_summary.lower():
            suggestions.append("The analysis requires simpler operations")
        if "execution" in error_summary.lower():
            suggestions.append("The analysis may need a smaller dataset")

        error_response = f"I encountered an issue: {error_summary}"
        if suggestions:
            error_response += f"\n\nSuggestions: {'; '.join(suggestions)}"

        message = ConversationMessage(
            timestamp=datetime.now(),
            role="assistant",
            content=error_response,
            message_type="error",
        )
        state["conversation_history"].append(message)

        state["workflow_complete"] = True
        state["next_step"] = "complete"

    except Exception as e:
        logger.error(f"Error in error handler: {str(e)}")
        message = ConversationMessage(
            timestamp=datetime.now(),
            role="assistant",
            content=(
                "I encountered an unexpected error during analysis. "
                "Please try again with a simpler request."
            ),
            message_type="error",
        )
        state["conversation_history"].append(message)
        state["workflow_complete"] = True
        state["next_step"] = "complete"

    return state
