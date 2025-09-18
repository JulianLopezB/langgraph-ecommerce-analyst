"""Console output and formatting helpers for the data analysis CLI."""

from typing import Dict, Any, List
from datetime import datetime

from rich.table import Table
from utils.sql_utils import format_error_message


def display_results(console, results: Dict[str, Any]) -> None:
    """Display analysis results in a conversational format."""
    try:
        if results.get("insights"):
            console.print()
            console.print("📊 Here's what I found:")
            console.print()
            insights_text = convert_markdown_to_text(results["insights"])
            console.print(insights_text)
            console.print()

        error_context = results.get("error_context", {})
        if error_context:
            show_error_summary(console, error_context)
    except Exception as e:
        console.print(f"I had trouble showing the results: {str(e)}")
        console.print("But I did complete the analysis!")


def show_error_summary(console, error_context: Dict[str, Any]) -> None:
    """Show error information in a friendly way."""
    if error_context:
        console.print("⚠️  I ran into a small issue:")
        for error_type, error_msg in error_context.items():
            friendly_msg = format_error_message(error_type, error_msg)
            console.print(f"   {friendly_msg}")
        console.print()
        console.print("Don't worry - I can try a different approach if you'd like!")


def show_session_history(console, conversation: List[Dict[str, Any]]) -> None:
    """Display conversation history."""
    if not conversation:
        console.print("We haven't chatted yet! Ask me a question to get started.")
        return

    table = Table(
        title="Conversation History", show_header=True, header_style="bold magenta"
    )
    table.add_column("Time", style="dim", width=20)
    table.add_column("Role", width=10)
    table.add_column("Message", width=60)

    for msg in conversation[-10:]:
        timestamp = datetime.fromisoformat(msg["timestamp"]).strftime("%H:%M:%S")
        role = "🧠 Agent" if msg["role"] == "assistant" else "👤 You"
        content = (
            msg["content"][:100] + "..."
            if len(msg["content"]) > 100
            else msg["content"]
        )
        table.add_row(timestamp, role, content)

    console.print(table)


def show_help(console) -> None:
    """Show help information."""
    console.print()
    console.print("🚀 Getting Started")
    console.print("Just ask me questions about your e-commerce data in plain English!")
    console.print()

    console.print("📝 Here are some things you can ask:")
    console.print("• Segment our customers using RFM analysis")
    console.print("• What are the sales trends for the last quarter?")
    console.print("• Which products have the highest revenue?")
    console.print("• Forecast sales for the next 3 months")
    console.print("• Show me customer churn analysis")
    console.print("• What products should we recommend to customer 12345?")
    console.print()

    console.print("💡 Commands you can use:")
    console.print("• help - Show this message")
    console.print("• history - Show our conversation")
    console.print("• clear - Clear the screen")
    console.print("• new - Start fresh")
    console.print("• exit or quit - Leave the chat")
    console.print()

    console.print("📊 I can help you with:")
    console.print("• Customer segmentation and behavior analysis")
    console.print("• Sales forecasting and trend analysis")
    console.print("• Product performance and recommendations")
    console.print("• Statistical analysis and data exploration")
    console.print("• Machine learning models and clustering")
    console.print()


def convert_markdown_to_text(markdown_text: str) -> str:
    """Convert markdown to plain text for conversational display."""
    import re

    text = markdown_text
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"^[-*+]\s+", "• ", text, flags=re.MULTILINE)
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text.strip()
