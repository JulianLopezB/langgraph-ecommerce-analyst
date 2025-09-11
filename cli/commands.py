"""Command handling for the data analysis CLI."""
from typing import List
from rich.prompt import Confirm

from . import session, display

COMMANDS: List[str] = [
    'help', 'exit', 'quit', 'clear', 'history', 'new', 'session',
    'analyze', 'customers', 'products', 'sales', 'forecast',
    'segment', 'trends', 'revenue', 'churn', 'recommend'
]


def handle_command(cli, user_input: str) -> bool:
    """Handle a user command. Returns False to exit."""
    command = user_input.lower()

    if command in ('exit', 'quit'):
        return False
    if command == 'help':
        display.show_help(cli.console)
        return True
    if command == 'clear':
        cli.console.clear()
        return True
    if command == 'history':
        if not cli.session_id:
            cli.console.print("No active session yet. Start by asking me a question!")
            return True
        history = session.get_session_history(cli.session_id)
        if "error" in history:
            cli.console.print(f"I couldn't find that session: {history['error']}")
            return True
        conversation = history.get("conversation_history", [])
        display.show_session_history(cli.console, conversation)
        return True
    if command == 'new':
        if Confirm.ask("Start a new session?"):
            cli.session_id = session.start_session()
            cli.console.print("✓ Started a fresh session! What would you like to analyze?")
        return True

    process_query(cli, user_input)
    return True


def process_query(cli, user_query: str) -> None:
    """Process a user analysis query."""
    try:
        results = session.analyze_query_with_progress(cli.console, user_query, cli.session_id)
        display.display_results(cli.console, results)
    except Exception as e:
        cli.console.print(f"I had trouble analyzing that question: {str(e)}")
        cli.console.print("Could you try rephrasing your question or ask something else?")
