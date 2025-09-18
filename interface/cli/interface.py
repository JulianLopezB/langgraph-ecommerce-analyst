"""Interactive CLI interface for the data analysis agent."""

import sys
from typing import Optional

from application.controllers import AnalysisController

from rich.console import Console
from rich.prompt import Confirm
from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.completion import WordCompleter

from infrastructure.logging import get_logger

from . import commands, session
from .display import show_help

logger = get_logger(__name__)


class DataAnalysisCLI:
    """Interactive CLI for data analysis conversations."""

    def __init__(self, controller: AnalysisController) -> None:
        self.console = Console()
        self.session_id: Optional[str] = None
        self.history = InMemoryHistory()
        self.completer = WordCompleter(commands.COMMANDS)
        self.controller = controller

        self.console.print("🤖 AI-Powered E-commerce Data Analysis Agent")
        self.console.print(
            "Ask me questions about your e-commerce data in natural language!"
        )
        self.console.print()

    def start_interactive_session(self) -> None:
        """Start an interactive analysis session."""
        try:
            self.session_id = session.start_session(self.controller)
            self.console.print("✓ Ready to analyze your data!")
            show_help(self.console)

            while True:
                try:
                    user_input = prompt(
                        "🔍 Your question: ",
                        history=self.history,
                        completer=self.completer,
                    ).strip()
                    if not user_input:
                        continue
                    if not commands.handle_command(self, user_input):
                        break
                except KeyboardInterrupt:
                    if Confirm.ask("\nDo you want to exit?"):
                        break
                except EOFError:
                    break
                except Exception as e:
                    self.console.print(f"I encountered an issue: {str(e)}")
                    self.console.print(
                        "Let's try that again, or type 'help' for assistance."
                    )
                    logger.error(f"CLI error: {str(e)}")

            self.console.print(
                "\n👋 Thanks for using the Data Analysis Agent! Have a great day!"
            )
        except Exception as e:
            self.console.print(f"I'm sorry, something went wrong: {str(e)}")
            self.console.print("Please try restarting the application.")
            logger.error(f"Fatal CLI error: {str(e)}")
            sys.exit(1)
