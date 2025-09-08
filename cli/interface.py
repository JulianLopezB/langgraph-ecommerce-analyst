"""Interactive CLI interface for the data analysis agent."""
import sys
from typing import Optional, Dict, Any
from datetime import datetime

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
from rich.text import Text
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.completion import WordCompleter

from agent.graph import session_manager
from config import config
from logging_config import get_logger

logger = get_logger(__name__)


class DataAnalysisCLI:
    """Interactive CLI for data analysis conversations."""
    
    def __init__(self):
        """Initialize the CLI interface."""
        self.console = Console()
        self.session_id: Optional[str] = None
        self.history = InMemoryHistory()
        
        # Create command completer
        self.completer = WordCompleter([
            'help', 'exit', 'quit', 'clear', 'history', 'new', 'session',
            'analyze', 'customers', 'products', 'sales', 'forecast',
            'segment', 'trends', 'revenue', 'churn', 'recommend'
        ])
        
        self.console.print(Panel.fit(
            "[bold blue]🤖 AI-Powered E-commerce Data Analysis Agent[/bold blue]\\n"
            "[dim]Ask questions about your e-commerce data in natural language[/dim]",
            title="Welcome",
            border_style="blue"
        ))
    
    def start_interactive_session(self):
        """Start an interactive analysis session."""
        try:
            # Start new session
            self.session_id = session_manager.start_session()
            self.console.print(f"[green]✓ Started new session: {self.session_id}[/green]")
            
            # Show help message
            self._show_help()
            
            # Main interaction loop
            while True:
                try:
                    # Get user input
                    user_input = prompt(
                        "🔍 Your question: ",
                        history=self.history,
                        completer=self.completer
                    ).strip()
                    
                    if not user_input:
                        continue
                    
                    # Handle commands
                    if user_input.lower() in ['exit', 'quit']:
                        break
                    elif user_input.lower() == 'help':
                        self._show_help()
                        continue
                    elif user_input.lower() == 'clear':
                        self.console.clear()
                        continue
                    elif user_input.lower() == 'history':
                        self._show_session_history()
                        continue
                    elif user_input.lower() == 'new':
                        if Confirm.ask("Start a new session?"):
                            self.session_id = session_manager.start_session()
                            self.console.print(f"[green]✓ Started new session: {self.session_id}[/green]")
                        continue
                    
                    # Process analysis query
                    self._process_query(user_input)
                    
                except KeyboardInterrupt:
                    if Confirm.ask("\\nDo you want to exit?"):
                        break
                    else:
                        continue
                except EOFError:
                    break
                except Exception as e:
                    self.console.print(f"[red]Error: {str(e)}[/red]")
                    logger.error(f"CLI error: {str(e)}")
            
            self.console.print("\\n[yellow]👋 Thank you for using the Data Analysis Agent![/yellow]")
            
        except Exception as e:
            self.console.print(f"[red]Fatal error: {str(e)}[/red]")
            logger.error(f"Fatal CLI error: {str(e)}")
            sys.exit(1)
    
    def _process_query(self, user_query: str):
        """Process a user analysis query."""
        try:
            # Show progress indicator
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
                transient=True
            ) as progress:
                task = progress.add_task("🧠 Analyzing your question...", total=None)
                
                # Perform analysis
                results = session_manager.analyze_query(user_query, self.session_id)
            
            # Display results
            self._display_results(results)
            
        except Exception as e:
            self.console.print(f"[red]Analysis failed: {str(e)}[/red]")
            logger.error(f"Query processing error: {str(e)}")
    
    def _display_results(self, results: Dict[str, Any]):
        """Display analysis results in a formatted way."""
        try:
            # Main insights
            if results.get("insights"):
                self.console.print()
                insights_panel = Panel(
                    Markdown(results["insights"]),
                    title="📊 Analysis Results",
                    border_style="green"
                )
                self.console.print(insights_panel)
            
            # Show execution details if available
            exec_results = results.get("execution_results", {})
            if exec_results and exec_results.get("status") == "success":
                self._show_execution_summary(exec_results)
            
            # Show any errors
            error_context = results.get("error_context", {})
            if error_context:
                self._show_error_summary(error_context)
            
            # Show data summary if available
            analysis_outputs = results.get("analysis_outputs", {})
            if "data_summary" in analysis_outputs:
                self._show_data_summary(analysis_outputs["data_summary"])
            
        except Exception as e:
            self.console.print(f"[red]Error displaying results: {str(e)}[/red]")
            logger.error(f"Display error: {str(e)}")
    
    def _show_execution_summary(self, exec_results: Dict[str, Any]):
        """Show execution performance summary."""
        execution_time = exec_results.get("execution_time", 0)
        memory_used = exec_results.get("memory_used_mb", 0)
        
        summary_text = f"⏱️  Execution time: {execution_time:.2f}s"
        if memory_used > 0:
            summary_text += f" | 💾 Memory used: {memory_used:.1f}MB"
        
        self.console.print(f"[dim]{summary_text}[/dim]")
    
    def _show_error_summary(self, error_context: Dict[str, Any]):
        """Show error information."""
        if error_context:
            error_panel = Panel(
                "\\n".join([f"• {error_type}: {error_msg}" for error_type, error_msg in error_context.items()]),
                title="⚠️  Issues Encountered",
                border_style="yellow"
            )
            self.console.print(error_panel)
    
    def _show_data_summary(self, data_summary: Dict[str, Any]):
        """Show data summary information."""
        rows = data_summary.get("rows", 0)
        columns = data_summary.get("columns", 0)
        
        self.console.print(f"[dim]📈 Analyzed {rows:,} rows across {columns} columns[/dim]")
    
    def _show_session_history(self):
        """Show conversation history for current session."""
        if not self.session_id:
            self.console.print("[yellow]No active session[/yellow]")
            return
        
        try:
            history = session_manager.get_session_history(self.session_id)
            
            if "error" in history:
                self.console.print(f"[red]{history['error']}[/red]")
                return
            
            conversation = history.get("conversation_history", [])
            
            if not conversation:
                self.console.print("[yellow]No conversation history[/yellow]")
                return
            
            # Create history table
            table = Table(title="Conversation History", show_header=True, header_style="bold magenta")
            table.add_column("Time", style="dim", width=20)
            table.add_column("Role", width=10)
            table.add_column("Message", width=60)
            
            for msg in conversation[-10:]:  # Show last 10 messages
                timestamp = datetime.fromisoformat(msg["timestamp"]).strftime("%H:%M:%S")
                role = "🧠 Agent" if msg["role"] == "assistant" else "👤 You"
                content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                
                table.add_row(timestamp, role, content)
            
            self.console.print(table)
            
        except Exception as e:
            self.console.print(f"[red]Error retrieving history: {str(e)}[/red]")
    
    def _show_help(self):
        """Show help information."""
        help_text = """
[bold]🚀 Getting Started[/bold]
Ask questions about your e-commerce data in natural language!

[bold]📝 Example Questions:[/bold]
• "Segment our customers using RFM analysis"
• "What are the sales trends for the last quarter?"
• "Which products have the highest revenue?"
• "Forecast sales for the next 3 months"
• "Show me customer churn analysis"
• "What products should we recommend to customer 12345?"

[bold]💡 Commands:[/bold]
• [cyan]help[/cyan] - Show this help message
• [cyan]history[/cyan] - Show conversation history
• [cyan]clear[/cyan] - Clear screen
• [cyan]new[/cyan] - Start new session
• [cyan]exit/quit[/cyan] - Exit the application

[bold]📊 Analysis Types Supported:[/bold]
• Customer segmentation and behavior analysis
• Sales forecasting and trend analysis
• Product performance and recommendations
• Statistical analysis and data exploration
• Machine learning models and clustering
        """.strip()
        
        help_panel = Panel(
            help_text,
            title="Help & Examples",
            border_style="cyan"
        )
        self.console.print(help_panel)


@click.command()
@click.option('--debug', is_flag=True, help='Enable debug logging')
@click.option('--session-id', help='Resume existing session')
def main(debug: bool, session_id: Optional[str]):
    """AI-Powered E-commerce Data Analysis Agent CLI."""
    
    # Logging is already configured by main.py
    logger.info(f"Starting CLI interface (debug={debug})")
    
    # Initialize and start CLI
    cli = DataAnalysisCLI()
    
    if session_id:
        cli.session_id = session_id
        cli.console.print(f"[green]Resuming session: {session_id}[/green]")
        logger.info(f"Resuming session: {session_id}")
    
    cli.start_interactive_session()


if __name__ == "__main__":
    main()
