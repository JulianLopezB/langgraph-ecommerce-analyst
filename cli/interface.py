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
        
        self.console.print("🤖 AI-Powered E-commerce Data Analysis Agent")
        self.console.print("Ask me questions about your e-commerce data in natural language!")
        self.console.print()
    
    def start_interactive_session(self):
        """Start an interactive analysis session."""
        try:
            # Start new session
            self.session_id = session_manager.start_session()
            self.console.print(f"✓ Ready to analyze your data!")
            
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
                            self.console.print("✓ Started a fresh session! What would you like to analyze?")
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
                    self.console.print(f"I encountered an issue: {str(e)}")
                    self.console.print("Let's try that again, or type 'help' for assistance.")
                    logger.error(f"CLI error: {str(e)}")
            
            self.console.print("\\n👋 Thanks for using the Data Analysis Agent! Have a great day!")
            
        except Exception as e:
            self.console.print(f"I'm sorry, something went wrong: {str(e)}")
            self.console.print("Please try restarting the application.")
            logger.error(f"Fatal CLI error: {str(e)}")
            sys.exit(1)
    
    def _process_query(self, user_query: str):
        """Process a user analysis query."""
        try:
            # Show detailed progress with agent thinking process
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
                transient=False  # Keep progress visible
            ) as progress:
                # Start with understanding the query
                task = progress.add_task("🧠 Understanding your question...", total=None)
                
                # Create a custom session manager that provides progress updates
                results = self._analyze_with_progress(user_query, progress, task)
            
            # Display results
            self._display_results(results)
            
        except Exception as e:
            self.console.print(f"I had trouble analyzing that question: {str(e)}")
            self.console.print("Could you try rephrasing your question or ask something else?")
            logger.error(f"Query processing error: {str(e)}")
    
    def _display_results(self, results: Dict[str, Any]):
        """Display analysis results in a conversational way."""
        try:
            # Main insights
            if results.get("insights"):
                self.console.print()
                self.console.print("📊 Here's what I found:")
                self.console.print()
                # Convert markdown to plain text
                insights_text = self._convert_markdown_to_text(results["insights"])
                self.console.print(insights_text)
                self.console.print()
            
            # Show any errors in a friendly way
            error_context = results.get("error_context", {})
            if error_context:
                self._show_error_summary(error_context)
            
        except Exception as e:
            self.console.print(f"I had trouble showing the results: {str(e)}")
            self.console.print("But I did complete the analysis!")
            logger.error(f"Display error: {str(e)}")
    
    
    def _show_error_summary(self, error_context: Dict[str, Any]):
        """Show error information in a friendly way."""
        if error_context:
            self.console.print("⚠️  I ran into a small issue:")
            for error_type, error_msg in error_context.items():
                friendly_msg = self._make_error_friendly(error_type, error_msg)
                self.console.print(f"   {friendly_msg}")
            self.console.print()
            self.console.print("Don't worry - I can try a different approach if you'd like!")
    
    
    def _show_session_history(self):
        """Show conversation history for current session."""
        if not self.session_id:
            self.console.print("No active session yet. Start by asking me a question!")
            return
        
        try:
            history = session_manager.get_session_history(self.session_id)
            
            if "error" in history:
                self.console.print(f"I couldn't find that session: {history['error']}")
                return
            
            conversation = history.get("conversation_history", [])
            
            if not conversation:
                self.console.print("We haven't chatted yet! Ask me a question to get started.")
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
            self.console.print(f"I couldn't retrieve the conversation history: {str(e)}")
    
    def _show_help(self):
        """Show help information."""
        self.console.print()
        self.console.print("🚀 Getting Started")
        self.console.print("Just ask me questions about your e-commerce data in plain English!")
        self.console.print()
        
        self.console.print("📝 Here are some things you can ask:")
        self.console.print("• Segment our customers using RFM analysis")
        self.console.print("• What are the sales trends for the last quarter?")
        self.console.print("• Which products have the highest revenue?")
        self.console.print("• Forecast sales for the next 3 months")
        self.console.print("• Show me customer churn analysis")
        self.console.print("• What products should we recommend to customer 12345?")
        self.console.print()
        
        self.console.print("💡 Commands you can use:")
        self.console.print("• help - Show this message")
        self.console.print("• history - Show our conversation")
        self.console.print("• clear - Clear the screen")
        self.console.print("• new - Start fresh")
        self.console.print("• exit or quit - Leave the chat")
        self.console.print()
        
        self.console.print("📊 I can help you with:")
        self.console.print("• Customer segmentation and behavior analysis")
        self.console.print("• Sales forecasting and trend analysis")
        self.console.print("• Product performance and recommendations")
        self.console.print("• Statistical analysis and data exploration")
        self.console.print("• Machine learning models and clustering")
        self.console.print()
    
    def _convert_markdown_to_text(self, markdown_text: str) -> str:
        """Convert markdown to plain text for conversational display."""
        import re
        
        # Remove markdown formatting
        text = markdown_text
        
        # Convert headers
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        
        # Convert bold/italic
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'\*(.*?)\*', r'\1', text)
        
        # Convert bullet points
        text = re.sub(r'^[-*+]\s+', '• ', text, flags=re.MULTILINE)
        
        # Clean up extra whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = text.strip()
        
        return text
    
    def _make_error_friendly(self, error_type: str, error_msg: str) -> str:
        """Convert technical error messages to friendly ones."""
        friendly_messages = {
            'sql_execution_error': "I had trouble running the database query",
            'code_generation_error': "I had difficulty creating the analysis code",
            'execution_error': "The analysis code ran into an issue",
            'validation_error': "I found a problem with the generated code",
            'understanding_error': "I had trouble understanding your question",
            'sql_generation_error': "I couldn't create the right database query"
        }
        
        friendly_msg = friendly_messages.get(error_type, "I encountered an unexpected issue")
        
        # Add specific details if they're helpful
        if "timeout" in error_msg.lower():
            friendly_msg += " (it took too long to complete)"
        elif "memory" in error_msg.lower():
            friendly_msg += " (it needed too much memory)"
        elif "syntax" in error_msg.lower():
            friendly_msg += " (there was a formatting issue)"
        
        return friendly_msg
    
    def _analyze_with_progress(self, user_query: str, progress, task):
        """Perform analysis with detailed progress updates showing agent actions."""
        import time
        
        try:
            # Create a progress tracker that follows the actual workflow steps
            progress_steps = [
                "🧠 Understanding your question...",
                "🎯 Determining analysis approach...", 
                "🔍 Analyzing data schema...",
                "⚡ Creating database query...",
                "📊 Executing query...",
                "🤖 Generating analysis code...",
                "🔬 Running calculations...",
                "✨ Preparing insights..."
            ]
            
            current_step = 0
            
            def update_progress():
                nonlocal current_step
                if current_step < len(progress_steps):
                    progress.update(task, description=progress_steps[current_step])
                    current_step += 1
            
            # Start with first step
            update_progress()
            
            # Monitor the analysis with periodic updates
            import threading
            
            def progress_monitor():
                while current_step < len(progress_steps):
                    time.sleep(2.5)  # Update every 2.5 seconds
                    update_progress()
            
            monitor_thread = threading.Thread(target=progress_monitor, daemon=True)
            monitor_thread.start()
            
            # Perform the actual analysis
            results = session_manager.analyze_query(user_query, self.session_id)
            
            # Complete
            progress.update(task, description="✅ Analysis complete!")
            time.sleep(0.3)
            
            return results
            
        except Exception as e:
            progress.update(task, description="❌ Something went wrong...")
            raise e

class AgentProgressTracker:
    """Helper class to track agent progress updates."""
    
    def __init__(self, progress, task):
        self.progress = progress
        self.task = task
        self.should_stop = False
    
    def update(self, description):
        if not self.should_stop:
            self.progress.update(self.task, description=description)
    
    def stop(self):
        self.should_stop = True


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
