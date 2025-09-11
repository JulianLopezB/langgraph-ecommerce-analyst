"""Session management helpers for the data analysis CLI."""
from typing import Dict, Any
import time
import threading

from rich.progress import Progress, SpinnerColumn, TextColumn

from application.controllers import AnalysisController


# Single controller instance used by the CLI
_controller = AnalysisController()


def start_session() -> str:
    """Start a new analysis session."""
    return _controller.start_session()


def analyze_query_with_progress(console, user_query: str, session_id: str) -> Dict[str, Any]:
    """Perform analysis with detailed progress updates."""
    progress_steps = [
        "🧠 Understanding your question...",
        "🎯 Determining analysis approach...",
        "🔍 Analyzing data schema...",
        "⚡ Creating database query...",
        "📊 Executing query...",
        "🤖 Generating analysis code...",
        "🔬 Running calculations...",
        "✨ Preparing insights...",
    ]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task(progress_steps[0], total=None)
        current_step = 0

        def update_progress():
            nonlocal current_step
            if current_step < len(progress_steps):
                progress.update(task, description=progress_steps[current_step])
                current_step += 1

        update_progress()

        def progress_monitor():
            while current_step < len(progress_steps):
                time.sleep(2.5)
                update_progress()

        monitor_thread = threading.Thread(target=progress_monitor, daemon=True)
        monitor_thread.start()

        results = _controller.analyze_query(user_query, session_id)
        progress.update(task, description="✅ Analysis complete!")
        time.sleep(0.3)
        return results


def get_session_history(session_id: str) -> Dict[str, Any]:
    """Retrieve session conversation history."""
    return _controller.get_session_history(session_id)
