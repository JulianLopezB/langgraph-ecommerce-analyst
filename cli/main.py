"""Entry point for the data analysis CLI."""
from typing import Optional

import click

from logging_config import get_logger

from .interface import DataAnalysisCLI

logger = get_logger(__name__)


@click.command()
@click.option('--debug', is_flag=True, help='Enable debug logging')
@click.option('--session-id', help='Resume existing session')
def main(debug: bool, session_id: Optional[str]) -> None:
    """AI-Powered E-commerce Data Analysis Agent CLI."""
    logger.info(f"Starting CLI interface (debug={debug})")
    cli = DataAnalysisCLI()

    if session_id:
        cli.session_id = session_id
        cli.console.print(f"[green]Resuming session: {session_id}[/green]")
        logger.info(f"Resuming session: {session_id}")

    cli.start_interactive_session()


if __name__ == "__main__":
    main()
