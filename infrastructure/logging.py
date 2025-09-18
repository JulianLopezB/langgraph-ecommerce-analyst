"""Centralized logging configuration for the Data Analysis Agent."""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

from .config import LoggingConfig

# Global flag to ensure logging is only configured once
_logging_configured = False


class JsonFormatter(logging.Formatter):
    """Simple JSON formatter for structured logs."""

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        log_record = {
            "time": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_record["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(log_record)


def setup_logging(
    logging_config: Optional[LoggingConfig] = None,
    debug: bool = False,
    log_file: Optional[str] = None,
) -> None:
    """Setup centralized logging configuration."""
    global _logging_configured

    if _logging_configured:
        return

    logging_config = logging_config or LoggingConfig()

    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Use config settings with overrides
    log_file = log_file or logging_config.file_path or "logs/agent.log"
    log_level = (
        logging.DEBUG if debug else getattr(logging, logging_config.level.upper())
    )
    log_format = logging_config.format

    # Create handlers based on config
    handlers = []

    if logging_config.console_output:
        handlers.append(logging.StreamHandler(sys.stdout))

    if logging_config.file_output:
        handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))

    # Ensure we have at least one handler
    if not handlers:
        handlers.append(logging.StreamHandler(sys.stdout))

    formatter: logging.Formatter
    if logging_config.json_format:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(log_format)

    for handler in handlers:
        handler.setFormatter(formatter)

    # Configure logging with force=True to override any existing config
    logging.basicConfig(level=log_level, handlers=handlers, force=True)

    # Set specific loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    # Mark as configured
    _logging_configured = True

    # Log that logging is now configured
    logger = logging.getLogger(__name__)
    logger.info("=== Logging system initialized ===")
    logger.info(f"Log level: {logging.getLevelName(log_level)}")
    logger.info(f"Log file: {log_file}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a module, ensuring logging is configured.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured logger instance
    """
    # Ensure logging is configured with defaults if not already done
    if not _logging_configured:
        setup_logging()

    return logging.getLogger(name)
