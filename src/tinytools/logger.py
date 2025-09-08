"""Logger tools."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from rich.console import Console
from rich.logging import RichHandler
from rich.style import Style
from rich.text import Text
from rich.theme import Theme

if TYPE_CHECKING:
    from logging import Logger, LogRecord


def get_level_text(record: LogRecord) -> Text:
    """Get the formatted level text for a log record.

    Args:
        self (RichHandler): The RichHandler instance.
        record (LogRecord): The log record.

    Returns:
        Text: The formatted level text.

    """
    level_name = record.levelname.lower()
    return Text.styled("[" + level_name.ljust(8) + "]", f"logging.level.{level_name}")


def setup_prettier_root_logger(level: str = "NOTSET") -> Logger:
    """Set up the root logger.

    Args:
        level (str, optional): The logging level. Defaults to "NOTSET".

    Returns:
        Logger: The root logger.

    """
    console = Console(theme=Theme({"log.time": "black", "log.path": Style(color="black", dim=True)}))
    handler = RichHandler(
        rich_tracebacks=True, tracebacks_show_locals=True, tracebacks_width=100, console=console, log_time_format="[%X]"
    )
    handler.get_level_text = get_level_text
    handler.setFormatter(logging.Formatter("[%(name)s:%(funcName)s] %(message)s", datefmt="%X"))

    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level.upper())  # Set a base level for all logs

    # IMPORTANT: Remove any default handlers and add your new one
    root_logger.handlers = [handler]


def get_logger(name: str, level: str = "NOTSET") -> Logger:
    """Get a logger with rich formatting.

    This function creates and configures a logger with rich formatting for console output.

    Args:
        name (str): The name of the logger.
        level (str, optional): The logging level. Defaults to "NOTSET".

    Returns:
        Logger: A configured logger instance.

    """
    logger = logging.getLogger(name)
    logger.setLevel(level.upper())

    return logger
