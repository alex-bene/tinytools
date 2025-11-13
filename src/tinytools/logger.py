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


def setup_prettier_logger(
    logger: Logger | None = None,
    level: str | None = None,
    rich_handler_kwargs: dict | None = None,
    replace_handlers: bool = True,
) -> None:
    """Set up the logger.

    Args:
        logger (Logger | None, optional): The logger to configure. If None, the root logger will be used.
            Defaults to None.
        level (str | None, optional): The logging level. If None, the default level will be used. Defaults to None.
        rich_handler_kwargs (dict, optional): Additional keyword arguments for the RichHandler.
        replace_handlers (bool, optional): Whether to replace the default handlers or add to them. Defaults to True.

    """
    console = Console(theme=Theme({"log.time": "black", "log.path": Style(color="black", dim=True)}))
    rich_handler_kwargs = {
        "rich_tracebacks": True,
        "tracebacks_show_locals": False,
        "tracebacks_width": 100,
        "console": console,
        "log_time_format": "[%X]",
    } | (rich_handler_kwargs or {})
    handler = RichHandler(**rich_handler_kwargs)
    handler.get_level_text = get_level_text
    handler.setFormatter(logging.Formatter("[%(name)s:%(funcName)s] %(message)s", datefmt="%X"))

    # Get the logger
    logger = logger or logging.getLogger()
    if level is not None:
        logger.setLevel(level.upper())  # Set a base level for all logs

    # IMPORTANT: Remove any default handlers and add your new one
    if replace_handlers:
        logger.handlers = [handler]
    else:
        logger.addHandler(handler)


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
