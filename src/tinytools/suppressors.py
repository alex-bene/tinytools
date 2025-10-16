"""Context managers to suppress logging, stdout, stderr and tqdm progress bars."""

import logging
import os
import sys
from collections.abc import Generator
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from io import StringIO

from tqdm import tqdm


@contextmanager
def suppress_logging() -> Generator[None, None, None]:
    """Context manager to suppress logging."""
    logger = logging.getLogger()
    previous_level = logger.level
    logging.disable(logging.CRITICAL)  # suppress everything
    try:
        yield
    finally:
        logging.disable(previous_level)  # restore


@contextmanager
def suppress_tqdm() -> Generator[None, None, None]:
    """Context manager that suppresses all tqdm progress bars.

    Example:
        with suppress_tqdm():
            for i in tqdm(range(100)):
                ...

    """
    original = tqdm.__init__

    def dummy_init(self: tqdm, *args, **kwargs) -> None:
        kwargs.pop("disable", None)
        original(self, *args, **kwargs, disable=True)

    tqdm.__init__ = dummy_init
    try:
        yield
    finally:
        tqdm.__init__ = original


@contextmanager
def suppress_output(low_level: bool = False, stderr: bool = False) -> Generator[None, None, None]:
    """Suppress stdout/stderr from both Python and C extensions."""
    if low_level:
        with open(os.devnull, "w") as devnull:  # noqa: PTH123
            # Save the original file descriptors
            old_stdout_fd = os.dup(1)
            if stderr:
                old_stderr_fd = os.dup(2)

            try:
                os.dup2(devnull.fileno(), 1)  # Redirect C-level stdout
                if stderr:
                    os.dup2(devnull.fileno(), 2)  # Redirect C-level stderr
                yield
            finally:
                # Restore the original descriptors
                os.dup2(old_stdout_fd, 1)
                if stderr:
                    os.dup2(old_stderr_fd, 2)
                os.close(old_stdout_fd)
                if stderr:
                    os.close(old_stderr_fd)
    else:
        old_stdout = sys.stdout
        if stderr:
            old_stderr = sys.stderr
        try:
            # Redirect stdout to StringIO to suppress progress bars
            if stderr:
                with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                    yield
            else:
                with redirect_stdout(StringIO()):
                    yield
        finally:
            # Restore original stdout and stderr
            sys.stdout = old_stdout
            if stderr:
                sys.stderr = old_stderr
