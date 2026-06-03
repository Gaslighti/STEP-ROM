"""Logging helpers for the STEP-ROM pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

CONSOLE_MARKER = "[CONSOLE]"
CONSOLE_RECORD_ATTR = "step_rom_console"


class ConsoleOnlyFilter(logging.Filter):
    """Allow only explicitly marked records and errors to reach stdout/stderr."""

    def filter(self, record: logging.LogRecord) -> bool:
        return (
            bool(getattr(record, CONSOLE_RECORD_ATTR, False))
            or record.levelno >= logging.ERROR
        )


class ConciseConsoleFormatter(logging.Formatter):
    """Format console records without detailed tracebacks kept in the file log."""

    def format(self, record: logging.LogRecord) -> str:
        exception_info = record.exc_info
        exception_text = record.exc_text
        record.exc_info = None
        record.exc_text = None
        try:
            return super().format(record)
        finally:
            record.exc_info = exception_info
            record.exc_text = exception_text


def log_console(
    logger: logging.Logger, level: int, msg: str, *args: Any, **kwargs: Any
) -> None:
    """Log a record to the detailed file and mark it as concise console output."""

    extra = dict(kwargs.pop("extra", {}))
    extra[CONSOLE_RECORD_ATTR] = True
    logger.log(level, msg, *args, extra=extra, **kwargs)


class LoggerWriter:
    """File-like adapter that sends redirected print output to a logger.

    Redirected legacy output is detailed-file-only by default.  A legacy line can
    opt in to concise console output by starting with ``CONSOLE_MARKER``.
    """

    def __init__(
        self, logger: logging.Logger, level: int, *, echo_to_console: bool = False
    ) -> None:
        self.logger = logger
        self.level = level
        self.echo_to_console = echo_to_console
        self._buffer = ""

    def write(self, text: str) -> int:
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._log_line(line)
        return len(text)

    def flush(self) -> None:
        if self._buffer.strip():
            self._log_line(self._buffer.rstrip())
        self._buffer = ""

    def _log_line(self, line: str) -> None:
        if not line.strip():
            return

        console = self.echo_to_console
        message = line
        if line.startswith(CONSOLE_MARKER):
            console = True
            message = line[len(CONSOLE_MARKER) :].lstrip()

        self.logger.log(self.level, message, extra={CONSOLE_RECORD_ATTR: console})


def configure_pipeline_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("step_rom.pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.addFilter(ConsoleOnlyFilter())
    console_handler.setFormatter(ConciseConsoleFormatter("%(message)s"))
    logger.addHandler(console_handler)

    return logger
