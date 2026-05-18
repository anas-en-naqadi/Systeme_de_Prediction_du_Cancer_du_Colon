"""Colored logging helpers for the FastAPI backend."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


RESET = "\033[0m"
BOLD = "\033[1m"
COLORS = {
    "INFO": "\033[36m",
    "SUCCESS": "\033[32m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
    "REQUEST": "\033[35m",
}


class _ColorFormatter(logging.Formatter):
    """ANSI color formatter for concise terminal output."""

    def format(self, record: logging.LogRecord) -> str:
        color = COLORS.get(record.levelname, "")
        timestamp = datetime.now().strftime("%H:%M:%S")
        message = super().format(record)
        return f"{color}[{timestamp}] [{record.levelname}] {message}{RESET}"


@dataclass(slots=True)
class BackendLogger:
    """Lightweight wrapper around the standard logging module."""

    name: str = "colon_cancer_api"
    _logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._logger = logging.getLogger(self.name)
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(_ColorFormatter("%(message)s"))
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)
            self._logger.propagate = False

    def info(self, message: str) -> None:
        self._logger.info(message)

    def success(self, message: str) -> None:
        print(f"{COLORS['SUCCESS']}[{datetime.now().strftime('%H:%M:%S')}] [SUCCESS] {message}{RESET}")

    def warning(self, message: str) -> None:
        print(f"{COLORS['WARNING']}[{datetime.now().strftime('%H:%M:%S')}] [WARNING] {message}{RESET}")

    def error(self, message: str) -> None:
        print(f"{COLORS['ERROR']}[{datetime.now().strftime('%H:%M:%S')}] [ERROR] {message}{RESET}")

    def request(self, method: str, path: str, status_code: int, duration_ms: float, client_host: str | None) -> None:
        host = client_host or "unknown"
        print(
            f"{COLORS['REQUEST']}[{datetime.now().strftime('%H:%M:%S')}] [REQUEST] "
            f"{method} {path} -> {status_code} | {duration_ms:.1f}ms | {host}{RESET}"
        )

    def startup(self, message: str) -> None:
        print(f"{COLORS['INFO']}[{datetime.now().strftime('%H:%M:%S')}] [INFO] {message}{RESET}")


logger = BackendLogger()
