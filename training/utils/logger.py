"""Colored terminal logging for the training pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from colorama import Fore, Style, init


init(autoreset=True)


@dataclass(slots=True)
class TrainingLogger:
    """Small console logger with color-coded training messages."""

    show_timestamps: bool = True

    def _prefix(self, level: str) -> str:
        timestamp = datetime.now().strftime("%H:%M:%S") if self.show_timestamps else ""
        if timestamp:
            return f"[{timestamp}] [{level}]"
        return f"[{level}]"

    def _log(self, level: str, message: str, color: str) -> None:
        print(f"{color}{self._prefix(level)} {message}{Style.RESET_ALL}")

    def info(self, message: str) -> None:
        self._log("INFO", message, Fore.CYAN)

    def success(self, message: str) -> None:
        self._log("SUCCESS", message, Fore.GREEN)

    def warning(self, message: str) -> None:
        self._log("WARNING", message, Fore.YELLOW)

    def error(self, message: str) -> None:
        self._log("ERROR", message, Fore.RED)

    def section(self, title: str) -> None:
        border = "=" * max(len(title), 72)
        print(f"\n{Fore.MAGENTA}{border}\n{title}\n{border}{Style.RESET_ALL}")

    def metric(self, label: str, value: Any) -> None:
        self._log("METRIC", f"{label}: {value}", Fore.BLUE)


logger = TrainingLogger()
