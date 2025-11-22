"""
Lightweight logging helpers for burn severity processing.
"""

from __future__ import annotations

import os


def append_log(log_path: str, line: str) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as file:
        file.write(line.rstrip("\n") + "\n")
        file.flush()
        os.fsync(file.fileno())


__all__ = ["append_log"]
