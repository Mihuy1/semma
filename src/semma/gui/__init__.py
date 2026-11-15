"""The Semma window (`semma gui`): a separate process that is only a client of the daemon.

Nothing here imports PySide6 at module level, so `semma` starts without it."""

from __future__ import annotations

import sys
from pathlib import Path


def main(config_path: Path | None = None) -> int:
    try:
        import PySide6.QtWidgets  # noqa: F401
    except ImportError as exc:
        print(f"semma gui needs PySide6 (pip install PySide6-Essentials): {exc}", file=sys.stderr)
        return 1
    from .app import run

    return run(config_path)
