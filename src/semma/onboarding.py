"""Whether this user has finished first-run setup.

The flag lives in config.toml (`onboarding_complete`). When the key is absent, an index that
has already been scanned or that already has files counts as finished, so an upgrade does not
ask again. The database file existing is not enough: the daemon creates an empty one at start.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .config import Config
from .store import Store


def _index_was_used(db_path: Path) -> bool:
    """True when a scan has finished or the index already has a file. Does not create the db."""
    if not db_path.is_file():
        return False
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error:
        return False
    try:
        try:
            row = conn.execute("SELECT value FROM meta WHERE key = 'last_scan_at'").fetchone()
        except sqlite3.Error:
            return False
        if row and row[0]:
            return True
        try:
            return conn.execute("SELECT 1 FROM files LIMIT 1").fetchone() is not None
        except sqlite3.Error:
            return False
    finally:
        conn.close()


def _store_was_used(store: Store) -> bool:
    if store.get_meta("last_scan_at"):
        return True
    try:
        return store.db.execute("SELECT 1 FROM files LIMIT 1").fetchone() is not None
    except sqlite3.Error:
        return False


def onboarding_done(cfg: Config, store: Store | None = None) -> bool:
    """True when the daemon may scan and index.

    An explicit false always waits, even if an index already exists (Run setup again).
    An explicit true allows indexing. A missing key allows it only for an index that was
    already in use.
    """
    if cfg.onboarding_complete is True:
        return True
    if cfg.onboarding_complete is False:
        return False
    if store is not None:
        return _store_was_used(store)
    return _index_was_used(cfg.db_path)
