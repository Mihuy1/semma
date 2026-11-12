"""SQLite storage. Schema v1: the `files` table filled by the scanner (the indexing queue)."""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterable
from pathlib import Path

from .scanner import ScannedFile

SCHEMA_V1 = """
CREATE TABLE files (
    id          INTEGER PRIMARY KEY,
    path        TEXT    NOT NULL UNIQUE,
    kind        TEXT    NOT NULL,
    size        INTEGER NOT NULL,
    mtime       REAL    NOT NULL,
    in_project  INTEGER NOT NULL,
    priority    REAL    NOT NULL,
    status      TEXT    NOT NULL DEFAULT 'pending',  -- pending | indexed | failed
    error       TEXT,
    indexed_at  REAL,
    last_seen   INTEGER NOT NULL                     -- scan generation that last saw the file
);
CREATE INDEX files_queue ON files (status, priority DESC, mtime DESC);
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
"""

MIGRATIONS: list[str] = [SCHEMA_V1]

UPSERT = """
INSERT INTO files (path, kind, size, mtime, in_project, priority, status, last_seen)
VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)
ON CONFLICT (path) DO UPDATE SET
    status = CASE WHEN files.size != excluded.size OR files.mtime != excluded.mtime
                       OR files.kind != excluded.kind
                  THEN 'pending' ELSE files.status END,
    error  = CASE WHEN files.size != excluded.size OR files.mtime != excluded.mtime
                  THEN NULL ELSE files.error END,
    kind = excluded.kind, size = excluded.size, mtime = excluded.mtime,
    in_project = excluded.in_project, priority = excluded.priority, last_seen = excluded.last_seen
"""


class Store:
    def __init__(self, path: Path | str):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = sqlite3.connect(str(path), timeout=30, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self._migrate()

    def _migrate(self) -> None:
        version = self.db.execute("PRAGMA user_version").fetchone()[0]
        for i, script in enumerate(MIGRATIONS[version:], start=version + 1):
            with self.db:
                self.db.executescript(f"BEGIN; {script}; PRAGMA user_version = {i}; COMMIT;")

    def close(self) -> None:
        self.db.close()

    # -- meta -----------------------------------------------------------------
    def get_meta(self, key: str, default: str | None = None) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else default

    def set_meta(self, key: str, value: str) -> None:
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))

    # -- scanning -------------------------------------------------------------
    def begin_scan(self) -> int:
        gen = int(self.get_meta("scan_generation", "0")) + 1
        self.set_meta("scan_generation", str(gen))
        return gen

    def upsert_scanned(self, files: Iterable[ScannedFile], generation: int, batch: int = 2000) -> int:
        n = 0
        rows: list[tuple] = []
        for f in files:
            rows.append((f.path, f.kind, f.size, f.mtime, int(f.in_project), f.score.total, generation))
            if len(rows) >= batch:
                n += self._flush(rows)
        n += self._flush(rows)
        return n

    def _flush(self, rows: list[tuple]) -> int:
        if not rows:
            return 0
        with self.db:
            self.db.executemany(UPSERT, rows)
        count = len(rows)
        rows.clear()
        return count

    def finish_scan(self, generation: int, scanned_roots: Iterable[Path]) -> int:
        """Forget files under the scanned roots that this scan did not see (deleted or now excluded).

        Only roots that were actually scanned are touched, so an unmounted drive keeps its index.
        """
        removed = 0
        with self.db:
            for root in scanned_roots:
                prefix = str(root).rstrip("/") + "/"
                cur = self.db.execute(
                    "DELETE FROM files WHERE last_seen < ? AND substr(path, 1, ?) = ?",
                    (generation, len(prefix), prefix),
                )
                removed += cur.rowcount
        self.set_meta("last_scan_at", str(time.time()))
        return removed

    # -- queue ----------------------------------------------------------------
    def counts(self) -> dict[tuple[str, str], int]:
        rows = self.db.execute("SELECT kind, status, count(*) FROM files GROUP BY kind, status")
        return {(r[0], r[1]): r[2] for r in rows}
