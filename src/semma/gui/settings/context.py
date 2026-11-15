"""What a settings section works from: the saved settings. No Qt here."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...config import Config, canonical, default_config_path, load_config
from ...config_write import ConfigWriteError, collapse_home, read_raw

Changes = dict[str, dict[str, Any]]
"""What `config_write.update_config` takes: section -> key -> value; None removes the key so
the setting goes back to its default."""


@dataclass
class SettingsContext:
    """The saved settings, read once when the Settings page loads or after Apply / Revert.

    `config` holds the effective values (defaults plus config.toml), `raw` only what
    config.toml sets explicitly."""
    config_path: Path
    config: Config
    raw: dict[str, Any]

    @property
    def home(self) -> Path:
        return self.config.home

    @classmethod
    def load(cls, config_path: Path | None = None, home: Path | None = None) -> SettingsContext:
        """Raises ConfigWriteError when config.toml cannot be read or has invalid values."""
        path = config_path or default_config_path(home)
        raw = read_raw(path)
        try:
            cfg = load_config(path, home)
        except Exception as exc:  # a value of the wrong type in a hand-edited file
            raise ConfigWriteError(f"cannot read {path}: {exc}") from exc
        return cls(path, cfg, raw)

    def is_set(self, section: str, key: str) -> bool:
        table = self.raw.get(section) if section else self.raw
        return isinstance(table, dict) and key in table

    def display_path(self, path: Path | str) -> str:
        """A path as written to config.toml and shown in the window: under home with "~"."""
        return collapse_home(Path(path), self.home)

    def resolve(self, text: str) -> Path:
        """The inverse of display_path, with symlinks resolved, for comparing folders."""
        if text == "~" or text.startswith("~/"):
            return canonical(self.home / text[2:])
        return canonical(Path(text))


@dataclass(frozen=True)
class Problem:
    """A note under a section's form. A blocking one keeps Apply disabled."""
    text: str
    blocking: bool = False


def merge_changes(*parts: Changes) -> Changes:
    out: Changes = {}
    for part in parts:
        for section, entries in part.items():
            if entries:
                out.setdefault(section, {}).update(entries)
    return out
