"""Atomic, comment-preserving writer for Semma's config.toml."""

from __future__ import annotations

import os
import stat
import tempfile
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import tomlkit
from tomlkit.container import OutOfOrderTableProxy
from tomlkit.exceptions import TOMLKitError
from tomlkit.items import InlineTable, Table

from . import config

_SECTION_NAMES: frozenset[str] = frozenset(k for k in config.KNOWN_KEYS if k)
_TOP_LEVEL_KEYS: frozenset[str] = config.KNOWN_KEYS[""] - _SECTION_NAMES
_TABLE_LIKE = (Table, InlineTable, OutOfOrderTableProxy)


class ConfigWriteError(Exception):
    """Raised when config.toml cannot be parsed, validated, or written."""


def _validate_changes(changes: Mapping[str, Mapping[str, Any]]) -> None:
    if not isinstance(changes, Mapping):
        raise ConfigWriteError("changes must be a mapping of section -> {key: value}")
    for section, entries in changes.items():
        if not isinstance(section, str) or section not in config.KNOWN_KEYS:
            raise ConfigWriteError(f"unknown config section {section!r}")
        if not isinstance(entries, Mapping):
            raise ConfigWriteError(f"changes for section {section!r} must be a mapping")
        allowed = _TOP_LEVEL_KEYS if section == "" else config.KNOWN_KEYS[section]
        for key in entries:
            if not isinstance(key, str) or key not in allowed:
                loc = f"section {section!r}" if section else "top level"
                raise ConfigWriteError(f"unknown config key {key!r} in {loc}")


def _to_toml_item(value: Any) -> Any:
    if isinstance(value, Mapping):
        for k in value:
            if not isinstance(k, str):
                raise ConfigWriteError(f"TOML table keys must be strings, got {type(k).__name__}")
    return tomlkit.item(value)


def _update_mapping(target: Any, mapping: Mapping[str, Any]) -> None:
    for k, v in mapping.items():
        if not isinstance(k, str):
            raise ConfigWriteError(f"TOML table keys must be strings, got {type(k).__name__}")
        _set_entry(target, k, v)
    for existing_key in list(target.keys()):
        if existing_key not in mapping:
            del target[existing_key]


def _set_entry(container: Any, key: str, value: Any) -> None:
    if isinstance(value, Mapping):
        if key in container and isinstance(container[key], _TABLE_LIKE):
            existing = container[key]
            if not value and isinstance(existing, OutOfOrderTableProxy):
                container[key] = tomlkit.inline_table()
            else:
                _update_mapping(existing, value)
            return
        container[key] = _to_toml_item(value)
        return
    if key in container and not isinstance(container[key], _TABLE_LIKE):
        existing_item = container[key]
        cur = existing_item.unwrap() if hasattr(existing_item, "unwrap") else existing_item
        if type(cur) is type(value) and cur == value:
            return
    container[key] = _to_toml_item(value)


def update_config(path: Path, changes: dict[str, dict[str, Any]]) -> None:
    """Apply *changes* to *path* atomically while preserving comments, blank lines, and unknown keys."""
    _validate_changes(changes)

    existing_mode: int | None = None
    try:
        target = path.resolve() if path.is_symlink() else path
        st = target.stat()
        existing_mode = stat.S_IMODE(st.st_mode)
        raw_text = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        doc = tomlkit.document()
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigWriteError(f"cannot read {path}: {exc}") from exc
    else:
        try:
            doc = tomlkit.parse(raw_text)
        except TOMLKitError as exc:
            raise ConfigWriteError(f"invalid TOML in {target}: {exc}") from exc

    try:
        for section, entries in changes.items():
            if section == "":
                for key, value in entries.items():
                    if value is None:
                        if key in doc:
                            del doc[key]
                    else:
                        _set_entry(doc, key, value)
                continue

            if section not in doc:
                if all(value is None for value in entries.values()):
                    continue
                tbl: Any = tomlkit.table()
                doc[section] = tbl
            else:
                tbl = doc[section]
                if not isinstance(tbl, _TABLE_LIKE):
                    raise ConfigWriteError(f"section {section!r} in {target} is not a table")

            for key, value in entries.items():
                if value is None:
                    if key in tbl:
                        del tbl[key]
                else:
                    _set_entry(tbl, key, value)

            if len(tbl) == 0:
                del doc[section]

        rendered = doc.as_string()
    except ConfigWriteError:
        raise
    except (TOMLKitError, TypeError, ValueError) as exc:
        raise ConfigWriteError(f"cannot serialize config changes: {exc}") from exc

    tmp_path: Path | None = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as fh:
            tmp_path = Path(fh.name)
            fh.write(rendered)
            fh.flush()
            os.fsync(fh.fileno())
        if existing_mode is not None:
            os.chmod(tmp_path, existing_mode)
        os.replace(tmp_path, target)
        tmp_path = None
    except OSError as exc:
        raise ConfigWriteError(f"cannot write {target}: {exc}") from exc
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def read_raw(path: Path) -> dict[str, Any]:
    """Return the contents of *path* as plain Python dicts and lists, or {} if missing."""
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except FileNotFoundError:
        return {}
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigWriteError(f"cannot read {path}: {exc}") from exc


def collapse_home(path: Path, home: Path) -> str:
    """Collapse *path* relative to *home* into '~/...' or '~', inverse of config._expand."""
    abs_path = Path(os.path.abspath(path))
    abs_home = Path(os.path.abspath(home))
    if abs_path == abs_home:
        return "~"
    try:
        rel = abs_path.relative_to(abs_home)
    except ValueError:
        return str(abs_path)
    return f"~/{rel.as_posix()}"
