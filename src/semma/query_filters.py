"""Start-of-query type filters (`image:`, `video:`, `screenshot:`).

To add a filter: call `register(QueryFilter(name, kinds, path_predicate))`.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from . import filetypes as ft
from .ocr import is_screenshot

_PREFIX = re.compile(r"\s*(\w[\w-]*):(.*)", re.DOTALL)


@dataclass(frozen=True, slots=True)
class QueryFilter:
    name: str
    kinds: frozenset[str]
    path_predicate: Callable[[str], bool] | None = None
    screenshot: bool = False


@dataclass(frozen=True, slots=True)
class ParsedQuery:
    text: str
    flt: QueryFilter | None = None


FILTERS: dict[str, QueryFilter] = {}


def register(flt: QueryFilter, *aliases: str) -> None:
    """Register `flt` under its casefolded name and any optional aliases."""
    FILTERS[flt.name.casefold()] = flt
    for alias in aliases:
        FILTERS[alias.casefold()] = flt


register(QueryFilter("image", frozenset({ft.IMAGE})))
register(QueryFilter("video", frozenset({ft.VIDEO})))
register(QueryFilter("screenshot", frozenset({ft.IMAGE}), is_screenshot, screenshot=True))


def parse_query(query: str) -> ParsedQuery:
    """Split a registered `<prefix>:` from the start of `query`; unknown prefixes stay literal."""
    m = _PREFIX.fullmatch(query)
    if m is not None:
        flt = FILTERS.get(m.group(1).casefold())
        if flt is not None:
            return ParsedQuery(m.group(2).strip(), flt)
    return ParsedQuery(query.strip(), None)


def sql_filter_fn(name: str, path: str) -> int:
    """SQLite scalar `gs_query_filter(name, path)`: 1 when `path` passes filter `name`'s path predicate."""
    flt = FILTERS.get(name.casefold())
    if flt is None or flt.path_predicate is None:
        return 1
    return int(flt.path_predicate(path))
