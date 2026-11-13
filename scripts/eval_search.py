"""Evaluate gemmasearch ranking against a local JSON query set.

Usage:
    uv run python scripts/eval_search.py [QUERIES.json] [--json OUT.json] [--compare OLD.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata
from pathlib import Path
from typing import Any

DEFAULT_QUERIES_PATH = Path("evals/local/search_queries.json")
SEARCH_LIMIT = 10
TOP_DISPLAY = 5


def _short(path: str, home: Path) -> str:
    h = str(home)
    return "~" + path[len(h):] if path == h or path.startswith(h + "/") else path


def _normalize_path(path: str, home: Path) -> str:
    if path == "~":
        expanded = str(home)
    elif path.startswith("~/"):
        expanded = str(home / path[2:])
    else:
        expanded = os.path.expanduser(path)
    return unicodedata.normalize("NFC", os.path.normpath(expanded))


def _rank_expected(expect: list[str], results: list[Any], home: Path) -> list[int | None]:
    path_rank: dict[str, int] = {}
    for rank, res in enumerate(results, start=1):
        for candidate in (res.path, *res.duplicate_paths):
            norm = _normalize_path(candidate, home)
            path_rank.setdefault(norm, rank)
    return [path_rank.get(_normalize_path(p, home)) for p in expect]


def _best_and_first_ranks(expect_ranks: list[int | None], any_first: bool = False) -> tuple[int | None, int | None]:
    """best_rank: the best rank of any expected file. first_rank: the rank of the ideal #1, the first
    expected file; with `any_first` (several files are equally right) the best rank of any of them."""
    found = [r for r in expect_ranks if r is not None]
    best_rank = min(found) if found else None
    first_rank = best_rank if any_first else (expect_ranks[0] if expect_ranks else None)
    return best_rank, first_rank


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    if n == 0:
        return {
            "queries": 0,
            "r_at_1": 0.0,
            "r_at_3": 0.0,
            "r_at_10": 0.0,
            "mrr": 0.0,
            "mean_results": 0.0,
            "mean_embed_ms": 0.0,
            "mean_search_ms": 0.0,
        }
    r1 = sum(1 for r in records if r["first_rank"] == 1) / n
    r3 = sum(1 for r in records if r["best_rank"] is not None and r["best_rank"] <= 3) / n
    r10 = sum(1 for r in records if r["best_rank"] is not None and r["best_rank"] <= 10) / n
    mrr = sum(r["reciprocal_rank"] for r in records) / n
    mean_results = sum(r["num_results"] for r in records) / n
    mean_embed_ms = sum(r["embed_ms"] for r in records) / n
    mean_search_ms = sum(r["search_ms"] for r in records) / n
    return {
        "queries": n,
        "r_at_1": round(r1, 4),
        "r_at_3": round(r3, 4),
        "r_at_10": round(r10, 4),
        "mrr": round(mrr, 4),
        "mean_results": round(mean_results, 2),
        "mean_embed_ms": round(mean_embed_ms, 2),
        "mean_search_ms": round(mean_search_ms, 2),
    }


def _fmt_rank(rank: int | None) -> str:
    return "-" if rank is None else str(rank)


def _print_query_report(idx: int, total: int, rec: dict[str, Any], home: Path) -> None:
    kinds_label = f"  kinds={rec['kinds']}" if rec["kinds"] else ""
    note_label = f"  ({rec['note']})" if rec["note"] else ""
    print(f"[{idx}/{total}] {rec['query']!r}{kinds_label}{note_label}")
    for exp, rank in zip(rec["expect"], rec["expect_ranks"], strict=True):
        print(f"  expect [{_fmt_rank(rank):>2}]  {_short(_normalize_path(exp, home), home)}")
    top = rec["results"][:TOP_DISPLAY]
    if not top:
        print("  top: (no results)")
        return
    for rank, res in enumerate(top, start=1):
        sim = f"{res['similarity']:.3f}" if res["similarity"] is not None else "  -  "
        matched = "+".join(res["matched"])
        dups = f" (+{res['duplicates']} dup)" if res["duplicates"] else ""
        print(f"    {rank}. {sim}  [{matched:<13}]  {_short(res['path'], home)}{dups}")


def _print_summary(summary: dict[str, Any], load_ms: float | None = None) -> None:
    n = summary["queries"]
    print("\n=== Summary ===")
    print(
        f"Queries: {n}  |  R@1: {summary['r_at_1']:.3f}  R@3: {summary['r_at_3']:.3f}  "
        f"R@10: {summary['r_at_10']:.3f}  MRR: {summary['mrr']:.3f}  "
        f"mean_results: {summary['mean_results']:.2f}"
    )
    timing = f"embed {summary['mean_embed_ms']:.1f} ms/q, search {summary['mean_search_ms']:.1f} ms/q"
    if load_ms is not None:
        timing = f"model load {load_ms:.0f} ms, " + timing
    print(f"Timings: {timing}")


def _rank_key(rec: dict[str, Any]) -> tuple[int, int]:
    inf = SEARCH_LIMIT + 1000
    best = rec["best_rank"] if rec["best_rank"] is not None else inf
    first = rec["first_rank"] if rec["first_rank"] is not None else inf
    return (best, first)


def _fmt_ranks(ranks: list[int | None]) -> str:
    return "[" + ", ".join(_fmt_rank(r) for r in ranks) + "]"


def _compare_runs(old_data: dict[str, Any], new_data: dict[str, Any], home: Path) -> None:
    old_sum = old_data["summary"]
    new_sum = new_data["summary"]
    print("\n=== Comparison (old -> new) ===")
    for key, label in (
        ("r_at_1", "R@1"),
        ("r_at_3", "R@3"),
        ("r_at_10", "R@10"),
        ("mrr", "MRR"),
        ("mean_results", "mean_results"),
    ):
        ov = old_sum[key]
        nv = new_sum[key]
        delta = nv - ov
        sign = "+" if delta > 0 else ""
        print(f"  {label:<12} {ov:.4f} -> {nv:.4f} ({sign}{delta:.4f})")

    def qkey(rec: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
        return (rec["query"], tuple(rec.get("kinds") or ()))

    old_by_q = {qkey(r): r for r in old_data["queries"]}
    new_by_q = {qkey(r): r for r in new_data["queries"]}

    better: list[tuple[dict[str, Any], dict[str, Any]]] = []
    worse: list[tuple[dict[str, Any], dict[str, Any]]] = []
    unchanged = 0
    for k, new_rec in new_by_q.items():
        old_rec = old_by_q.get(k)
        if old_rec is None:
            continue
        ok = _rank_key(old_rec)
        nk = _rank_key(new_rec)
        if nk < ok:
            better.append((old_rec, new_rec))
        elif nk > ok:
            worse.append((old_rec, new_rec))
        else:
            unchanged += 1

    print(f"\nBetter: {len(better)}  |  Worse: {len(worse)}  |  Unchanged: {unchanged}")
    if better:
        print("\nImproved queries:")
        for old_rec, new_rec in better:
            old_top = _short(old_rec["results"][0]["path"], home) if old_rec["results"] else "(none)"
            new_top = _short(new_rec["results"][0]["path"], home) if new_rec["results"] else "(none)"
            print(
                f"  + {new_rec['query']!r}: {_fmt_ranks(old_rec['expect_ranks'])} -> "
                f"{_fmt_ranks(new_rec['expect_ranks'])}  |  #1: {old_top} -> {new_top}"
            )
    if worse:
        print("\nRegressed queries:")
        for old_rec, new_rec in worse:
            old_top = _short(old_rec["results"][0]["path"], home) if old_rec["results"] else "(none)"
            new_top = _short(new_rec["results"][0]["path"], home) if new_rec["results"] else "(none)"
            print(
                f"  - {new_rec['query']!r}: {_fmt_ranks(old_rec['expect_ranks'])} -> "
                f"{_fmt_ranks(new_rec['expect_ranks'])}  |  #1: {old_top} -> {new_top}"
            )


def _load_queries(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a JSON list")
    queries: list[dict[str, Any]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict) or not isinstance(item.get("query"), str):
            raise ValueError(f"{path}[{i}] must be an object with a string 'query'")
        expect = item.get("expect")
        if not isinstance(expect, list) or not all(isinstance(p, str) for p in expect):
            raise ValueError(f"{path}[{i}] must have a list of string 'expect' paths")
        kinds = item.get("kinds")
        if kinds is not None and (not isinstance(kinds, list) or not all(isinstance(k, str) for k in kinds)):
            raise ValueError(f"{path}[{i}].kinds must be a list of strings or null")
        queries.append(
            {
                "query": item["query"],
                "expect": expect,
                "kinds": kinds,
                "any": bool(item.get("any", False)),
                "note": item.get("note") or "",
            }
        )
    return queries


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate gemmasearch ranking on a query set.")
    p.add_argument(
        "queries_file",
        nargs="?",
        type=Path,
        default=DEFAULT_QUERIES_PATH,
        help=f"JSON query file (default: {DEFAULT_QUERIES_PATH})",
    )
    p.add_argument("--config", type=Path, default=None, help="Config file path (optional)")
    p.add_argument("--json", dest="json_out", type=Path, default=None, help="Write per-query JSON results")
    p.add_argument("--compare", type=Path, default=None, help="Compare against an earlier --json output")
    return p


def main(argv: list[str] | None = None) -> int:
    from gemmasearch.config import load_config
    from gemmasearch.embedder import Embedder
    from gemmasearch.search import Searcher
    from gemmasearch.store import Store

    args = _build_parser().parse_args(argv)
    if not args.queries_file.is_file():
        print(f"Query file not found: {args.queries_file}", file=sys.stderr)
        return 1

    queries = _load_queries(args.queries_file)
    cfg = load_config(args.config)
    if not cfg.db_path.exists():
        print(f"No index at {cfg.db_path}. Run `gemmasearch scan` and `gemmasearch index`.", file=sys.stderr)
        return 1

    store = Store(cfg.db_path)
    store.db.execute("PRAGMA query_only=1")
    embedder = Embedder(cfg)
    try:
        stored_model = store.get_meta("model_id")
        if stored_model and stored_model != embedder.model_id:
            print(
                f"Index was built with {stored_model}, not {embedder.model_id}.",
                file=sys.stderr,
            )
            return 1

        embedder.load_query()
        # Rank like the daemon: all vectors in memory (older Stores without the cache use sqlite-vec).
        if hasattr(store, "enable_vector_cache"):
            cache = store.enable_vector_cache(cfg.vector_cache_max_chunks)
            with cache.lock:
                cache.ensure_fresh(store.reader())
        searcher = Searcher(store, cfg, embedder)

        records: list[dict[str, Any]] = []
        total = len(queries)
        for idx, entry in enumerate(queries, start=1):
            results = searcher.search(entry["query"], limit=SEARCH_LIMIT, kinds=entry["kinds"])
            t = searcher.last_timings
            expect_ranks = _rank_expected(entry["expect"], results, cfg.home)
            best_rank, first_rank = _best_and_first_ranks(expect_ranks, entry["any"])
            rr = (1.0 / best_rank) if best_rank is not None else 0.0
            rec = {
                "query": entry["query"],
                "kinds": entry["kinds"],
                "note": entry["note"],
                "expect": entry["expect"],
                "expect_ranks": expect_ranks,
                "best_rank": best_rank,
                "first_rank": first_rank,
                "reciprocal_rank": round(rr, 6),
                "num_results": len(results),
                "embed_ms": round(t.embed_ms, 2),
                "search_ms": round(t.search_ms, 2),
                "results": [r.to_dict() for r in results],
            }
            records.append(rec)
            _print_query_report(idx, total, rec, cfg.home)

        summary = _summarize(records)
        _print_summary(summary, load_ms=embedder.load_seconds * 1000)

        payload = {
            "model_id": embedder.model_id,
            "summary": summary,
            "queries": records,
        }
        if args.json_out is not None:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Wrote {args.json_out}")

        if args.compare is not None:
            old_data = json.loads(args.compare.read_text(encoding="utf-8"))
            _compare_runs(old_data, payload, cfg.home)
    finally:
        embedder.unload_query()
        store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
