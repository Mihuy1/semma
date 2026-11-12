"""Command line interface: `gemmasearch scan | index | search | status | reset`."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

from . import filetypes as ft
from .config import Config, load_config
from .scanner import ScannedFile, Scanner
from .store import Store


def _human(n: float) -> str:
    for unit in ("B", "K", "M", "G", "T"):
        if n < 1024 or unit == "T":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return str(n)


def _short(path: str, home: Path) -> str:
    h = str(home)
    return "~" + path[len(h):] if path == h or path.startswith(h + "/") else path


def _location_label(cfg: Config, scanner: Scanner, path: str) -> str:
    anchor, _ = scanner.scorer.location(Path(path))
    return _short(str(anchor), cfg.home) if anchor else "(other)"


def cmd_scan(args: argparse.Namespace, cfg: Config) -> int:
    scanner = Scanner(cfg)
    roots = [Path(r).expanduser().resolve() for r in args.root] if args.root else None

    if args.dry_run:
        files = list(scanner.scan(roots))
    else:
        store = Store(cfg.db_path)
        gen = store.begin_scan()
        files = []
        store.upsert_scanned((files.append(f) or f for f in scanner.scan(roots)), gen)
        removed = store.finish_scan(gen, scanner.stats.roots)

    st = scanner.stats
    print(f"Scanned {', '.join(_short(str(r), cfg.home) for r in st.roots)} in {st.seconds:.1f}s: "
          f"{st.dirs:,} dirs, {st.files_seen:,} files, {st.matched:,} indexable, "
          f"{st.too_large} too large, {st.errors} unreadable")
    top_pruned = ", ".join(f"{k} x{v}" for k, v in st.pruned.most_common(8))
    print(f"Pruned dirs: {top_pruned}")

    by_kind = Counter(f.kind for f in files)
    size_by_kind: Counter[str] = Counter()
    for f in files:
        size_by_kind[f.kind] += f.size
    print("\nBy kind:   " + "  ".join(f"{k} {by_kind[k]:,} ({_human(size_by_kind[k])})" for k in ft.KINDS if by_kind[k]))

    groups: dict[str, list[ScannedFile]] = defaultdict(list)
    for f in files:
        groups[_location_label(cfg, scanner, f.path)].append(f)
    print("\nBy location (files, score min / avg / max, files in projects):")
    for label, fs in sorted(groups.items(), key=lambda kv: -max(f.score.total for f in kv[1])):
        scores = [f.score.total for f in fs]
        proj = sum(f.in_project for f in fs)
        print(f"  {label:<28} {len(fs):>7,}   {min(scores):5.1f} / {sum(scores)/len(scores):5.1f} / {max(scores):5.1f}"
              f"   {proj:,} in projects")

    buckets = Counter(int(f.score.total // 10) * 10 for f in files)
    print("\nScore histogram:  " + "  ".join(f"{b}-{b+9}: {buckets[b]:,}" for b in sorted(buckets, reverse=True)))

    files.sort(key=lambda f: (-f.score.total, -f.mtime))
    show = files[: args.top]
    if args.per_location:
        show = []
        for fs in groups.values():
            show += sorted(fs, key=lambda f: (-f.score.total, -f.mtime))[: args.per_location]
        show.sort(key=lambda f: (-f.score.total, -f.mtime))
    if show:
        print(f"\n{'score':>5}  {'kind':<6} {'size':>7}  {'modified':<10}  path")
        for f in show:
            line = (f"{f.score.total:5.1f}  {f.kind:<6} {_human(f.size):>7}  "
                    f"{time.strftime('%Y-%m-%d', time.localtime(f.mtime))}  {_short(f.path, cfg.home)}")
            if args.explain:
                line += f"   [{f.score.explain()}]"
            print(line)

    if not args.dry_run:
        print(f"\nDatabase {cfg.db_path}: {len(files):,} files upserted, {removed:,} removed")
    return 0


def _duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def cmd_status(args: argparse.Namespace, cfg: Config) -> int:
    if not cfg.db_path.exists():
        print("No index yet. Run `gemmasearch scan`.")
        return 1
    store = Store(cfg.db_path)
    counts = store.counts()
    statuses = sorted({s for _, s in counts})
    print(f"{'kind':<8}" + "".join(f"{s:>10}" for s in statuses))
    for kind in ft.KINDS:
        if any(k == kind for k, _ in counts):
            print(f"{kind:<8}" + "".join(f"{counts.get((kind, s), 0):>10,}" for s in statuses))

    index = store.index_counts()
    vectors = index.pop("vectors")
    chunks = sum(index.values())
    per_modality = ", ".join(f"{m} {n:,}" for m, n in sorted(index.items()))
    print(f"\nChunks: {chunks:,}" + (f" ({per_modality})" if per_modality else "") + f"   vectors: {vectors:,}")
    model = store.get_meta("model_id")
    if model:
        print(f"Model: {model} ({store.embedding_dim()}-d)")
    last = store.get_meta("last_scan_at")
    if last:
        print(f"Last scan: {time.strftime('%Y-%m-%d %H:%M', time.localtime(float(last)))}")

    failed = store.failed_files()
    if failed:
        errors = Counter((r["error"] or "").split("\n")[0][:100] for r in failed)
        print(f"\nFailed files: {len(failed):,}" + ("" if args.failed else "  (--failed to list them)"))
        for err, n in errors.most_common(5 if not args.failed else None):
            print(f"  {n:>6,}  {err}")
        if args.failed:
            print()
            for r in failed:
                print(f"  {r['kind']:<6} {_short(r['path'], cfg.home)}\n         {r['error']}")
    return 0


def _install_stop_handler(stop: threading.Event) -> None:
    """First Ctrl-C asks for a graceful stop; a second one aborts immediately."""
    def handler(signum: int, frame: object) -> None:
        if stop.is_set():
            raise KeyboardInterrupt
        stop.set()
        print("\nStopping after the current batch (Ctrl-C again to abort)...", file=sys.stderr, flush=True)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def cmd_index(args: argparse.Namespace, cfg: Config) -> int:
    from .embedder import Embedder
    from .indexer import IndexStats, Indexer

    if not cfg.db_path.exists():
        print("No index yet. Run `gemmasearch scan` first.")
        return 1
    store = Store(cfg.db_path)
    embedder = Embedder(cfg)
    indexer = Indexer(store, cfg, embedder, prefetch=not args.no_prefetch)
    stop = threading.Event()
    _install_stop_handler(stop)

    if indexer.count_pending(args.kind) == 0:
        print("Nothing to index.")
        return 0
    print(f"Loading {cfg.model_name}...", end=" ", flush=True)
    embedder.load()
    print(f"{embedder.device}, {embedder.load_seconds:.1f}s")

    tty = sys.stdout.isatty()
    last_print = 0.0

    def progress(st: IndexStats) -> None:
        nonlocal last_print
        now = time.monotonic()
        if now - last_print < (0.25 if tty else 10) and st.current:
            return
        last_print = now
        rate = st.done / st.seconds if st.seconds > 0 else 0.0
        eta = _duration((st.total - st.done) / rate) if rate > 0 else "?"
        line = (f"{st.done:,}/{st.total:,} files  {rate:.1f} files/s  {st.chunks:,} chunks  "
                f"{st.failed} failed  {st.skipped} skipped  ETA {eta}")
        if tty:
            cur = _short(st.current, cfg.home) if st.current else ""
            width = 120 - len(line)
            cur = ("..." + cur[-(width - 3):]) if len(cur) > width > 3 else cur
            print(f"\r\033[K{line}  {cur}", end="", flush=True)
        else:
            print(line, flush=True)

    st = indexer.run(limit=args.limit, kinds=args.kind, progress=progress, stop=stop)
    embedder.release_cache()
    if tty:
        print()
    rate = st.done / st.seconds if st.seconds > 0 else 0.0
    print(f"{'Stopped' if st.stopped else 'Done'}: {st.indexed:,} indexed, {st.failed:,} failed, "
          f"{st.skipped:,} skipped (changed since scan), {st.chunks:,} chunks in {_duration(st.seconds)} "
          f"({rate:.1f} files/s, embedding {st.embed_seconds:.1f}s)")
    return 130 if st.stopped else 0


def cmd_search(args: argparse.Namespace, cfg: Config) -> int:
    from .embedder import Embedder
    from .search import Searcher

    if not cfg.db_path.exists():
        print("No index yet. Run `gemmasearch scan` and `gemmasearch index`.")
        return 1
    store = Store(cfg.db_path)
    embedder = Embedder(cfg)
    stored_model = store.get_meta("model_id")
    if stored_model and stored_model != embedder.model_id:
        print(f"Index was built with {stored_model}, not {embedder.model_id}; re-run `gemmasearch index`.",
              file=sys.stderr)
        return 1
    embedder.load()
    searcher = Searcher(store, cfg, embedder)
    results = searcher.search(args.query, limit=args.n, kinds=args.kind)
    t = searcher.last_timings
    if args.json:
        print(json.dumps({"results": [r.to_dict() for r in results],
                          "timings_ms": {"load": round(embedder.load_seconds * 1000),
                                         "embed": round(t.embed_ms, 1), "search": round(t.search_ms, 1)}},
                         ensure_ascii=False, indent=2))
        return 0
    for r in results:
        sim = f"{r.similarity:.3f}" if r.similarity is not None else "  -  "
        where = f" [{r.locator}]" if r.locator else ""
        print(f"{sim}  {r.kind:<6} {_short(r.path, cfg.home)}{where}   ({'+'.join(r.matched)})")
        if r.snippet:
            print(f"               {r.snippet[:150]}")
    if not results:
        print("No matches.")
    print(f"\n{len(results)} results; model load {embedder.load_seconds * 1000:.0f} ms, "
          f"query embedding {t.embed_ms:.1f} ms, search {t.search_ms:.1f} ms")
    return 0


def cmd_reset(args: argparse.Namespace, cfg: Config) -> int:
    if not args.yes:
        what = "the whole database (scan queue included)" if args.all else "all embeddings and chunks"
        print(f"This deletes {what} in {cfg.db_path}. Re-run with --yes to confirm.")
        return 1
    if args.all:
        for suffix in ("", "-wal", "-shm"):
            Path(f"{cfg.db_path}{suffix}").unlink(missing_ok=True)
        print(f"Deleted {cfg.db_path}")
        return 0
    if not cfg.db_path.exists():
        print("No index.")
        return 0
    store = Store(cfg.db_path)
    store.reset_index()
    store.db.execute("VACUUM")
    print("Index cleared; every file is pending again.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gemmasearch", description="Semantic desktop search")
    p.add_argument("--config", type=Path, help="config file (default ~/.config/gemmasearch/config.toml)")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("scan", help="walk the filesystem and score files (phase 1)")
    s.add_argument("--dry-run", action="store_true", help="print the ranking without writing the database")
    s.add_argument("--top", type=int, default=25, help="show the N highest-priority files")
    s.add_argument("--per-location", type=int, default=0, metavar="N", help="show the top N files of every location")
    s.add_argument("--explain", action="store_true", help="show how each score was computed")
    s.add_argument("--root", action="append", help="scan only this directory (repeatable)")
    s.set_defaults(func=cmd_scan)

    st = sub.add_parser("status", help="show index counts")
    st.add_argument("--failed", action="store_true", help="list every file that failed to index")
    st.set_defaults(func=cmd_status)

    ix = sub.add_parser("index", help="extract and embed pending files in priority order (phase 2)")
    ix.add_argument("--limit", type=int, help="index at most N files")
    ix.add_argument("--kind", action="append", choices=ft.KINDS, help="only this kind (repeatable)")
    ix.add_argument("--no-prefetch", action="store_true", help="extract and embed on one thread")
    ix.set_defaults(func=cmd_index)

    se = sub.add_parser("search", help="search the index")
    se.add_argument("query")
    se.add_argument("-n", type=int, default=10, help="number of results")
    se.add_argument("--kind", action="append", choices=ft.KINDS, help="only this kind (repeatable)")
    se.add_argument("--json", action="store_true", help="machine-readable output")
    se.set_defaults(func=cmd_search)

    rs = sub.add_parser("reset", help="drop all embeddings (files are queued again)")
    rs.add_argument("--yes", action="store_true", help="confirm")
    rs.add_argument("--all", action="store_true", help="delete the whole database file instead")
    rs.set_defaults(func=cmd_reset)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    try:
        return args.func(args, cfg)
    except BrokenPipeError:
        return 0


if __name__ == "__main__":
    sys.exit(main())
