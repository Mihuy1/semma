"""Command line interface: `gemmasearch scan | status`."""

from __future__ import annotations

import argparse
import sys
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
    last = store.get_meta("last_scan_at")
    if last:
        print(f"\nLast scan: {time.strftime('%Y-%m-%d %H:%M', time.localtime(float(last)))}")
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
    st.set_defaults(func=cmd_status)
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
