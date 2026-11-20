#!/usr/bin/env python3
"""Ratcheted comment-banning linter for Semma.

Bans prose `#` comments across Python, shell, and YAML files while allowing tool
directives (`#!`, `# noqa`, `# type: ignore`, `# pyright:`, `# fmt:`, `# pragma:`,
`# ruff:`, `# pylint:`, `# mypy:`, `# shellcheck`, encoding cookies).

Uses `.comment-baseline.json` to enforce a strict ratchet:
- Any file with more comments than its baseline entry (default 0) fails.
- Any file with fewer comments than its baseline entry fails so the baseline is
  ratcheted downward via `--update-baseline`.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import tokenize

SKIP_DIRS = frozenset({
    ".git",
    ".venv",
    "__pycache__",
    ".ruff_cache",
    ".mypy_cache",
    ".pytest_cache",
    "dist",
    "build",
    "node_modules",
})

DIRECTIVE_RE = re.compile(
    r"^#\s*(?:!|noqa\b|type:\s*ignore|pyright:|fmt:|pragma:|ruff:|pylint:|mypy:|coding[:=]|-\*-|shellcheck\b)",
    re.IGNORECASE,
)

HEREDOC_RE = re.compile(r"<<-?\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?")

DEFAULT_BASELINE = ".comment-baseline.json"


def _is_skipped(path: pathlib.Path, root: pathlib.Path) -> bool:
    rel = path.relative_to(root)
    return any(part in SKIP_DIRS for part in rel.parts)


def scan_python_file(path: pathlib.Path) -> list[tuple[int, str]]:
    """Return `(line_number, comment_text)` for banned prose comments in a Python file."""
    violations: list[tuple[int, str]] = []
    with path.open("rb") as fh:
        for tok in tokenize.tokenize(fh.readline):
            if tok.type == tokenize.COMMENT and not DIRECTIVE_RE.match(tok.string):
                violations.append((tok.start[0], tok.string))
    return violations


def scan_shell_or_yaml_file(path: pathlib.Path) -> list[tuple[int, str]]:
    """Return `(line_number, comment_text)` for banned standalone prose comments."""
    violations: list[tuple[int, str]] = []
    is_shell = path.suffix == ".sh" or path.name == "semma"
    in_heredoc: str | None = None
    for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        stripped = line.lstrip()
        if in_heredoc is not None:
            if line.strip() == in_heredoc:
                in_heredoc = None
            continue
        if is_shell and not stripped.startswith("#"):
            match = HEREDOC_RE.search(line)
            if match:
                in_heredoc = match.group(1)
        if stripped.startswith("#"):
            if (lineno == 1 and stripped.startswith("#!")) or DIRECTIVE_RE.match(stripped):
                continue
            violations.append((lineno, stripped))
    return violations


def iter_target_files(root: pathlib.Path) -> list[pathlib.Path]:
    """Collect all Python, shell, and YAML files under `root`."""
    files: list[pathlib.Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or _is_skipped(path, root):
            continue
        if path.suffix in {".py", ".sh", ".yml", ".yaml"} or (
            path.relative_to(root).as_posix() == "packaging/semma"
        ):
            files.append(path)
    return files


def scan_repo(root: pathlib.Path) -> dict[str, list[tuple[int, str]]]:
    """Scan all target files and return relative POSIX path -> list of violations."""
    results: dict[str, list[tuple[int, str]]] = {}
    for path in iter_target_files(root):
        rel = path.relative_to(root).as_posix()
        if path.suffix == ".py":
            found = scan_python_file(path)
        else:
            found = scan_shell_or_yaml_file(path)
        if found:
            results[rel] = found
    return results


def load_baseline(baseline_path: pathlib.Path) -> dict[str, int]:
    """Load the per-file comment ceiling map from JSON."""
    if not baseline_path.exists():
        return {}
    raw = json.loads(baseline_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{baseline_path} must contain a JSON object mapping file paths to counts")
    return {str(k): int(v) for k, v in raw.items() if int(v) > 0}


def write_baseline(baseline_path: pathlib.Path, counts: dict[str, int]) -> None:
    """Write a sorted, minimal baseline file omitting zero-comment files."""
    trimmed = {k: v for k, v in sorted(counts.items()) if v > 0}
    baseline_path.write_text(json.dumps(trimmed, indent=2) + "\n", encoding="utf-8")


def check_ratchet(
    root: pathlib.Path,
    baseline_path: pathlib.Path,
    *,
    allow_stale: bool = False,
) -> tuple[list[str], dict[str, int]]:
    """Compare current comment counts against the ratchet baseline."""
    violations_by_file = scan_repo(root)
    counts = {rel: len(items) for rel, items in violations_by_file.items()}
    baseline = load_baseline(baseline_path)
    errors: list[str] = []

    for rel, items in sorted(violations_by_file.items()):
        allowed = baseline.get(rel, 0)
        if len(items) > allowed:
            errors.append(
                f"{rel}: found {len(items)} prose comment(s), allowed {allowed} by baseline"
            )
            for lineno, text in items:
                errors.append(f"  {rel}:{lineno}: {text}")

    if not allow_stale:
        for rel, allowed in sorted(baseline.items()):
            actual = counts.get(rel, 0)
            if actual < allowed:
                errors.append(
                    f"{rel}: baseline allows {allowed} comment(s) but file now has {actual}; "
                    f"run `uv run python scripts/lint_comments.py --update-baseline` to ratchet down"
                )

    return errors, counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ratcheted comment-banning linter.")
    parser.add_argument(
        "--root",
        type=pathlib.Path,
        default=pathlib.Path("."),
        help="Repository root to scan (default: current directory).",
    )
    parser.add_argument(
        "--baseline",
        type=pathlib.Path,
        default=None,
        help=f"Path to ratchet baseline JSON (default: <root>/{DEFAULT_BASELINE}).",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Update the baseline file to current comment counts (ratchet down).",
    )
    parser.add_argument(
        "--allow-stale-baseline",
        action="store_true",
        help="Do not fail when files have fewer comments than recorded in the baseline.",
    )
    args = parser.parse_args(argv)

    root = args.root.resolve()
    baseline_path = args.baseline.resolve() if args.baseline else root / DEFAULT_BASELINE

    if args.update_baseline:
        violations_by_file = scan_repo(root)
        counts = {rel: len(items) for rel, items in violations_by_file.items()}
        write_baseline(baseline_path, counts)
        total = sum(counts.values())
        print(f"Updated {baseline_path} ({len(counts)} file(s), {total} comment(s)).")
        return 0

    errors, counts = check_ratchet(root, baseline_path, allow_stale=args.allow_stale_baseline)
    if errors:
        for line in errors:
            print(line, file=sys.stderr)
        return 1

    total = sum(counts.values())
    print(f"Comment lint passed ({total} baselined prose comment(s) remaining).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
