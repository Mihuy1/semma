from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from gemmasearch import filetypes as ft
from gemmasearch.config import Config, apply_overrides, load_config
from gemmasearch.priority import Scorer
from gemmasearch.scanner import Scanner
from gemmasearch.store import Store

NOW = 1_800_000_000.0
DAY = 86_400


def make_cfg(home: Path) -> Config:
    cfg = load_config(path=home / "nonexistent.toml", home=home)
    cfg.include_mounts = False
    return cfg


def touch(path: Path, size: int = 5000, age_days: float = 1000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    t = NOW - age_days * DAY
    os.utime(path, (t, t))
    return path


@pytest.mark.parametrize(
    ("name", "kind"),
    [("a.PDF", ft.PDF), ("notes.md", ft.TEXT), ("x.odt", ft.OFFICE), ("p.HEIC", ft.IMAGE),
     ("s.flac", ft.AUDIO), ("v.mkv", ft.VIDEO), ("main.py", None), ("Makefile", None), (".md", None)],
)
def test_classify(name: str, kind: str | None) -> None:
    assert ft.classify(name) == kind


def test_location_beats_type_and_downloads_ranks_lower(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    scorer = Scorer(cfg)
    doc = scorer.score(tmp_path / "Documents/a.pdf", ft.PDF, 50_000, NOW - 2000 * DAY, False, tmp_path, NOW)
    dl = scorer.score(tmp_path / "Downloads/a.pdf", ft.PDF, 50_000, NOW - 2000 * DAY, False, tmp_path, NOW)
    other = scorer.score(Path("/mnt/x/a.pdf"), ft.PDF, 50_000, NOW - 2000 * DAY, False, Path("/mnt/x"), NOW)
    assert doc.parts["loc"] == 40 and dl.parts["loc"] == 15 and other.parts["loc"] == 10
    assert doc.total > dl.total > other.total


def test_modifiers(tmp_path: Path) -> None:
    scorer = Scorer(make_cfg(tmp_path))
    deep = tmp_path / "Documents/a/b/c/d/e/f/g/x.txt"
    s = scorer.score(deep, ft.TEXT, 50_000, NOW - 5 * DAY, True, tmp_path, NOW)
    assert s.parts["project"] == -20
    assert s.parts["depth"] == -8  # 7 levels below Documents, 3 free, -2 each
    assert s.parts["recent"] == 15
    tiny = scorer.score(tmp_path / "Pictures/icon.png", ft.IMAGE, 900, NOW, False, tmp_path, NOW)
    assert tiny.parts["tiny"] == -15
    capped = scorer.score(tmp_path / "Documents/a/b/c/d/e/f/g/h/i/j/k/x.txt", ft.TEXT, 50_000, NOW, False, tmp_path, NOW)
    assert capped.parts["depth"] == -10


def test_score_is_clamped(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    cfg.other_location = -100
    s = Scorer(cfg).score(Path("/elsewhere/a.mp4"), ft.VIDEO, 10, NOW, True, Path("/elsewhere"), NOW)
    assert s.total == 0


def test_scanner_exclusions_and_projects(tmp_path: Path) -> None:
    touch(tmp_path / "Documents/thesis.odt")
    touch(tmp_path / "Documents/.secret/notes.md")
    touch(tmp_path / "Documents/.hidden.md")
    touch(tmp_path / "Documents/proj/package.json")
    touch(tmp_path / "Documents/proj/README.md")
    touch(tmp_path / "Documents/proj/src/main.py")
    touch(tmp_path / "Documents/proj/node_modules/lib/readme.md")
    touch(tmp_path / "Documents/proj/build/out.pdf")
    touch(tmp_path / "Documents/build/manual.pdf")  # "build" outside a project is a real folder
    touch(tmp_path / "Downloads/SteamLibrary/game/intro.mp4")
    touch(tmp_path / "Downloads/movie.mp4", size=300_000)
    touch(tmp_path / "Downloads/setup.exe")  # a lone installer does not make Downloads an app
    touch(tmp_path / "Games/Zomboid/game.exe")
    touch(tmp_path / "Games/Zomboid/fmod.dll")
    touch(tmp_path / "Games/Zomboid/media/sound.ogg")

    cfg = make_cfg(tmp_path)
    scanner = Scanner(cfg)
    found = {Path(f.path).relative_to(tmp_path).as_posix(): f for f in scanner.scan(now=NOW)}

    assert set(found) == {
        "Documents/thesis.odt", "Documents/proj/README.md", "Documents/build/manual.pdf", "Downloads/movie.mp4",
    }
    assert found["Documents/proj/README.md"].in_project
    assert not found["Documents/thesis.odt"].in_project
    assert found["Documents/thesis.odt"].score.total > found["Downloads/movie.mp4"].score.total
    assert scanner.stats.pruned["hidden"] >= 1 and scanner.stats.pruned["node_modules"] == 1
    assert scanner.stats.pruned["project/build"] == 1
    assert scanner.stats.pruned["app-install"] == 1


def test_symlinks_not_followed(tmp_path: Path) -> None:
    touch(tmp_path / "Documents/a.md")
    (tmp_path / "Documents/loop").symlink_to(tmp_path)
    (tmp_path / "Documents/link.md").symlink_to(tmp_path / "Documents/a.md")
    found = [f.path for f in Scanner(make_cfg(tmp_path)).scan(now=NOW)]
    assert found == [str(tmp_path / "Documents/a.md")]


def test_overrides(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    apply_overrides(cfg, {
        "scan": {"exclude_paths": ["~/Documents/private"], "extra_exclude_dir_names": ["Games"]},
        "priority": {"locations": {"~/Downloads": 33}, "type_weights": {"video": 1}},
        "media": {"max_minutes": 10},
    })
    assert cfg.exclude_paths == [tmp_path / "Documents/private"]
    assert "Games" in cfg.exclude_dir_names and "node_modules" in cfg.exclude_dir_names
    assert cfg.locations[tmp_path / "Downloads"] == 33 and cfg.locations[tmp_path / "Documents"] == 40
    assert cfg.type_weights["video"] == 1
    assert cfg.max_media_seconds == 600


def test_store_requeues_changed_and_forgets_deleted(tmp_path: Path) -> None:
    root = tmp_path
    a = touch(root / "Documents/a.md")
    b = touch(root / "Documents/b.md")
    cfg = make_cfg(root)
    store = Store(tmp_path / "db/index.db")
    scanner = Scanner(cfg)

    gen = store.begin_scan()
    store.upsert_scanned(scanner.scan(now=NOW), gen)
    store.finish_scan(gen, [root])
    store.db.execute("UPDATE files SET status = 'indexed'")
    store.db.commit()

    a.write_bytes(b"changed" * 1000)
    b.unlink()
    gen = store.begin_scan()
    store.upsert_scanned(scanner.scan(now=NOW), gen)
    removed = store.finish_scan(gen, [root])

    rows = {Path(r["path"]).name: r["status"] for r in store.db.execute("SELECT path, status FROM files")}
    assert removed == 1
    assert rows == {"a.md": "pending"}


def test_unscanned_root_is_kept(tmp_path: Path) -> None:
    store = Store(":memory:")
    cfg = make_cfg(tmp_path)
    touch(tmp_path / "Documents/a.md")
    gen = store.begin_scan()
    store.upsert_scanned(Scanner(cfg).scan(now=NOW), gen)
    gen = store.begin_scan()
    assert store.finish_scan(gen, [Path("/mnt/unplugged")]) == 0
    assert store.db.execute("SELECT count(*) FROM files").fetchone()[0] == 1


def test_user_dirs_are_respected(tmp_path: Path) -> None:
    (tmp_path / ".config").mkdir()
    (tmp_path / ".config/user-dirs.dirs").write_text('XDG_DOCUMENTS_DIR="$HOME/Asiakirjat"\n')
    os.environ.pop("XDG_CONFIG_HOME", None)
    cfg = make_cfg(tmp_path)
    assert cfg.locations[tmp_path / "Asiakirjat"] == 40
