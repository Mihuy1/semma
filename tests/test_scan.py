from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from semma import filetypes as ft
from semma.config import DEFAULT_MAX_BYTES, Config, apply_overrides, check_unknown_keys, load_config
from semma.priority import Scorer
from semma.scanner import Scanner
from semma.store import ChunkRecord, Store

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
     ("s.flac", ft.AUDIO), ("v.mkv", ft.VIDEO), ("main.py", None), ("Makefile", None), (".md", None),
     ("CMakeLists.txt", None), ("requirements-dev.txt", None), ("LICENSE.md", None), ("LICENSE.electron.txt", None), ("notes.TXT", ft.TEXT), ("notice-board.md", ft.TEXT)],
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
    assert s.parts["depth"] == -8
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
    touch(tmp_path / "Documents/build/manual.pdf")
    touch(tmp_path / "Downloads/SteamLibrary/game/intro.mp4")
    touch(tmp_path / "Downloads/movie.mp4", size=300_000)
    touch(tmp_path / "Downloads/setup.exe")
    touch(tmp_path / "Games/Zomboid/game.exe")
    touch(tmp_path / "Games/Zomboid/fmod.dll")
    touch(tmp_path / "Games/Zomboid/steam_api64.dll")
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


def scan_paths(cfg: Config, **kw) -> set[str]:
    return {Path(f.path).relative_to(cfg.home).as_posix() for f in Scanner(cfg).scan(now=NOW, **kw)}


def test_non_utf8_name_is_skipped_not_fatal(tmp_path: Path) -> None:
    touch(tmp_path / "Documents/good.md")
    bad = os.fsencode(tmp_path / "Documents") + b"/caf\xe9.md"
    with open(bad, "wb") as fh:
        fh.write(b"x" * 5000)
    cfg = make_cfg(tmp_path)
    scanner = Scanner(cfg)
    store = Store(":memory:")
    gen = store.begin_scan()
    assert store.upsert_scanned(scanner.scan(now=NOW), gen) == 1
    assert scanner.stats.bad_names == 1


def test_unreadable_dir_keeps_its_index(tmp_path: Path) -> None:
    touch(tmp_path / "Documents/locked/a.md")
    touch(tmp_path / "Documents/b.md")
    cfg = make_cfg(tmp_path)
    store = Store(":memory:")
    scanner = Scanner(cfg)
    gen = store.begin_scan()
    store.upsert_scanned(scanner.scan(now=NOW), gen)
    store.finish_scan(gen, scanner.stats.roots, scanner.stats.failed_dirs)

    locked = tmp_path / "Documents/locked"
    locked.chmod(0)
    try:
        gen = store.begin_scan()
        store.upsert_scanned(scanner.scan(now=NOW), gen)
        removed = store.finish_scan(gen, scanner.stats.roots, scanner.stats.failed_dirs)
    finally:
        locked.chmod(0o755)
    if os.geteuid() == 0:
        pytest.skip("root can read anything")
    assert removed == 0
    assert store.db.execute("SELECT count(*) FROM files").fetchone()[0] == 2


def test_empty_root_is_treated_as_unmounted(tmp_path: Path) -> None:
    drive = tmp_path / "drive"
    touch(drive / "a.md")
    cfg = make_cfg(tmp_path)
    store = Store(":memory:")
    scanner = Scanner(cfg)
    gen = store.begin_scan()
    store.upsert_scanned(scanner.scan([drive], now=NOW), gen)
    (drive / "a.md").unlink()
    gen = store.begin_scan()
    store.upsert_scanned(scanner.scan([drive], now=NOW), gen)
    assert store.finish_scan(gen, scanner.stats.roots, scanner.stats.failed_dirs) == 0


def test_downloads_with_installer_and_dlls_is_not_pruned(tmp_path: Path) -> None:
    touch(tmp_path / "Downloads/setup.exe")
    touch(tmp_path / "Downloads/a.dll")
    touch(tmp_path / "Downloads/b.dll")
    touch(tmp_path / "Downloads/paper.pdf")
    touch(tmp_path / "Downloads/tool/tool.exe")
    touch(tmp_path / "Downloads/tool/one.dll")
    touch(tmp_path / "Downloads/tool/notes.md")
    assert scan_paths(make_cfg(tmp_path)) == {"Downloads/paper.pdf", "Downloads/tool/notes.md"}


def test_marker_in_root_or_anchor_is_not_a_project(tmp_path: Path) -> None:
    touch(tmp_path / ".git/HEAD")
    touch(tmp_path / "Documents/package.json")
    touch(tmp_path / "Documents/build/report.pdf")
    touch(tmp_path / "notes.md")
    files = {Path(f.path).name: f for f in Scanner(make_cfg(tmp_path)).scan(now=NOW)}
    assert set(files) == {"report.pdf", "notes.md"}
    assert not any(f.in_project for f in files.values())


def test_system_dirs_only_pruned_at_root_and_case_insensitive(tmp_path: Path) -> None:
    touch(tmp_path / "Windows/system.txt")
    touch(tmp_path / "$Recycle.Bin/x.md")
    touch(tmp_path / "Documents/Windows/tips.md")
    touch(tmp_path / "Documents/NODE_MODULES/x.md")
    assert scan_paths(make_cfg(tmp_path)) == {"Documents/Windows/tips.md"}


def test_virtualenv_pruned_anywhere(tmp_path: Path) -> None:
    touch(tmp_path / "myenv/pyvenv.cfg")
    touch(tmp_path / "myenv/lib/readme.md")
    touch(tmp_path / "Documents/a.md")
    assert scan_paths(make_cfg(tmp_path)) == {"Documents/a.md"}


def test_nested_roots_are_walked_once(tmp_path: Path) -> None:
    touch(tmp_path / "Documents/a.md")
    touch(tmp_path / "b.md")
    cfg = make_cfg(tmp_path)
    cfg.roots = [tmp_path, tmp_path / "Documents"]
    files = [f.path for f in Scanner(cfg).scan(now=NOW)]
    assert sorted(files) == sorted({str(tmp_path / "Documents/a.md"), str(tmp_path / "b.md")})


def test_symlinked_home_is_canonicalised(tmp_path: Path) -> None:
    real = tmp_path / "var/home/user"
    touch(real / "Documents/a.md")
    link = tmp_path / "home"
    link.symlink_to(tmp_path / "var/home")
    cfg = load_config(path=tmp_path / "none.toml", home=link / "user")
    cfg.include_mounts = False
    files = list(Scanner(cfg).scan(now=NOW))
    assert [f.path for f in files] == [str(real / "Documents/a.md")]
    assert files[0].score.parts["loc"] == 40


def test_scan_kinds_and_exclude_extensions_config(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = make_cfg(tmp_path)
    assert cfg.kinds == frozenset(ft.KINDS)
    assert cfg.exclude_extensions == frozenset()
    assert check_unknown_keys({"scan": {"kinds": ["text"], "exclude_extensions": ["gif"]}}) == []

    apply_overrides(cfg, {
        "scan": {
            "kinds": ["text", "pdf", "office", "image"],
            "exclude_extensions": ["GIF", ".wav", " .Mp4 "],
        },
    })
    assert cfg.kinds == frozenset({"text", "pdf", "office", "image"})
    assert cfg.exclude_extensions == frozenset({"gif", "wav", "mp4"})
    assert capsys.readouterr().err == ""

    cfg_bad = make_cfg(tmp_path)
    apply_overrides(cfg_bad, {
        "scan": {
            "kinds": ["text", "bogus_kind", 123],
            "exclude_extensions": [".JPG", 456, "."],
        },
    })
    err = capsys.readouterr().err
    assert "bogus_kind" in err
    assert "123" in err
    assert "456" in err
    assert cfg_bad.kinds == frozenset({"text"})
    assert cfg_bad.exclude_extensions == frozenset({"jpg"})

    cfg_nonlist = make_cfg(tmp_path)
    apply_overrides(cfg_nonlist, {
        "scan": {
            "kinds": "text",
            "exclude_extensions": ".gif",
        },
    })
    err = capsys.readouterr().err
    assert "scan.kinds" in err
    assert "scan.exclude_extensions" in err
    assert cfg_nonlist.kinds == frozenset(ft.KINDS)
    assert cfg_nonlist.exclude_extensions == frozenset()


def test_scanner_skips_disabled_kinds_and_excluded_extensions(tmp_path: Path) -> None:
    touch(tmp_path / "Documents/notes.md")
    touch(tmp_path / "Documents/readme.txt")
    touch(tmp_path / "Documents/paper.pdf")
    touch(tmp_path / "Pictures/photo.jpg")
    touch(tmp_path / "Pictures/anim.GIF")
    touch(tmp_path / "Music/song.flac")
    touch(tmp_path / "Videos/clip.mp4")

    cfg = make_cfg(tmp_path)
    apply_overrides(cfg, {
        "scan": {
            "kinds": ["text", "pdf", "image"],
            "exclude_extensions": ["GIF", ".txt"],
        },
    })
    assert scan_paths(cfg) == {
        "Documents/notes.md",
        "Documents/paper.pdf",
        "Pictures/photo.jpg",
    }
    assert ft.classify("anim.GIF") == ft.IMAGE
    assert ft.classify("song.flac") == ft.AUDIO


def test_disallowed_files_disappear_on_next_full_scan(tmp_path: Path) -> None:
    import numpy as np

    touch(tmp_path / "Documents/notes.md")
    touch(tmp_path / "Documents/paper.pdf")
    touch(tmp_path / "Pictures/photo.jpg")
    touch(tmp_path / "Pictures/anim.gif")
    touch(tmp_path / "Music/song.flac")

    cfg = make_cfg(tmp_path)
    store = Store(tmp_path / "db/index.db")

    scanner = Scanner(cfg)
    gen = store.begin_scan()
    store.upsert_scanned(scanner.scan(now=NOW), gen)
    assert store.finish_scan(gen, scanner.stats.roots, scanner.stats.failed_dirs) == 0

    vec = np.zeros(768, dtype=np.float32)
    vec[0] = 1.0
    for row in store.db.execute("SELECT id, size, mtime, kind FROM files").fetchall():
        store.replace_chunks(
            row["id"], row["size"], row["mtime"], row["kind"],
            [ChunkRecord("text", "", "s", None, "hello", vec)],
        )
    assert store.db.execute("SELECT count(*) FROM files WHERE status = 'indexed'").fetchone()[0] == 5
    assert store.db.execute("SELECT count(*) FROM chunks").fetchone()[0] == 5

    apply_overrides(cfg, {
        "scan": {
            "kinds": ["text", "pdf", "office", "image", "video"],
            "exclude_extensions": [".GIF"],
        },
    })
    scanner2 = Scanner(cfg)
    gen2 = store.begin_scan()
    store.upsert_scanned(scanner2.scan(now=NOW), gen2)
    removed = store.finish_scan(gen2, scanner2.stats.roots, scanner2.stats.failed_dirs)

    assert removed == 2
    remaining = {
        Path(r["path"]).name: r["status"]
        for r in store.db.execute("SELECT path, status FROM files")
    }
    assert remaining == {
        "notes.md": "indexed",
        "paper.pdf": "indexed",
        "photo.jpg": "indexed",
    }
    assert store.db.execute("SELECT count(*) FROM chunks").fetchone()[0] == 3
    assert store.db.execute("SELECT count(*) FROM vec_chunks").fetchone()[0] == 3
    store.close()


def test_scan_max_file_mb_config(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = make_cfg(tmp_path)
    assert cfg.max_bytes == DEFAULT_MAX_BYTES
    assert check_unknown_keys({"scan": {"max_file_mb": {"video": 20480, "audio": 1024}}}) == []

    apply_overrides(cfg, {
        "scan": {"max_file_mb": {"video": 20480, "audio": 1024, "text": 0.5}},
    })
    assert cfg.max_bytes == {
        **DEFAULT_MAX_BYTES,
        "video": 20480 * 2**20,
        "audio": 1024 * 2**20,
        "text": round(0.5 * 2**20),
    }
    assert capsys.readouterr().err == ""

    cfg_bad = make_cfg(tmp_path)
    apply_overrides(cfg_bad, {
        "scan": {
            "max_file_mb": {
                "audio": 512,
                "bogus": 10,
                "video": 0,
                "pdf": -5,
                "office": True,
                "image": "large",
            },
        },
    })
    err = capsys.readouterr().err
    for bad in ("bogus", "scan.max_file_mb.video", "scan.max_file_mb.pdf",
                "scan.max_file_mb.office", "scan.max_file_mb.image"):
        assert bad in err
    assert cfg_bad.max_bytes == {**DEFAULT_MAX_BYTES, "audio": 512 * 2**20}

    cfg_nondict = make_cfg(tmp_path)
    apply_overrides(cfg_nondict, {"scan": {"max_file_mb": 1024}})
    assert "scan.max_file_mb" in capsys.readouterr().err
    assert cfg_nondict.max_bytes == DEFAULT_MAX_BYTES


def test_ocr_config(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = make_cfg(tmp_path)
    assert cfg.ocr_enabled is False and cfg.ocr_languages is None
    assert check_unknown_keys({"ocr": {"enabled": True, "languages": ["eng"]}}) == []
    assert check_unknown_keys({"ocr": {"langs": ["eng"]}}) == ["ocr.langs"]

    apply_overrides(cfg, {"ocr": {"enabled": True, "languages": ["fin", "ENG", "fin", "chi_sim"]}})
    assert cfg.ocr_enabled is True
    assert cfg.ocr_languages == ("fin", "eng", "chi_sim")
    assert capsys.readouterr().err == ""

    apply_overrides(cfg, {"ocr": {"languages": ["deu", "fra", 7, "English", "swe", "ita"]}})
    err = capsys.readouterr().err
    assert "7" in err and "'English'" in err and "more than 3" in err
    assert cfg.ocr_languages == ("deu", "fra", "swe")

    apply_overrides(cfg, {"ocr": {"enabled": "yes", "languages": "eng"}})
    err = capsys.readouterr().err
    assert "ocr.enabled" in err and "ocr.languages" in err
    assert cfg.ocr_enabled is True and cfg.ocr_languages == ("deu", "fra", "swe")
    apply_overrides(cfg, {"ocr": {"languages": []}})
    assert cfg.ocr_languages is None
    apply_overrides(cfg, {"ocr": "on"})
    assert "[ocr] must be a table" in capsys.readouterr().err
