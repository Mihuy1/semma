"""The window's Qt-free parts: the Status page's text, formatting, the Folders section's changes."""

from __future__ import annotations

from pathlib import Path

import pytest
from gui_fakes import status

from semma.config_write import ConfigWriteError, read_raw, update_config
from semma.gui import model as m
from semma.gui.settings import folders_model as fm
from semma.gui.settings.context import Problem, SettingsContext, merge_changes

HOME = Path("/home/u")


@pytest.mark.parametrize(("n", "text"), [
    (0, "0 B"), (999, "999 B"), (1000, "1.0 kB"), (1500, "1.5 kB"), (12_345_678, "12 MB"),
    (864_676_480, "865 MB"), (2_500_000_000, "2.5 GB"),
])
def test_format_bytes(n: int, text: str) -> None:
    assert m.format_bytes(n) == text


def test_format_bytes_of_uses_the_unit_of_the_total() -> None:
    assert m.format_bytes_of(412_000_000, 864_676_480) == "412 of 865 MB"
    assert m.format_bytes_of(5_000_000, 2_500_000_000) == "0.0 of 2.5 GB"
    assert m.format_bytes_of(900, 1000) == "0.9 of 1.0 kB"
    assert m.format_bytes_of(2000, 1000) == "1.0 of 1.0 kB"


def test_counts_and_plurals() -> None:
    assert m.format_count(1204) == "1,204"
    assert m.plural(1, "file") == "1 file"
    assert m.plural(0, "file") == "0 files"
    assert m.plural(12_400, "file") == "12,400 files"


@pytest.mark.parametrize(("ago", "text"), [
    (0, "Just now"), (59, "Just now"), (60, "1 minute ago"), (12 * 60 + 30, "12 minutes ago"),
    (3600, "1 hour ago"), (5 * 3600, "5 hours ago"), (86400, "1 day ago"), (3 * 86400, "3 days ago"),
    (-30, "Just now"),
])
def test_format_ago(ago: float, text: str) -> None:
    assert m.format_ago(1_000_000.0 - ago, now=1_000_000.0) == text


def test_format_ago_never() -> None:
    assert m.format_ago(0, now=1_000_000.0) == "Not yet"


@pytest.mark.parametrize(("seconds", "text"), [
    (10, "less than a minute"), (60, "about 1 minute"), (34 * 60, "about 34 minutes"),
    (2 * 3600, "about 2 hours"), (3 * 86400, "about 3 days"),
])
def test_format_duration(seconds: float, text: str) -> None:
    assert m.format_duration(seconds) == text


def test_short_path() -> None:
    assert m.short_path("/home/u", HOME) == "~"
    assert m.short_path("/home/u/Documents/a.md", HOME) == "~/Documents/a.md"
    assert m.short_path("/home/user2/a.md", HOME) == "/home/user2/a.md"
    assert m.short_path("/mnt/data", HOME) == "/mnt/data"
    assert m.short_path("", HOME) == ""


def test_elide_middle() -> None:
    assert m.elide_middle("short", 10) == "short"
    text = m.elide_middle("~/" + "a" * 100 + "/end", 20)
    assert len(text) == 20 and text.startswith("~/aa") and text.endswith("/end") and "…" in text


def test_headline_downloading_uses_all_model_files() -> None:
    h = m.headline(status(state="downloading", download_file="model.gguf", download_done=100_000_000,
                          download_total=300_000_000, download_all_done=412_000_000,
                          download_all_total=864_676_480), HOME)
    assert h.text == "Downloading the model: 412 of 865 MB"
    assert h.progress == (412_000_000, 864_676_480) and h.detail == "model.gguf"


def test_headline_downloading_one_file_or_unknown_size() -> None:
    h = m.headline(status(state="downloading", download_file="a.gguf", download_done=5_000_000,
                          download_total=10_000_000), HOME)
    assert h.text == "Downloading the model: 5.0 of 10 MB" and h.progress == (5_000_000, 10_000_000)
    h = m.headline(status(state="downloading", download_file="a.gguf"), HOME)
    assert h.text == "Downloading the model" and h.busy and h.progress is None


def test_headline_downloading_wins_over_other_states() -> None:
    h = m.headline(status(state="indexing", download_file="a.gguf", download_all_total=10, download_all_done=1), HOME)
    assert h.text.startswith("Downloading the model")


def test_headline_scanning() -> None:
    h = m.headline(status(state="scanning", scan_dir="/home/u/Documents", scan_files=12_400, scan_dirs=842,
                          scan_matched=3210), HOME)
    assert h.text == "Scanning ~/Documents (12,400 files seen)"
    assert h.busy and h.progress is None and h.detail == "842 folders listed, 3,210 to index"
    assert m.headline(status(state="scanning"), HOME).text == "Scanning folders"


def test_headline_loading() -> None:
    h = m.headline(status(state="loading"), HOME)
    assert h.text == "Loading the model" and h.busy


def test_headline_indexing() -> None:
    h = m.headline(status(state="indexing", indexed=1204, pending=8666, run_rate=4.2,
                          current_file="/home/u/Documents/report.pdf"), HOME)
    assert h.text == "Indexing: 1,204 of 9,870 files"
    assert h.progress == (1204, 9870)
    assert h.detail == "~/Documents/report.pdf"
    assert h.secondary == "4.2 files per second, about 34 minutes left"


def test_headline_indexing_without_a_rate_yet() -> None:
    h = m.headline(status(state="indexing", indexed=0, pending=1), HOME)
    assert h.text == "Indexing: 0 of 1 file" and h.secondary == ""


@pytest.mark.parametrize(("reason", "text"), [
    ("paused by request", "Paused by you"),
    ("", "Paused by you"),
    ("the GPU is in use by another application (over 30% SM)",
     "Paused: the GPU is in use by another application (over 30% SM)"),
])
def test_headline_paused(reason: str, text: str) -> None:
    h = m.headline(status(state="paused", paused=True, pause_reason=reason, indexed=10, pending=5), HOME)
    assert h.text == text and h.progress == (10, 15) and h.detail == "5 files waiting to be indexed"


def test_headline_paused_by_user_before_the_worker_noticed() -> None:
    h = m.headline(status(state="indexing", paused=True, indexed=10, pending=5), HOME)
    assert h.text == "Paused by you"


def test_headline_idle() -> None:
    assert m.headline(status(indexed=9870), HOME).text == "Up to date: 9,870 files indexed"
    assert m.headline(status(indexed=9870), HOME).progress is None
    assert m.headline(status(), HOME).text == "Nothing indexed yet"
    h = m.headline(status(indexed=10, pending=3), HOME)
    assert h.text == "Waiting to index 3 files" and h.progress == (10, 13)


def test_headline_unknown_state() -> None:
    assert m.headline(status(state="defragmenting"), HOME).text == "Defragmenting"


def test_download_footer() -> None:
    assert m.download_footer(None).offer_start
    assert m.download_footer(None).text == "The Semma service is not running."
    view = m.download_footer(status(download_file="a.gguf", download_all_done=412_000_000,
                                    download_all_total=865_000_000))
    assert view.text == "Downloading the model: 412 of 865 MB"
    assert view.progress == (412_000_000, 865_000_000)
    ready = m.download_footer(status(model_files_ready=True, model_bytes_done=100, model_bytes_total=100))
    assert ready.text == "Model ready" and ready.progress == (100, 100)
    failed = m.download_footer(status(download_error="offline", model_bytes_done=10, model_bytes_total=100))
    assert failed.text == "Download failed: offline Retrying." and failed.progress == (10, 100)
    waiting = m.download_footer(status(model_bytes_done=0, model_bytes_total=4000))
    assert waiting.text.startswith("Waiting to download") and waiting.progress == (0, 4000)


def test_status_view_of_an_older_daemon_without_the_new_keys() -> None:
    old = {"state": "indexing", "paused": False, "pause_reason": "", "pending": 3, "indexed": 7, "failed": 1,
           "chunks": 20, "current_file": "/home/u/a.md", "last_scan_at": 940.0, "uptime": 5.0,
           "watched_dirs": 4, "model_loaded": True}
    v = m.status_view(old, HOME, now=1000.0)
    assert v.headline.text == "Indexing: 7 of 10 files" and v.headline.secondary == ""
    assert [(r.name, r.indexed, r.pending, r.failed) for r in v.kinds] == [("All files", 7, 3, 1)]
    assert v.failed == 1 and v.last_scan == "1 minute ago"
    assert m.status_view({}, HOME).headline.text == "Nothing indexed yet"


def test_kind_rows() -> None:
    st = status(indexed_by_kind={"text": 5, "pdf": 0, "image": 2}, pending_by_kind={"pdf": 1, "weird": 4},
                failed_by_kind={"video": 1})
    rows = m.kind_rows(st)
    assert [(r.name, r.indexed, r.pending, r.failed) for r in rows] == [
        ("Text", 5, 0, 0), ("PDF", 0, 1, 0), ("Images", 2, 0, 0), ("Video", 0, 0, 1), ("Weird", 0, 4, 0)]
    assert {m.KIND_NAMES[k] for k in m.KIND_NAMES} == {"Text", "PDF", "Office documents", "Images", "Audio", "Video"}


def test_status_view_paused_flag() -> None:
    assert m.status_view(status(paused=True), HOME).paused
    assert m.status_view(status(state="paused"), HOME).paused
    assert not m.status_view(status(state="indexing"), HOME).paused


@pytest.fixture
def home(tmp_path: Path) -> Path:
    h = tmp_path / "home"
    for d in ("Documents", "Pictures", "Documents/tmp"):
        (h / d).mkdir(parents=True)
    return h


def ctx_for(home: Path, text: str | None = None) -> SettingsContext:
    path = home / ".config" / "semma" / "config.toml"
    if text is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return SettingsContext.load(path, home)


def test_folders_load_defaults(home: Path) -> None:
    ctx = ctx_for(home)
    assert fm.load(ctx) == fm.FoldersValues(["~"], [], True, True)
    assert not ctx.is_set("scan", "roots")


def test_folders_load_from_config(home: Path) -> None:
    ctx = ctx_for(home, '[scan]\nroots = ["~/Documents", "/srv/data"]\nexclude_paths = ["$HOME/Documents/tmp"]\n'
                        'skip_hidden = false\n')
    v = fm.load(ctx)
    assert v.roots == ["~/Documents", "/srv/data"] and v.exclude_paths == ["~/Documents/tmp"]
    assert v.include_mounts and not v.skip_hidden
    assert ctx.is_set("scan", "roots") and not ctx.is_set("scan", "include_mounts")


def test_folders_changes_only_what_changed(home: Path) -> None:
    saved = fm.load(ctx_for(home))
    assert fm.changes(saved, saved.copy()) == {}
    edited = saved.copy()
    edited.roots = ["~/Documents", "~/Pictures"]
    assert fm.changes(saved, edited) == {"scan": {"roots": ["~/Documents", "~/Pictures"]}}
    edited.include_mounts = False
    assert fm.changes(saved, edited) == {"scan": {"roots": ["~/Documents", "~/Pictures"], "include_mounts": False}}


def test_folders_changes_back_to_the_default_remove_the_key(home: Path) -> None:
    saved = fm.load(ctx_for(home, '[scan]\nroots = ["~/Documents"]\nskip_hidden = false\n'
                                  'exclude_paths = ["~/Documents/tmp"]\n'))
    edited = saved.copy()
    edited.roots = ["~"]
    edited.skip_hidden = True
    edited.exclude_paths = []
    assert fm.changes(saved, edited) == {"scan": {"roots": None, "skip_hidden": None, "exclude_paths": None}}


def test_folders_changes_written_by_update_config(home: Path) -> None:
    ctx = ctx_for(home, '# mine\n[scan]\nroots = ["~/Documents"]  # docs only\nskip_app_installs = false\n')
    saved = fm.load(ctx)
    edited = saved.copy()
    edited.roots, _ = fm.add_folder(edited.roots, str(home / "Pictures"), ctx)
    edited.skip_hidden = False
    update_config(ctx.config_path, fm.changes(saved, edited))
    raw = read_raw(ctx.config_path)
    assert raw == {"scan": {"roots": ["~/Documents", "~/Pictures"], "skip_app_installs": False, "skip_hidden": False}}
    assert "# mine" in ctx.config_path.read_text()
    again = fm.load(ctx_for(home))
    assert again == edited


def test_add_folder(home: Path) -> None:
    ctx = ctx_for(home)
    folders, msg = fm.add_folder(["~"], str(home / "Documents"), ctx)
    assert folders == ["~", "~/Documents"] and msg is None
    same, msg = fm.add_folder(folders, str(home / "Documents") + "/", ctx)
    assert same == folders and msg == "~/Documents is in the list already."
    (home / "link").symlink_to(home / "Pictures")
    folders, _ = fm.add_folder(folders, str(home / "link"), ctx)
    assert folders[-1] == "~/Pictures"
    folders, msg = fm.add_folder(folders, "/srv/x", ctx)
    assert folders[-1] == "/srv/x" and msg is None


def test_folders_problems(home: Path) -> None:
    ctx = ctx_for(home)
    ok = fm.FoldersValues(["~/Documents"], ["~/Documents/tmp"], False, True)
    assert fm.problems(ok, ctx) == []
    v = fm.FoldersValues(["~/Documents", "~/Documents"], ["~/Pictures", "~/Documents"], False, True)
    texts = [p.text for p in fm.problems(v, ctx)]
    assert texts == ["~/Documents is in the indexed folders twice.",
                     "~/Pictures is not inside an indexed folder, so skipping it changes nothing.",
                     "~/Documents is both indexed and skipped."]
    assert not any(p.blocking for p in fm.problems(v, ctx))
    assert [p.text for p in fm.problems(fm.FoldersValues([], [], False, True), ctx)] == [
        "Nothing is indexed: add a folder, or index external drives."]
    assert fm.problems(fm.FoldersValues(["~"], ["/run/media/u/disk/junk"], True, True), ctx) == []


def test_settings_context_invalid_config(home: Path) -> None:
    with pytest.raises(ConfigWriteError):
        ctx_for(home, "[scan\nroots = 1\n")
    with pytest.raises(ConfigWriteError):
        ctx_for(home, '[scan]\nroots = "~/not-a-list"\nskip_hidden = 3\nexclude_paths = 5\n')


def test_merge_changes() -> None:
    assert merge_changes({"scan": {"a": 1}}, {}, {"scan": {"b": None}, "media": {}}, {"daemon": {"c": 2}}) == {
        "scan": {"a": 1, "b": None}, "daemon": {"c": 2}}
    assert Problem("x") == Problem("x", blocking=False)


def test_cli_and_qt_free_modules_do_not_import_pyside6() -> None:
    import subprocess
    import sys

    code = ("import sys, semma.cli, semma.gui, semma.gui.model, semma.gui.client, semma.gui.single, "
            "semma.gui.settings.context, semma.gui.settings.folders_model, "
            "semma.gui.settings.filetypes_model, semma.gui.settings.priorities_model, "
            "semma.gui.settings.media_model, semma.gui.settings.performance_model, "
            "semma.gui.settings.screenshots_model; "
            "semma.cli.build_parser(); "
            "bad = [m for m in sys.modules if m.startswith(('PySide6', 'shiboken6'))]; "
            "assert not bad, bad")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_gui_without_pyside6_prints_one_line(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import builtins

    from semma import gui

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("PySide6"):
            raise ImportError("No module named 'PySide6'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert gui.main() == 1
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "PySide6" in err


def test_headline_says_when_setup_is_not_finished() -> None:
    h = m.headline({"state": "idle", "setup_pending": True, "indexed": 5}, "/home/u")
    assert h.text == "Setup is not finished"
    assert "semma setup" in h.detail and "Run setup again" in h.detail
