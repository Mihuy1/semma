from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from semma import filetypes as ft
from semma import gnome_search as gs
from semma import krunner as kr
from semma import viewer
from semma.gnome_search import GnomeLauncher, GnomeSearchService
from semma.krunner import KRunnerService, Launcher
from semma.search import SearchResult
from semma.viewer import Spot, build_argv, parse_exec, spot_for, viewer_name

PDF = "/home/u/Docs/paper one.pdf"
SPOT = Spot(7)


def _pdf_result(position: float | None = 7.0, modality: str = "text",
                snippet: str = "Quantum walrus sings at midnight on the shore…") -> SearchResult:
    return SearchResult(
        path=PDF, kind=ft.PDF, score=3.0, similarity=0.7, modality=modality, locator="p. 7",
        snippet=snippet, position=position, mtime=1.0, matched=["vector"],
    )


def test_spot_for_only_pdf_hits_with_a_page() -> None:
    assert spot_for(_pdf_result()) == Spot(7)
    assert spot_for(_pdf_result(modality="image")) == Spot(7)  # a scanned page
    assert spot_for(_pdf_result(position=None)) is None
    assert spot_for(_pdf_result(position=0.0)) is None
    assert spot_for(replace(_pdf_result(), kind=ft.OFFICE)) is None


def test_viewer_name_native_env_and_flatpak() -> None:
    assert viewer_name(["okular", "%U"]) == "okular"
    assert viewer_name(["/usr/bin/evince", "%U"]) == "evince"
    assert viewer_name(["env", "GDK_BACKEND=x11", "/usr/bin/zathura", "%f"]) == "zathura"
    assert viewer_name(["/usr/bin/flatpak", "run", "--branch=stable", "--command=papers",
                        "org.gnome.Papers", "@@u", "%U", "@@"]) == "papers"
    assert viewer_name(["flatpak", "run", "--arch=x86_64", "org.kde.okular", "%U"]) == "okular"
    assert viewer_name(["flatpak", "run", "com.example.Other", "%U"]) is None
    assert viewer_name(["env"]) is None and viewer_name([]) is None


def test_build_argv_per_viewer() -> None:
    assert build_argv(["okular", "%U"], PDF, SPOT) == ["okular", "--page", "7", PDF]
    assert build_argv(["evince", "%U"], PDF, SPOT) == ["evince", "--page-index=7", PDF]
    assert build_argv(["papers", "%U"], PDF, SPOT) == ["papers", "--page-index=7", PDF]
    assert build_argv(["zathura", "%f"], PDF, SPOT) == ["zathura", "--page=7", PDF]
    assert build_argv(["firefox", "%u"], PDF, SPOT) == ["firefox", "file:///home/u/Docs/paper%20one.pdf#page=7"]


def test_build_argv_flatpak_options_go_before_forwarding_block() -> None:
    exec_argv = ["/usr/bin/flatpak", "run", "--branch=stable", "--command=okular", "--file-forwarding",
                 "org.kde.okular", "@@u", "%U", "@@"]
    assert build_argv(exec_argv, PDF, SPOT) == [
        "/usr/bin/flatpak", "run", "--branch=stable", "--command=okular", "--file-forwarding",
        "org.kde.okular", "--page", "7", "@@u", PDF, "@@"]


def test_build_argv_field_codes() -> None:
    # No field code: options and file appended. %i/%c/%k dropped, %% unescaped, one file only.
    assert build_argv(["okular"], PDF, Spot(1)) == ["okular", "--page", "1", PDF]
    assert build_argv(["okular", "%i", "--caption", "%c", "%k", "%U", "%F"], PDF, Spot(1)) == [
        "okular", "--caption", "--page", "1", PDF]
    assert build_argv(["okular", "--x=100%%", "%U"], PDF, Spot(1)) == ["okular", "--x=100%", "--page", "1", PDF]
    # Unknown viewer or a field code inside an argument: give up (caller opens plainly).
    assert build_argv(["libreoffice", "--draw", "%U"], PDF, SPOT) is None
    assert build_argv(["okular", "--file=%f"], PDF, SPOT) is None


def test_parse_exec_escapes_and_quotes() -> None:
    assert parse_exec('"/opt/My Viewer/okular" %U') == ["/opt/My Viewer/okular", "%U"]
    assert parse_exec(r'sh -c "echo \\"hi\\"" %f') == ["sh", "-c", 'echo "hi"', "%f"]
    assert parse_exec(r"okular\s--x %U") == ["okular", "--x", "%U"]
    assert parse_exec('okular "unterminated') is None
    assert parse_exec("") is None


def _desktop(dir_: Path, name: str, body: str) -> None:
    apps = dir_ / "applications"
    apps.mkdir(parents=True, exist_ok=True)
    (apps / name).write_text(body)


def test_find_and_read_desktop_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home, sys_a, sys_b = tmp_path / "home", tmp_path / "a", tmp_path / "b"
    _desktop(sys_b, "org.kde.okular.desktop",
             "[Desktop Entry]\nName=Okular\nExec=okular %U\n[Desktop Action new]\nExec=okular --new\n")
    _desktop(home, "org.kde.okular.desktop",
             "[Desktop Action x]\nExec=nope\n[Desktop Entry]\nExec[fi]=wrong\nExec = /usr/bin/okular %U\n")
    monkeypatch.setenv("XDG_DATA_HOME", str(home))
    monkeypatch.setenv("XDG_DATA_DIRS", f"{sys_a}:{sys_b}")
    found = viewer.find_desktop_file("org.kde.okular.desktop")
    assert found == home / "applications" / "org.kde.okular.desktop"
    assert viewer.read_exec(found) == ["/usr/bin/okular", "%U"]
    assert viewer.read_exec(sys_b / "applications" / "org.kde.okular.desktop") == ["okular", "%U"]
    assert viewer.find_desktop_file("missing.desktop") is None
    assert viewer.find_desktop_file("../x.desktop") is None
    assert viewer.read_exec(tmp_path / "nothing.desktop") is None


def test_viewer_argv_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _desktop(tmp_path, "org.gnome.Papers.desktop", "[Desktop Entry]\nExec=papers %U\n")
    _desktop(tmp_path, "draw.desktop", "[Desktop Entry]\nExec=libreoffice --draw %U\n")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.setenv("XDG_DATA_DIRS", str(tmp_path / "none"))
    default: list[str | None] = ["org.gnome.Papers.desktop"]

    async def fake_default(mime: str = viewer.PDF_MIME) -> str | None:
        assert mime == "application/pdf"
        return default[0]

    monkeypatch.setattr(viewer, "default_desktop_id", fake_default)
    assert asyncio.run(viewer.viewer_argv(PDF, SPOT)) == ["papers", "--page-index=7", PDF]
    for missing in ("draw.desktop", "gone.desktop", None):
        default[0] = missing
        assert asyncio.run(viewer.viewer_argv(PDF, SPOT)) is None


def test_default_desktop_id_runs_xdg_mime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "xdg-mime"
    script.write_text('#!/bin/sh\n[ "$1 $2 $3" = "query default application/pdf" ] && echo org.kde.okular.desktop\n')
    script.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))
    assert asyncio.run(viewer.default_desktop_id()) == "org.kde.okular.desktop"
    script.write_text("#!/bin/sh\nexit 3\n")
    assert asyncio.run(viewer.default_desktop_id()) is None
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert asyncio.run(viewer.default_desktop_id()) is None


def test_default_desktop_id_times_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "xdg-mime"
    script.write_text("#!/bin/sh\nexec sleep 10\n")
    script.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.setattr(viewer, "MIME_QUERY_TIMEOUT_SECONDS", 0.1)
    assert asyncio.run(viewer.default_desktop_id()) is None


def _fake_viewer_argv(monkeypatch: pytest.MonkeyPatch, module, argv: list[str] | None) -> list[tuple[str, Spot]]:
    calls: list[tuple[str, Spot]] = []

    async def fake(path: str, spot: Spot) -> list[str] | None:
        calls.append((path, spot))
        return argv

    monkeypatch.setattr(module, "viewer_argv", fake)
    return calls


def test_krunner_launcher_opens_viewer_at_spot(monkeypatch: pytest.MonkeyPatch) -> None:
    spawned: list[tuple[list[str], str | None]] = []
    monkeypatch.setattr(kr.shutil, "which", lambda n: "/usr/bin/kde-open" if n == "kde-open" else None)
    launcher = Launcher()
    monkeypatch.setattr(launcher, "_spawn_detached", lambda argv, activation_token=None:
                        spawned.append((argv, activation_token)) or True)

    calls = _fake_viewer_argv(monkeypatch, kr, ["okular", "--page", "7", PDF])
    asyncio.run(launcher.open_file(PDF, activation_token="tok", spot=SPOT))
    assert calls == [(PDF, SPOT)] and spawned == [(["okular", "--page", "7", PDF], "tok")]

    # No spot: the viewer is not looked up. Unknown viewer: plain desktop opener.
    spawned.clear()
    asyncio.run(launcher.open_file(PDF, activation_token="tok"))
    assert len(calls) == 1 and spawned == [(["/usr/bin/kde-open", PDF], "tok")]
    spawned.clear()
    _fake_viewer_argv(monkeypatch, kr, None)
    asyncio.run(launcher.open_file(PDF, spot=SPOT))
    assert spawned == [(["/usr/bin/kde-open", PDF], None)]


def test_krunner_launcher_falls_back_when_viewer_fails_to_spawn(monkeypatch: pytest.MonkeyPatch) -> None:
    spawned: list[list[str]] = []
    monkeypatch.setattr(kr.shutil, "which", lambda n: "/usr/bin/kde-open" if n == "kde-open" else None)
    launcher = Launcher()

    def spawn(argv: list[str], activation_token: str | None = None) -> bool:
        spawned.append(argv)
        return argv[0] != "okular"

    monkeypatch.setattr(launcher, "_spawn_detached", spawn)
    _fake_viewer_argv(monkeypatch, kr, ["okular", PDF])
    asyncio.run(launcher.open_file(PDF, spot=SPOT))
    assert spawned == [["okular", PDF], ["/usr/bin/kde-open", PDF]]


def test_gnome_launcher_opens_viewer_at_spot(monkeypatch: pytest.MonkeyPatch) -> None:
    spawned: list[list[str]] = []
    monkeypatch.setattr(gs.shutil, "which", lambda n: "/usr/bin/gio" if n == "gio" else None)
    launcher = GnomeLauncher()

    async def spawn(argv: list[str]) -> bool:
        spawned.append(argv)
        return True

    monkeypatch.setattr(launcher, "spawn", spawn)
    _fake_viewer_argv(monkeypatch, gs, ["papers", "--page-index=7", PDF])
    asyncio.run(launcher.open_file(PDF, spot=SPOT))
    assert spawned == [["papers", "--page-index=7", PDF]]

    spawned.clear()
    _fake_viewer_argv(monkeypatch, gs, None)
    asyncio.run(launcher.open_file(PDF, spot=SPOT))
    assert spawned == [["/usr/bin/gio", "open", PDF]]


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Spot | None]] = []

    async def open_file(self, path: str, activation_token: str | None = None, spot: Spot | None = None) -> None:
        self.calls.append((path, spot))


async def _no_search(q: str) -> list[SearchResult]:
    return []


def test_services_pass_the_hit_spot(tmp_path: Path) -> None:
    async def check() -> None:
        hit = _pdf_result()
        rec = _Recorder()
        krunner = KRunnerService(_no_search, launcher=rec)  # type: ignore[arg-type]
        match_id = krunner._remember(hit)
        await type(krunner).__dict__["Run"].__wrapped__(krunner, match_id, "")
        assert rec.calls == [(PDF, Spot(7))]

        rec.calls.clear()
        gnome = GnomeSearchService(_no_search, launcher=rec)  # type: ignore[arg-type]
        activate = type(gnome).__dict__["ActivateResult"].__wrapped__
        await activate(gnome, gnome._remember(hit), [], 0)
        # An id no longer remembered opens without a page.
        doc = tmp_path / "old.pdf"
        doc.write_bytes(b"%PDF")
        await activate(gnome, str(doc), [], 0)
        assert rec.calls == [(PDF, Spot(7)), (str(doc), None)]

    asyncio.run(check())
