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
from semma.viewer import Spot, build_argv, media_mime, parse_exec, spot_for, viewer_name

PDF = "/home/u/Docs/paper one.pdf"
VIDEO = "/m/a b.mp4"
AUDIO = "/m/song one.flac"
SPOT = Spot(7)
TIME_SPOT = Spot(seconds=120)


def _pdf_result(position: float | None = 7.0, modality: str = "text",
                snippet: str = "Quantum walrus sings at midnight on the shore…") -> SearchResult:
    return SearchResult(
        path=PDF, kind=ft.PDF, score=3.0, similarity=0.7, modality=modality, locator="p. 7",
        snippet=snippet, position=position, mtime=1.0, matched=["vector"],
    )


def _media_result(kind: str = ft.VIDEO, position: float | None = 12.5) -> SearchResult:
    return SearchResult(
        path=VIDEO if kind == ft.VIDEO else AUDIO,
        kind=kind, score=3.0, similarity=0.7, modality=kind, locator="0:12",
        snippet="", position=position, mtime=1.0, matched=["vector"],
    )


def test_spot_for_only_pdf_hits_with_a_page() -> None:
    assert spot_for(_pdf_result()) == Spot(7)
    assert not Spot(7).is_time
    assert spot_for(_pdf_result(modality="image")) == Spot(7)
    assert spot_for(_pdf_result(position=None)) is None
    assert spot_for(_pdf_result(position=0.0)) is None
    assert spot_for(replace(_pdf_result(), kind=ft.OFFICE)) is None


def test_spot_for_audio_and_video() -> None:
    for kind in (ft.AUDIO, ft.VIDEO):
        spot = spot_for(_media_result(kind, 12.5))
        assert spot == Spot(seconds=12.5)
        assert spot is not None and spot.is_time and spot.page is None
        assert spot_for(_media_result(kind, 0.0)) is None
        assert spot_for(_media_result(kind, -1.0)) is None
        assert spot_for(_media_result(kind, None)) is None
    assert spot_for(replace(_media_result(ft.VIDEO, 10.0), kind=ft.IMAGE)) is None


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
    uri = "file:///home/u/Docs/paper%20one.pdf#page=7"
    for exe in ("/usr/bin/google-chrome-stable", "/usr/bin/brave-browser-stable", "chromium-browser",
                "/usr/bin/microsoft-edge-stable", "vivaldi-stable", "opera"):
        assert build_argv([exe, "%U"], PDF, SPOT) == [exe, uri]
    flatpak = ["/usr/bin/flatpak", "run", "--branch=stable", "--command=/app/bin/chrome", "--file-forwarding",
               "com.google.Chrome", "@@u", "%U", "@@"]
    assert build_argv(flatpak, PDF, SPOT) == [*flatpak[:6], "@@u", uri, "@@"]
    assert viewer_name(["flatpak", "run", "com.brave.Browser", "%U"]) == "brave"


def test_build_argv_flatpak_options_go_before_forwarding_block() -> None:
    exec_argv = ["/usr/bin/flatpak", "run", "--branch=stable", "--command=okular", "--file-forwarding",
                 "org.kde.okular", "@@u", "%U", "@@"]
    assert build_argv(exec_argv, PDF, SPOT) == [
        "/usr/bin/flatpak", "run", "--branch=stable", "--command=okular", "--file-forwarding",
        "org.kde.okular", "--page", "7", "@@u", PDF, "@@"]


def test_build_argv_field_codes() -> None:
    assert build_argv(["okular"], PDF, Spot(1)) == ["okular", "--page", "1", PDF]
    assert build_argv(["okular", "%i", "--caption", "%c", "%k", "%U", "%F"], PDF, Spot(1)) == [
        "okular", "--caption", "--page", "1", PDF]
    assert build_argv(["okular", "--x=100%%", "%U"], PDF, Spot(1)) == ["okular", "--x=100%", "--page", "1", PDF]
    assert build_argv(["libreoffice", "--draw", "%U"], PDF, SPOT) is None
    assert build_argv(["okular", "--file=%f"], PDF, SPOT) is None


def test_build_argv_time_players_and_end_of_options() -> None:
    assert build_argv(["mpv", "--player-operation-mode=pseudo-gui", "--", "%U"], VIDEO, TIME_SPOT) == [
        "mpv", "--player-operation-mode=pseudo-gui", "--start=120", "--", VIDEO]
    assert build_argv(["/usr/bin/vlc", "--started-from-file", "%U"], VIDEO, TIME_SPOT) == [
        "/usr/bin/vlc", "--started-from-file", "--start-time=120", VIDEO]
    assert build_argv(["cvlc", "%U"], AUDIO, Spot(seconds=12.5)) == ["cvlc", "--start-time=12.5", AUDIO]
    assert build_argv(["celluloid", "%U"], VIDEO, Spot(seconds=45)) == ["celluloid", "--mpv-start=45", VIDEO]
    assert build_argv(["gnome-mpv", "%U"], VIDEO, Spot(seconds=45.25)) == ["gnome-mpv", "--mpv-start=45.25", VIDEO]
    assert build_argv(["firefox", "%u"], VIDEO, TIME_SPOT) == ["firefox", "file:///m/a%20b.mp4#t=120"]
    assert build_argv(["/usr/bin/google-chrome-stable", "%U"], AUDIO, Spot(seconds=12.5)) == [
        "/usr/bin/google-chrome-stable", "file:///m/song%20one.flac#t=12.5"]
    for exe in ("dragon", "elisa", "haruna", "totem", "showtime"):
        assert build_argv([exe, "%U"], VIDEO, TIME_SPOT) is None


def test_build_argv_time_flatpak_and_markers() -> None:
    mpv_flatpak = ["/usr/bin/flatpak", "run", "--branch=stable", "--file-forwarding",
                   "io.mpv.Mpv", "--", "@@u", "%U", "@@"]
    assert build_argv(mpv_flatpak, VIDEO, TIME_SPOT) == [
        "/usr/bin/flatpak", "run", "--branch=stable", "--file-forwarding",
        "io.mpv.Mpv", "--start=120", "--", "@@u", VIDEO, "@@"]
    mpv_flatpak_reversed = ["/usr/bin/flatpak", "run", "io.mpv.Mpv", "@@u", "--", "%U", "@@"]
    assert build_argv(mpv_flatpak_reversed, VIDEO, TIME_SPOT) == [
        "/usr/bin/flatpak", "run", "io.mpv.Mpv", "--start=120", "@@u", "--", VIDEO, "@@"]
    vlc_flatpak = ["/usr/bin/flatpak", "run", "--branch=stable", "--file-forwarding",
                   "org.videolan.VLC", "@@u", "%U", "@@"]
    assert build_argv(vlc_flatpak, VIDEO, TIME_SPOT) == [
        "/usr/bin/flatpak", "run", "--branch=stable", "--file-forwarding",
        "org.videolan.VLC", "--start-time=120", "@@u", VIDEO, "@@"]
    assert viewer_name(["flatpak", "run", "io.github.celluloid_player.Celluloid", "%U"]) == "celluloid"


def test_build_argv_seconds_formatting() -> None:
    for secs, expected in ((120.0, "120"), (10.0, "10"), (12.5, "12.5"), (12.500, "12.5"), (1.23456, "1.235")):
        assert build_argv(["mpv", "%U"], VIDEO, Spot(seconds=secs)) == ["mpv", f"--start={expected}", VIDEO]


def test_build_argv_spot_type_mismatch() -> None:
    assert build_argv(["okular", "%U"], VIDEO, TIME_SPOT) is None
    assert build_argv(["mpv", "%U"], PDF, SPOT) is None
    assert build_argv(["vlc", "%U"], PDF, SPOT) is None
    assert build_argv(["okular", "%U"], PDF, Spot()) is None


def test_media_mime(monkeypatch: pytest.MonkeyPatch) -> None:
    assert media_mime("/m/video.mp4") == "video/mp4"
    assert media_mime("/m/song.mp3") == "audio/mpeg"
    for ext in ("mkv", "WEBM"):
        mime = media_mime(f"/m/clip.{ext}")
        assert mime is not None and mime.startswith("video/")
    for ext in ("opus", "m4a", "flac"):
        mime = media_mime(f"/m/track.{ext}")
        assert mime is not None and mime.startswith("audio/")
    assert media_mime("/m/doc.pdf") is None
    assert media_mime("/m/notes.txt") is None
    assert media_mime("/m/noext") is None
    monkeypatch.setattr(viewer.mimetypes, "guess_type", lambda _p: ("application/ogg", None))
    assert media_mime("/m/voice.opus") == "audio/ogg"
    assert media_mime("/m/song.mp3") is None
    monkeypatch.setattr(viewer.mimetypes, "guess_type", lambda _p: (None, None))
    assert media_mime("/m/movie.mkv") == "video/x-matroska"
    assert media_mime("/m/clip.WEBM") == "video/webm"
    assert media_mime("/m/voice.opus") == "audio/ogg"
    assert media_mime("/m/track.m4a") == "audio/mp4"
    assert media_mime("/m/lossless.flac") == "audio/flac"


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


def test_viewer_argv_time_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _desktop(tmp_path, "mpv.desktop", "[Desktop Entry]\nExec=mpv --player-operation-mode=pseudo-gui -- %U\n")
    _desktop(tmp_path, "elisa.desktop", "[Desktop Entry]\nExec=elisa %U\n")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.setenv("XDG_DATA_DIRS", str(tmp_path / "none"))
    seen_mimes: list[str] = []
    default: list[str | None] = ["mpv.desktop"]

    async def fake_default(mime: str = viewer.PDF_MIME) -> str | None:
        seen_mimes.append(mime)
        return default[0]

    monkeypatch.setattr(viewer, "default_desktop_id", fake_default)
    assert asyncio.run(viewer.viewer_argv(VIDEO, TIME_SPOT)) == [
        "mpv", "--player-operation-mode=pseudo-gui", "--start=120", "--", VIDEO]
    assert seen_mimes == ["video/mp4"]
    for missing in ("elisa.desktop", "gone.desktop", None):
        default[0] = missing
        assert asyncio.run(viewer.viewer_argv(VIDEO, TIME_SPOT)) is None
    seen_mimes.clear()
    assert asyncio.run(viewer.viewer_argv("/m/unknown.xyz", TIME_SPOT)) is None
    assert seen_mimes == []


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


def _fake_mpris(monkeypatch: pytest.MonkeyPatch, events: list) -> None:
    from semma import mpris

    async def snapshot() -> set[str]:
        events.append("snapshot")
        return {"org.mpris.MediaPlayer2.old"}

    monkeypatch.setattr(mpris, "snapshot", snapshot)
    monkeypatch.setattr(mpris, "seek_in_background",
                        lambda path, seconds, before: events.append(("seek", path, seconds, before)))


def test_launchers_seek_over_mpris_when_the_player_has_no_start_option(monkeypatch: pytest.MonkeyPatch) -> None:
    media, spot = "/m/pod cast.mp3", Spot(seconds=720.0)
    seek = ("seek", media, 720.0, {"org.mpris.MediaPlayer2.old"})

    events: list = []
    _fake_mpris(monkeypatch, events)
    monkeypatch.setattr(kr.shutil, "which", lambda n: "/usr/bin/kde-open" if n == "kde-open" else None)
    launcher = Launcher()
    monkeypatch.setattr(launcher, "_spawn_detached", lambda argv, activation_token=None: events.append(argv) or True)
    _fake_viewer_argv(monkeypatch, kr, None)
    asyncio.run(launcher.open_file(media, activation_token="tok", spot=spot))
    assert events == ["snapshot", ["/usr/bin/kde-open", media], seek]

    events.clear()
    _fake_viewer_argv(monkeypatch, kr, ["mpv", "--start=720", media])
    asyncio.run(launcher.open_file(media, spot=spot))
    assert events == [["mpv", "--start=720", media]]

    events.clear()
    monkeypatch.setattr(gs.shutil, "which", lambda n: "/usr/bin/gio" if n == "gio" else None)
    glauncher = GnomeLauncher()

    async def spawn(argv: list[str]) -> bool:
        events.append(argv)
        return True

    monkeypatch.setattr(glauncher, "spawn", spawn)
    _fake_viewer_argv(monkeypatch, gs, None)
    asyncio.run(glauncher.open_file(media, spot=spot))
    assert events == ["snapshot", ["/usr/bin/gio", "open", media], seek]

    events.clear()
    asyncio.run(glauncher.open_file(PDF, spot=SPOT))
    assert events == [["/usr/bin/gio", "open", PDF]]


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
        await krunner.run(match_id, "")
        assert rec.calls == [(PDF, Spot(7))]

        rec.calls.clear()
        gnome = GnomeSearchService(_no_search, launcher=rec)  # type: ignore[arg-type]
        await gnome.activate_result(gnome._remember(hit))
        doc = tmp_path / "old.pdf"
        doc.write_bytes(b"%PDF")
        await gnome.activate_result(str(doc))
        assert rec.calls == [(PDF, Spot(7)), (str(doc), None)]

    asyncio.run(check())
