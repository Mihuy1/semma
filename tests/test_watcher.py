from __future__ import annotations

import errno
import logging
import os
import threading
import time
from pathlib import Path

import pytest
from inotify_simple import Event, flags

from semma.config import load_config
from semma.scanner import Scanner
from semma.watcher import DirWatcher


def _wait_for(predicate, timeout: float = 2.0, interval: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def test_scanner_walked_dirs_records_only_unpruned_dirs(tmp_path: Path) -> None:
    (tmp_path / "Documents/sub").mkdir(parents=True)
    (tmp_path / "Documents/sub/note.md").write_text("hello")
    (tmp_path / "Documents/.hidden").mkdir(parents=True)
    (tmp_path / "Documents/.hidden/secret.md").write_text("secret")
    (tmp_path / "Documents/node_modules/pkg").mkdir(parents=True)
    (tmp_path / "Documents/venv").mkdir(parents=True)
    (tmp_path / "Documents/venv/pyvenv.cfg").write_text("home = /usr/bin\n")

    cfg = load_config(path=tmp_path / "none.toml", home=tmp_path)
    cfg.include_mounts = False
    scanner = Scanner(cfg)
    list(scanner.scan())

    walked = set(scanner.stats.walked_dirs)
    assert str(tmp_path) in walked
    assert str(tmp_path / "Documents") in walked
    assert str(tmp_path / "Documents/sub") in walked
    assert str(tmp_path / "Documents/.hidden") not in walked
    assert str(tmp_path / "Documents/node_modules") not in walked
    assert str(tmp_path / "Documents/venv") not in walked


def test_watcher_file_and_subdir_operations(tmp_path: Path) -> None:
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()

    batches: list[set[str]] = []
    lock = threading.Lock()

    def on_change(dirs: set[str]) -> None:
        with lock:
            batches.append(set(dirs))

    watcher = DirWatcher(on_change, debounce=0.15, max_delay=2.0)
    watcher.set_dirs([str(dir_a), str(dir_b)])
    watcher.start()
    try:
        f1 = dir_a / "doc.md"
        f1.write_text("initial")
        assert _wait_for(lambda: len(batches) >= 1)
        with lock:
            assert batches[-1] == {str(dir_a)}
            batches.clear()

        f1.write_text("updated")
        f2 = dir_a / "move_me.txt"
        f2.write_text("moving")
        f2.rename(dir_b / "moved.txt")
        f1.unlink()
        assert _wait_for(lambda: len(batches) >= 1)
        with lock:
            combined = set().union(*batches)
            assert combined == {str(dir_a), str(dir_b)}
            batches.clear()

        sub_new = dir_a / "new_sub"
        sub_new.mkdir()
        assert _wait_for(lambda: len(batches) >= 1)
        with lock:
            combined = set().union(*batches)
            assert str(dir_a) in combined
            assert str(sub_new) in combined
            batches.clear()

        sub_moved = dir_b / "moved_sub"
        sub_new.rename(sub_moved)
        assert _wait_for(lambda: len(batches) >= 1)
        with lock:
            combined = set().union(*batches)
            assert str(dir_a) in combined
            assert str(dir_b) in combined
            assert str(sub_moved) in combined
    finally:
        watcher.stop()


def test_watcher_ignores_temp_files_but_not_dirs(tmp_path: Path) -> None:
    batches: list[set[str]] = []
    watcher = DirWatcher(lambda d: batches.append(set(d)), debounce=0.15, max_delay=1.0)
    watcher.set_dirs([str(tmp_path)])
    watcher.start()
    try:
        for name in (".hidden.txt", "backup~", "dl.part", "dl.crdownload", "x.tmp", "edit.swp"):
            (tmp_path / name).write_text("temp")
        time.sleep(0.3)
        assert batches == []

        (tmp_path / ".newdir").mkdir()
        assert _wait_for(lambda: len(batches) >= 1)
        assert str(tmp_path / ".newdir") in batches[-1]
    finally:
        watcher.stop()


def test_set_dirs_incremental_add_remove_and_skip_missing(tmp_path: Path) -> None:
    d1 = tmp_path / "d1"
    d2 = tmp_path / "d2"
    d3 = tmp_path / "d3"
    d1.mkdir()
    d2.mkdir()
    d3.mkdir()

    batches: list[set[str]] = []
    watcher = DirWatcher(lambda d: batches.append(set(d)), debounce=0.15, max_delay=1.0)
    watcher.set_dirs([str(d1), str(d2), str(tmp_path / "does_not_exist")])
    wd_d2_before = watcher._path_to_wd[str(d2)]
    watcher.start()
    try:
        watcher.set_dirs([str(d2), str(d3)])
        assert watcher._path_to_wd[str(d2)] == wd_d2_before
        assert str(d1) not in watcher.watched_dirs
        assert str(d3) in watcher.watched_dirs

        (d1 / "ignored.txt").write_text("no watch")
        (d3 / "watched.txt").write_text("watched")
        assert _wait_for(lambda: len(batches) >= 1)
        assert batches[-1] == {str(d3)}
    finally:
        watcher.stop()


def test_overflow_triggers_wildcard_rescan(tmp_path: Path) -> None:
    batches: list[set[str]] = []
    watcher = DirWatcher(lambda d: batches.append(set(d)), debounce=5.0, max_delay=30.0)
    watcher.set_dirs([str(tmp_path)])
    try:
        overflow_event = Event(wd=-1, mask=flags.Q_OVERFLOW, cookie=0, name="")
        watcher._handle_event(overflow_event)
        assert batches == [{"*"}]
    finally:
        watcher.stop()


def test_max_delay_fires_under_continuous_churn(tmp_path: Path) -> None:
    batches: list[tuple[float, set[str]]] = []
    watcher = DirWatcher(
        lambda d: batches.append((time.monotonic(), set(d))),
        debounce=0.35,
        max_delay=0.45,
    )
    watcher.set_dirs([str(tmp_path)])
    watcher.start()
    try:
        started = time.monotonic()
        i = 0
        while time.monotonic() - started < 0.65:
            (tmp_path / f"file_{i}.txt").write_text(str(i))
            i += 1
            time.sleep(0.08)

        assert len(batches) >= 1
        first_fire_elapsed = batches[0][0] - started
        assert first_fire_elapsed < 0.70
        assert batches[0][1] == {str(tmp_path)}
    finally:
        watcher.stop()


def test_stop_is_prompt_and_closes_fd(tmp_path: Path) -> None:
    watcher = DirWatcher(lambda d: None, debounce=10.0, max_delay=60.0)
    watcher.set_dirs([str(tmp_path)])
    assert watcher._inotify is not None
    fd = watcher._inotify.fileno()
    watcher.start()
    time.sleep(0.05)

    t0 = time.monotonic()
    watcher.stop()
    elapsed = time.monotonic() - t0
    assert elapsed < 0.3
    assert watcher._inotify is None
    with pytest.raises(OSError):
        os.fstat(fd)


def test_on_change_exception_does_not_kill_thread(tmp_path: Path) -> None:
    calls: list[set[str]] = []

    def flaky_callback(dirs: set[str]) -> None:
        calls.append(set(dirs))
        if len(calls) == 1:
            raise RuntimeError("boom in callback")

    watcher = DirWatcher(flaky_callback, debounce=0.12, max_delay=1.0)
    watcher.set_dirs([str(tmp_path)])
    watcher.start()
    try:
        (tmp_path / "first.txt").write_text("1")
        assert _wait_for(lambda: len(calls) >= 1)

        assert watcher._thread is not None and watcher._thread.is_alive()

        (tmp_path / "second.txt").write_text("2")
        assert _wait_for(lambda: len(calls) >= 2)
        assert calls[1] == {str(tmp_path)}
    finally:
        watcher.stop()


def test_enospc_logs_warning_once_and_keeps_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    d1 = tmp_path / "d1"
    d2 = tmp_path / "d2"
    d3 = tmp_path / "d3"
    for d in (d1, d2, d3):
        d.mkdir()

    watcher = DirWatcher(lambda d: None, debounce=0.2)
    try:
        assert watcher._inotify is not None
        real_add = watcher._inotify.add_watch
        calls = 0

        def fake_add_watch(path: str, mask: int) -> int:
            nonlocal calls
            calls += 1
            if calls >= 2:
                raise OSError(errno.ENOSPC, "No space left on device")
            return real_add(path, mask)

        monkeypatch.setattr(watcher._inotify, "add_watch", fake_add_watch)
        with caplog.at_level(logging.WARNING):
            watcher.set_dirs([str(d1), str(d2), str(d3)])
            watcher.set_dirs([str(d1), str(d2), str(d3)])

        warnings = [r for r in caplog.records if "ENOSPC" in r.message]
        assert len(warnings) == 1
        assert len(watcher.watched_dirs) == 1
    finally:
        watcher.stop()
