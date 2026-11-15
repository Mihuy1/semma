"""First-run setup: the completion flag, and the daemon waiting to scan and index."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fakes import FakeEmbedder
from test_daemon import LifecycleEmbedder, counts, make_cfg, make_daemon, until_idle, wait_until, write_docs
from test_reload import base_settings, reload_and_settle, start_cfg

from semma.config import load_config
from semma.onboarding import onboarding_done
from semma.store import Store

DOCS = {
    "cats.md": "the cat sleeps on the warm sofa all afternoon",
    "rockets.md": "rocket engines burn liquid oxygen and kerosene during launch",
}


def test_missing_flag_follows_whether_the_index_was_used(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    cfg = load_config(home / "no.toml", home=home)
    cfg.db_path = tmp_path / "index.db"
    assert cfg.onboarding_complete is None
    assert onboarding_done(cfg) is False

    store = Store(cfg.db_path)
    store.close()
    assert onboarding_done(cfg) is False  # an empty database is created at daemon start

    store = Store(cfg.db_path)
    store.set_meta("last_scan_at", "10")
    store.close()
    assert onboarding_done(cfg) is True

    cfg.db_path = tmp_path / "files.db"
    store = Store(cfg.db_path)
    with store.write() as (db, _delta):
        db.execute(
            "INSERT INTO files (path, kind, size, mtime, in_project, priority, status, last_seen) "
            "VALUES ('/tmp/a.md', 'text', 1, 0, 0, 0, 'indexed', 1)")
    store.close()
    fresh = load_config(home / "no.toml", home=home)
    fresh.db_path = cfg.db_path
    assert fresh.onboarding_complete is None
    assert onboarding_done(fresh) is True


def test_explicit_flag_wins_over_an_existing_index(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "no.toml", home=tmp_path)
    cfg.db_path = tmp_path / "index.db"
    store = Store(cfg.db_path)
    store.set_meta("last_scan_at", "10")
    store.close()
    cfg.onboarding_complete = True
    assert onboarding_done(cfg) is True
    cfg.onboarding_complete = False
    assert onboarding_done(cfg) is False


def test_invalid_flag_is_ignored(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "config.toml"
    path.write_text('onboarding_complete = "yes"\n', encoding="utf-8")
    cfg = load_config(path, home=tmp_path)
    assert cfg.onboarding_complete is None
    assert "onboarding_complete" in capsys.readouterr().err


def test_setup_holds_back_scan_search_and_the_model(tmp_path: Path) -> None:
    write_docs(tmp_path, DOCS)
    cfg = make_cfg(tmp_path)
    cfg.onboarding_complete = False
    emb = LifecycleEmbedder()
    calls: list[str] = []

    def download(config: object) -> None:
        calls.append("download")
        raise RuntimeError("offline")

    async def main() -> None:
        d = make_daemon(cfg, embedder=emb, model_download=download)
        d.start()
        try:
            await wait_until(lambda: calls and d._download_error == "offline", what="download failed")
            await asyncio.sleep(0.2)
            assert counts(cfg.db_path) == {}
            assert d._setup_done is False
            assert await d.search("rocket launch") == []
            assert emb.load_calls == 0 and emb.query_load_calls == 0
            st = await d.status()
            assert st["download_error"] == "offline"
            assert st["state"] != "scanning"
        finally:
            await d.stop()

    asyncio.run(main())


def test_download_retries_until_the_files_are_fetched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from semma import daemon as daemon_mod

    monkeypatch.setattr(daemon_mod, "DOWNLOAD_BACKOFF_S", (0.01, 0.05))
    cfg = make_cfg(tmp_path)
    cfg.onboarding_complete = False
    calls: list[int] = []

    def download(config: object) -> None:
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("offline")

    async def main() -> None:
        d = make_daemon(cfg, model_download=download)
        d.start()
        try:
            await wait_until(lambda: len(calls) >= 2 and d._download_error == "", what="download retried")
        finally:
            await d.stop()

    asyncio.run(main())


def test_indexing_waits_for_the_model_files(tmp_path: Path) -> None:
    write_docs(tmp_path, DOCS)

    class Gated(FakeEmbedder):
        def __init__(self) -> None:
            super().__init__()
            self.ready = False

        def model_files_status(self) -> dict[str, object]:
            return {"ready": self.ready, "done": 0, "total": 10}

    async def main() -> None:
        cfg = make_cfg(tmp_path)
        emb = Gated()
        d = make_daemon(cfg, embedder=emb)
        d.start()
        try:
            await wait_until(lambda: counts(cfg.db_path).get("pending", 0) > 0, what="scan queued files")
            await asyncio.sleep(0.2)
            assert "indexed" not in counts(cfg.db_path)
            emb.ready = True
            d._wake()
            await until_idle(d)
        finally:
            await d.stop()

    asyncio.run(main())


def test_finishing_setup_starts_a_scan_and_running_it_again_keeps_the_index(tmp_path: Path) -> None:
    write_docs(tmp_path, DOCS)

    async def main() -> None:
        settings = base_settings(tmp_path)
        settings["onboarding_complete"] = False
        cfg = start_cfg(tmp_path, settings)
        d = make_daemon(cfg)
        d.start()
        try:
            await asyncio.sleep(0.2)
            assert counts(cfg.db_path) == {}
            settings["onboarding_complete"] = True
            assert await reload_and_settle(d, tmp_path, settings) == []
            indexed = counts(cfg.db_path).get("indexed", 0)
            assert indexed == 2

            settings["onboarding_complete"] = False
            assert await reload_and_settle(d, tmp_path, settings, rescan=False) == []
            await wait_until(lambda: d._setup_done is False, what="setup paused")
            assert counts(cfg.db_path).get("indexed") == indexed

            settings["onboarding_complete"] = True
            assert await reload_and_settle(d, tmp_path, settings) == []
            assert counts(cfg.db_path).get("indexed") == indexed
        finally:
            await d.stop()

    asyncio.run(main())


def test_semma_setup_finishes_setup_without_the_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from semma import cli
    from semma import daemon as daemon_mod

    home = tmp_path / "home"
    path = home / ".config" / "semma" / "config.toml"
    path.parent.mkdir(parents=True)
    path.write_text(f'db_path = "{tmp_path / "index.db"}"\n[scan]\nroots = ["~/Documents"]\n', encoding="utf-8")
    reloads: list[bool] = []
    monkeypatch.setattr(daemon_mod, "daemon_reload_config", lambda **kw: reloads.append(True) or [])

    assert not onboarding_done(load_config(path, home))
    assert cli.main(["--config", str(path), "setup"]) == 0
    cfg = load_config(path, home)
    assert cfg.onboarding_complete is True and onboarding_done(cfg)
    assert [str(r) for r in cfg.roots] == [str(home / "Documents")]  # the other settings are kept
    assert reloads == [True]
    assert "Setup finished" in capsys.readouterr().out

    assert cli.main(["--config", str(path), "setup"]) == 0  # again: nothing to write
    assert "already finished" in capsys.readouterr().out


def test_status_reports_pending_setup_and_rescan_waits(tmp_path: Path) -> None:
    from semma import cli

    async def main() -> None:
        cfg = make_cfg(tmp_path)
        cfg.onboarding_complete = False
        d = make_daemon(cfg, embedder=FakeEmbedder())
        d.start()
        try:
            st = await d.status()
            assert st["setup_pending"] is True
            assert any("semma setup" in line for line in cli.format_daemon_status(st, cfg.home))
            d.rescan()
            assert d._scan_full is False
        finally:
            await d.stop()

    asyncio.run(main())
