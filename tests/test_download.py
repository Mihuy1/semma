"""Model download: resuming a partial file, restarting when ranges are ignored, verification."""

from __future__ import annotations

import hashlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from semma import embedder as emb

DATA = bytes(range(256)) * 4096  # 1 MiB
NAME = "model.gguf"


class _Server:
    def __init__(self, honour_range: bool = True, cut_first: int = 0) -> None:
        self.ranges: list[str | None] = []
        self.cut = cut_first  # the first answer stops after this many bytes (a clean early close)
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                rng = self.headers.get("Range")
                outer.ranges.append(rng)
                body, status = DATA, 200
                if rng and honour_range:
                    start = int(rng.removeprefix("bytes=").rstrip("-"))
                    if start >= len(DATA):
                        self.send_error(416)
                        return
                    body, status = DATA[start:], 206
                self.send_response(status)
                if status == 206:
                    self.send_header("Content-Range", f"bytes {start}-{len(DATA) - 1}/{len(DATA)}")
                if outer.cut:
                    body, outer.cut = body[: outer.cut], 0
                    self.send_header("Connection", "close")  # no length: the close ends the body
                else:
                    self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args) -> None:
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"


@pytest.fixture
def serve(monkeypatch: pytest.MonkeyPatch):
    servers: list[_Server] = []

    def start(honour_range: bool = True, sha: str | None = None, cut_first: int = 0, size: int = len(DATA)) -> _Server:
        server = _Server(honour_range, cut_first)
        servers.append(server)
        monkeypatch.setattr(emb, "GGUF_BASE_URL", server.url)
        digest = sha or hashlib.sha256(DATA).hexdigest()
        monkeypatch.setattr(emb, "GGUF_SPECS", {"text": (NAME, size, digest)})
        return server

    yield start
    for server in servers:
        server.httpd.shutdown()


def test_fresh_download(tmp_path: Path, serve) -> None:
    server = serve()
    dest = emb.ensure_model_file("text", tmp_path)
    assert dest.read_bytes() == DATA
    assert server.ranges == [None]
    assert not (tmp_path / f"{NAME}.part").exists()


def test_resumes_partial_file(tmp_path: Path, serve) -> None:
    server = serve()
    (tmp_path / f"{NAME}.part").write_bytes(DATA[:300_000])
    dest = emb.ensure_model_file("text", tmp_path)
    assert dest.read_bytes() == DATA
    assert server.ranges == ["bytes=300000-"]


def test_restarts_when_server_ignores_range(tmp_path: Path, serve) -> None:
    server = serve(honour_range=False)
    (tmp_path / f"{NAME}.part").write_bytes(b"x" * 300_000)  # wrong bytes: must not survive
    dest = emb.ensure_model_file("text", tmp_path)
    assert dest.read_bytes() == DATA
    assert server.ranges == ["bytes=300000-"]


def test_complete_partial_file_needs_no_request(tmp_path: Path, serve) -> None:
    server = serve()
    (tmp_path / f"{NAME}.part").write_bytes(DATA)
    assert emb.ensure_model_file("text", tmp_path).read_bytes() == DATA
    assert server.ranges == []


def test_bad_checksum_discards_file(tmp_path: Path, serve) -> None:
    serve(sha="0" * 64)
    with pytest.raises(RuntimeError, match="failed verification"):
        emb.ensure_model_file("text", tmp_path)
    assert not (tmp_path / NAME).exists()
    assert not (tmp_path / f"{NAME}.part").exists()


def test_early_close_keeps_partial_file_for_resume(tmp_path: Path, serve) -> None:
    server = serve(cut_first=400_000)
    with pytest.raises(RuntimeError, match="stopped at 400000"):
        emb.ensure_model_file("text", tmp_path)
    assert (tmp_path / f"{NAME}.part").stat().st_size == 400_000
    assert emb.ensure_model_file("text", tmp_path).read_bytes() == DATA
    assert server.ranges == [None, "bytes=400000-"]


def test_partial_file_longer_than_servers_restarts(tmp_path: Path, serve) -> None:
    # A part left by another revision: longer than the server's file, shorter than expected.
    server = serve(size=len(DATA) + 10)
    (tmp_path / f"{NAME}.part").write_bytes(b"x" * (len(DATA) + 5))
    with pytest.raises(RuntimeError, match="stopped at"):  # the server's file is the short one here
        emb.ensure_model_file("text", tmp_path)
    assert server.ranges == [f"bytes={len(DATA) + 5}-", None]
    assert (tmp_path / f"{NAME}.part").read_bytes() == DATA


# -- progress (what the daemon's Status reports) ------------------------------------------

def _two_files(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(emb, "GGUF_SPECS", {"text": ("a.gguf", 1000, "0" * 64),
                                            "mmproj": ("b.gguf", 3000, "0" * 64)})


def _hold_lock(path: Path):
    import fcntl

    fh = path.open("w")
    fcntl.flock(fh, fcntl.LOCK_EX)
    return fh


def test_model_files_status_counts_partial_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _two_files(monkeypatch)
    assert emb.model_files_status(tmp_path / "missing") == {"ready": False, "done": 0, "total": 4000}
    (tmp_path / "a.gguf.part").write_bytes(b"x" * 400)
    assert emb.model_files_status(tmp_path) == {"ready": False, "done": 400, "total": 4000}
    (tmp_path / "a.gguf").write_bytes(b"x" * 1000)
    (tmp_path / "b.gguf").write_bytes(b"x" * 3000)
    assert emb.model_files_status(tmp_path) == {"ready": True, "done": 4000, "total": 4000}
    # Ready does not depend on the download lock: a complete file is ready with nobody downloading.
    assert emb.download_progress(tmp_path) is None


def test_no_download_progress_without_a_running_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _two_files(monkeypatch)
    assert emb.download_progress(tmp_path / "missing") is None
    assert emb.download_progress(tmp_path) is None
    # An interrupted download (part file and lock file, nobody holds the lock) is not running.
    (tmp_path / "a.gguf.part").write_bytes(b"x" * 400)
    (tmp_path / "a.gguf.lock").touch()
    assert emb.download_progress(tmp_path) is None
    assert not (tmp_path / "b.gguf.lock").exists()  # probing creates no files


def test_download_progress_while_the_lock_is_held(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _two_files(monkeypatch)
    (tmp_path / "a.gguf").write_bytes(b"x" * 1000)  # complete
    (tmp_path / "b.gguf.part").write_bytes(b"x" * 1200)
    with _hold_lock(tmp_path / "b.gguf.lock"):
        assert emb.download_progress(tmp_path) == {
            "file": "b.gguf", "done": 1200, "total": 3000, "all_done": 2200, "all_total": 4000}
        # Started, no bytes yet.
        (tmp_path / "b.gguf.part").unlink()
        assert emb.download_progress(tmp_path)["done"] == 0
    assert emb.download_progress(tmp_path) is None


def test_download_progress_during_a_real_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ensure_model_file against a local server that stalls half-way: progress is the bytes written."""
    half = len(DATA) // 2
    sent_half, go_on = threading.Event(), threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Length", str(len(DATA)))
            self.end_headers()
            self.wfile.write(DATA[:half])
            self.wfile.flush()
            sent_half.set()
            go_on.wait(10)
            self.wfile.write(DATA[half:])

        def log_message(self, *args) -> None:
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    monkeypatch.setattr(emb, "GGUF_BASE_URL", f"http://127.0.0.1:{httpd.server_address[1]}")
    monkeypatch.setattr(emb, "GGUF_SPECS", {"text": (NAME, len(DATA), hashlib.sha256(DATA).hexdigest())})
    monkeypatch.setattr(emb, "DOWNLOAD_CHUNK", 64 * 1024)
    result: list[Path] = []
    t = threading.Thread(target=lambda: result.append(emb.ensure_model_file("text", tmp_path)))
    t.start()
    try:
        assert sent_half.wait(10)
        tick = threading.Event()
        for _ in range(200):  # the client writes what it has read
            p = emb.download_progress(tmp_path)
            if p and p["done"] == half:
                break
            tick.wait(0.01)
        assert p == {"file": NAME, "done": half, "total": len(DATA), "all_done": half, "all_total": len(DATA)}
    finally:
        go_on.set()
        t.join(10)
        httpd.shutdown()
    assert result and result[0].read_bytes() == DATA
    assert emb.download_progress(tmp_path) is None
