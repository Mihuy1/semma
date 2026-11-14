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
    def __init__(self, honour_range: bool = True) -> None:
        self.ranges: list[str | None] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                rng = self.headers.get("Range")
                outer.ranges.append(rng)
                body, status = DATA, 200
                if rng and honour_range:
                    body, status = DATA[int(rng.removeprefix("bytes=").rstrip("-")):], 206
                self.send_response(status)
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

    def start(honour_range: bool = True, sha: str | None = None) -> _Server:
        server = _Server(honour_range)
        servers.append(server)
        monkeypatch.setattr(emb, "GGUF_BASE_URL", server.url)
        digest = sha or hashlib.sha256(DATA).hexdigest()
        monkeypatch.setattr(emb, "GGUF_SPECS", {"text": (NAME, len(DATA), digest)})
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
