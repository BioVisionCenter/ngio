# conftest.py
import io
import os
import socket
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import boto3
import pytest
from aiomoto import mock_aws


class _NoListingHTTPHandler(SimpleHTTPRequestHandler):
    def list_directory(self, path):
        self.send_error(403, "Directory listing not allowed")
        return None

    def send_head(self):
        """Serve `Range: bytes=a-b` requests (206) so a zip can be read in place."""
        rng = self.headers.get("Range")
        path = self.translate_path(self.path)
        if rng is None or not rng.startswith("bytes=") or not os.path.isfile(path):
            return super().send_head()
        size = os.path.getsize(path)
        first, _, last = rng.removeprefix("bytes=").partition("-")
        start = int(first) if first else max(size - int(last), 0)
        end = min(int(last), size - 1) if last and first else size - 1
        if start > end or start >= size:
            self.send_error(416, "Requested range not satisfiable")
            return None
        with open(path, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start + 1)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(len(chunk)))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        return io.BytesIO(chunk)

    def log_message(self, format, *args):
        pass


@pytest.fixture(scope="session")
def moto_s3_server():
    """Mock S3 backend via aiomoto in server mode.

    ``server_mode=True`` starts an in-process Moto server and injects
    ``AWS_ENDPOINT_URL`` (plus dummy ``test`` credentials and the ``us-east-1``
    region), so both boto3 and s3fs discover it automatically — no subprocess,
    no hardcoded port. The Moto backend is shared between the synchronous boto3
    client used here and the aiobotocore backend that s3fs drives under the hood.

    Session-scoped: starting/stopping the server costs ~4s per test. Tests
    isolate from each other by writing to `random_zarr_path()` keys.
    """
    bucket_name = "s3-ci-test-bucket"
    with mock_aws(server_mode=True) as ctx:
        s3 = boto3.client("s3")
        s3.create_bucket(Bucket=bucket_name)
        yield {"endpoint_url": ctx.server_endpoint, "bucket_name": bucket_name}


def _find_free_port(host="127.0.0.1"):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def http_static_server(tmp_path_factory):
    """
    Serve a temporary directory via a non-listable HTTP server.

    Directory listing is disabled (403) to match production HTTP stores that
    do not support listing. Individual file GETs work normally.
    """
    root = tmp_path_factory.mktemp("http_static_root")
    host = "127.0.0.1"
    port = _find_free_port(host)

    server = ThreadingHTTPServer(
        (host, port),
        lambda *a, **kw: _NoListingHTTPHandler(*a, directory=str(root), **kw),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    yield {"url": f"http://{host}:{port}", "root": root}

    server.shutdown()
    server.server_close()
    thread.join(timeout=5)
