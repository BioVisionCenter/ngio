# conftest.py
import os
import socket
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import boto3
import pytest
from aiomoto import mock_aws


class _NoListingHTTPHandler(SimpleHTTPRequestHandler):
    def list_directory(self, path):
        self.send_error(403, "Directory listing not allowed")
        return None

    def log_message(self, format, *args):
        pass


def _running_on_github_ci() -> bool:
    return os.getenv("GITHUB_ACTIONS") == "true" or os.getenv("CI") == "true"


# NOTE(temporary): force the store tests ON for the feat/ngio-settings PR so the full
# CI matrix actually exercises the aiomoto server-mode harness on macOS (see ci.yml).
# Drop this exception together with the CI matrix override.
_FORCE_MACOS_STORE_TESTS = os.getenv("GITHUB_HEAD_REF") == "feat/ngio-settings"

if sys.platform == "darwin" and _running_on_github_ci() and not _FORCE_MACOS_STORE_TESTS:
    # The store tests require local servers which seem to have issues on macOS CI.
    # Skip the whole module in this case.
    pytestmark = pytest.skip(
        reason="Integration tests (local servers) are skipped on macOS CI",
        allow_module_level=True,
    )


@pytest.fixture
def moto_s3_server():
    """Mock S3 backend via aiomoto in server mode.

    ``server_mode=True`` starts an in-process Moto server and injects
    ``AWS_ENDPOINT_URL`` (plus dummy ``test`` credentials and the ``us-east-1``
    region), so both boto3 and s3fs discover it automatically — no subprocess,
    no hardcoded port. The Moto backend is shared between the synchronous boto3
    client used here and the aiobotocore backend that s3fs drives under the hood.
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
