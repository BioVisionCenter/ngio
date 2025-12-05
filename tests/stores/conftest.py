# conftest.py
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import boto3
import pytest


def _running_on_github_ci() -> bool:
    return os.getenv("GITHUB_ACTIONS") == "true" or os.getenv("CI") == "true"


if sys.platform == "darwin" and _running_on_github_ci():
    # The store tests require local servers which seem to have issues on macOS CI.
    # Skip the whole module in this case.
    pytestmark = pytest.skip(
        reason="Integration tests (local servers) are skipped on macOS CI",
        allow_module_level=True,
    )


def _wait_for_port(proc, host, port, timeout=20):
    """Wait until a TCP port starts accepting connections, or the process dies."""
    start = time.time()
    while time.time() - start < timeout:
        # bail out early if the process crashed
        if proc.poll() is not None:
            raise RuntimeError("moto server process exited early")

        try:
            with socket.create_connection((host, port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.2)

    raise RuntimeError(f"Port {port} did not open in time")


@pytest.fixture(scope="session")
def moto_s3_server(tmp_path_factory):
    host = "127.0.0.1"
    port = 5005

    log_dir = tmp_path_factory.mktemp("moto_logs")
    log_file = log_dir / "server.log"

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # only start S3 service f
    env["MOTO_SERVICE"] = "s3"

    # NOTE: no "s3" argument here anymore
    cmd = [
        sys.executable,
        "-m",
        "moto.server",
        "-p",
        str(port),
    ]

    with log_file.open("wb") as lf:
        proc = subprocess.Popen(
            cmd,
            stdout=lf,
            stderr=subprocess.STDOUT,
            env=env,
        )

    try:
        _wait_for_port(proc, host, port, timeout=20)
    except Exception as e:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        log_text = log_file.read_text(errors="replace")
        raise RuntimeError(
            f"Failed to start moto server: {e}\n--- moto log ---\n{log_text}"
        ) from e

    s3_endpoint_url = f"http://{host}:{port}"
    bucket_name = "s3-ci-test-bucket"
    s3 = boto3.client(
        "s3",
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
        endpoint_url=s3_endpoint_url,
    )
    s3.create_bucket(Bucket=bucket_name)
    yield {"endpoint_url": s3_endpoint_url, "bucket_name": bucket_name}

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def _find_free_port(host="127.0.0.1"):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def http_static_server(tmp_path_factory):
    """
    Serve a temporary directory via `python -m http.server`.

    From the test code's perspective this is read-only: you write files
    directly into `root` on disk, and then access them via HTTP.
    """
    host = "127.0.0.1"
    port = _find_free_port(host)

    root = tmp_path_factory.mktemp("http_static_root")

    cmd = [
        sys.executable,
        "-m",
        "http.server",
        str(port),
        "--bind",
        host,
    ]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        cmd,
        cwd=str(root),  # serve this directory
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )

    try:
        _wait_for_port(proc, host, port, timeout=10)
    except Exception as e:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        raise RuntimeError(f"Failed to start http server: {e}") from e

    yield {"url": f"http://{host}:{port}", "root": Path(root)}

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
