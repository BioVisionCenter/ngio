import asyncio
import errno
from collections.abc import Mapping
from pathlib import Path

import pytest

import ngio.config._config as config_mod
import ngio.utils._retry as retry_mod
from ngio.config import (
    ConstantBackoff,
    ExponentialBackoff,
    LinearBackoff,
    NgioConfig,
    RetryConfig,
)
from ngio.utils import NgioFileExistsError, NgioValueError, retry_io
from ngio.utils._retry import (
    _SHARING_VIOLATION_WINERRORS,
    aretry_call,
    is_retryable,
    is_sharing_violation,
    retry_call,
)

_NO_BACKOFF = ConstantBackoff(delay_s=0.0, jitter=False)


def _policy(**kwargs) -> RetryConfig:
    kwargs.setdefault("max_retries", 3)
    kwargs.setdefault("backoff", _NO_BACKOFF)
    return RetryConfig(**kwargs)


class TestIsRetryable:
    def test_no_markers_never_retries(self):
        assert not is_retryable(OSError("boom"), _policy())

    def test_matches_exception_class_name(self):
        assert is_retryable(OSError("boom"), _policy(retry_on=["OSError"]))

    def test_matches_message_fragment(self):
        exc = ValueError("RequestTimeTooSkewed: clock drift")
        assert is_retryable(exc, _policy(retry_on=["RequestTimeTooSkewed"]))

    def test_non_matching_marker(self):
        assert not is_retryable(OSError("boom"), _policy(retry_on=["Timeout"]))

    def test_blanket_mode_matches_everything(self):
        with pytest.warns(UserWarning):
            policy = _policy(retry_all_errors=True)
        assert is_retryable(OSError("boom"), policy)

    def test_ngio_errors_never_retried(self):
        with pytest.warns(UserWarning):
            policy = _policy(retry_all_errors=True)
        assert not is_retryable(NgioValueError("bad value"), policy)

    def test_ngio_errors_never_retried_by_marker(self):
        policy = _policy(retry_on=["NgioValueError"])
        assert not is_retryable(NgioValueError("bad value"), policy)

    @pytest.mark.parametrize(
        "exc", [KeyboardInterrupt(), SystemExit(), asyncio.CancelledError()]
    )
    def test_control_flow_exceptions_never_retried(self, exc):
        with pytest.warns(UserWarning):
            policy = _policy(retry_all_errors=True)
        assert not is_retryable(exc, policy)


def _win_error(winerror: int) -> PermissionError:
    """Build the error Windows raises for a file-sharing conflict.

    `winerror` is assignable on any platform, so these tests run everywhere.
    """
    exc = PermissionError(13, "Access is denied")
    exc.winerror = winerror
    return exc


def _crt_error(filename, err: int = errno.EACCES) -> PermissionError:
    """Build the error `open()` raises on Windows for the same conflict.

    CPython opens files through the CRT, which sets `errno` and leaves
    `winerror` unset — hence the shape a concurrent reader of a
    delete-pending `zarr.json` sees.
    """
    return PermissionError(err, "Permission denied", str(filename))


class TestIsSharingViolation:
    @pytest.mark.parametrize("winerror", sorted(_SHARING_VIOLATION_WINERRORS))
    def test_matches_win32_codes(self, winerror):
        assert is_sharing_violation(_win_error(winerror))

    def test_other_win32_code_not_matched(self):
        assert not is_sharing_violation(_win_error(2))

    def test_permission_error_without_winerror_not_matched(self):
        # The shape botocore/s3fs raise for an S3 403: no code of either kind.
        assert not is_sharing_violation(PermissionError("Access Denied"))

    def test_matches_crt_eacces(self, tmp_path: Path):
        target = tmp_path / "zarr.json"
        target.write_text("{}")
        assert is_sharing_violation(_crt_error(target))

    def test_directory_eacces_not_matched(self, tmp_path: Path):
        # Opening a directory is EACCES on Windows too, but permanently so.
        assert not is_sharing_violation(_crt_error(tmp_path))

    def test_other_errno_not_matched(self, tmp_path: Path):
        assert not is_sharing_violation(
            _crt_error(tmp_path / "zarr.json", errno.ENOENT)
        )

    def test_ngio_errors_never_matched(self):
        # NgioFileExistsError is an OSError subclass, so it reaches the check.
        exc = NgioFileExistsError("already there")
        exc.winerror = 5
        assert not is_sharing_violation(exc)

    def test_non_os_error_not_matched(self):
        exc = ValueError("boom")
        exc.winerror = 5
        assert not is_sharing_violation(exc)


class TestBackoffStrategies:
    def test_constant(self):
        backoff = ConstantBackoff(delay_s=0.5, jitter=False)
        assert backoff.compute_delay(1) == 0.5
        assert backoff.compute_delay(5) == 0.5

    def test_linear(self):
        backoff = LinearBackoff(delay_s=0.1, jitter=False)
        assert backoff.compute_delay(1) == pytest.approx(0.1)
        assert backoff.compute_delay(2) == pytest.approx(0.2)
        assert backoff.compute_delay(3) == pytest.approx(0.3)

    def test_exponential(self):
        backoff = ExponentialBackoff(delay_s=0.1, jitter=False)
        assert backoff.compute_delay(1) == pytest.approx(0.1)
        assert backoff.compute_delay(2) == pytest.approx(0.2)
        assert backoff.compute_delay(3) == pytest.approx(0.4)

    def test_max_delay_cap(self):
        backoff = ExponentialBackoff(delay_s=1.0, max_delay_s=3.0, jitter=False)
        assert backoff.compute_delay(10) == 3.0

    def test_jitter_bounds(self, monkeypatch):
        backoff = ExponentialBackoff(delay_s=1.0, jitter=True, max_delay_s=10.0)
        monkeypatch.setattr(config_mod.random, "uniform", lambda a, b: b)
        assert backoff.compute_delay(1) == pytest.approx(1.5)
        monkeypatch.setattr(config_mod.random, "uniform", lambda a, b: a)
        assert backoff.compute_delay(1) == pytest.approx(0.5)

    def test_jitter_respects_max_delay(self, monkeypatch):
        backoff = ConstantBackoff(delay_s=1.0, jitter=True, max_delay_s=1.2)
        monkeypatch.setattr(config_mod.random, "uniform", lambda a, b: b)
        assert backoff.compute_delay(1) == 1.2


class _Flaky:
    def __init__(self, fail_times: int, exc: Exception | None = None):
        self.fail_times = fail_times
        self.calls = 0
        self.exc = exc or OSError("flaky")

    def __call__(self):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.exc
        return "ok"

    async def async_call(self):
        return self()


class TestRetryCall:
    @pytest.fixture(autouse=True)
    def _no_sleep(self, monkeypatch):
        self.sleeps: list[float] = []
        monkeypatch.setattr(retry_mod, "_sleep", self.sleeps.append)

    def test_succeeds_after_failures(self):
        fn = _Flaky(fail_times=2)
        policy = _policy(retry_on=["OSError"])
        assert retry_call(fn, policy) == "ok"
        assert fn.calls == 3
        assert len(self.sleeps) == 2

    def test_exhausts_retries_and_raises(self):
        fn = _Flaky(fail_times=10)
        policy = _policy(retry_on=["OSError"])
        with pytest.raises(OSError, match="flaky"):
            retry_call(fn, policy)
        assert fn.calls == policy.max_retries + 1

    def test_non_retryable_raises_immediately(self):
        fn = _Flaky(fail_times=1, exc=ValueError("nope"))
        policy = _policy(retry_on=["OSError"])
        with pytest.raises(ValueError, match="nope"):
            retry_call(fn, policy)
        assert fn.calls == 1

    def test_zero_retries_single_call(self):
        fn = _Flaky(fail_times=1)
        policy = _policy(max_retries=0, retry_on=["OSError"])
        with pytest.raises(OSError, match="flaky"):
            retry_call(fn, policy)
        assert fn.calls == 1
        assert self.sleeps == []


class TestARetryCall:
    def test_succeeds_after_failures(self, monkeypatch):
        sleeps: list[float] = []

        async def fake_sleep(delay):
            sleeps.append(delay)

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)
        fn = _Flaky(fail_times=2)
        policy = _policy(retry_on=["OSError"])
        assert asyncio.run(aretry_call(fn.async_call, policy)) == "ok"
        assert fn.calls == 3
        assert len(sleeps) == 2

    def test_non_retryable_raises_immediately(self, monkeypatch):
        async def fake_sleep(delay):
            pass

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)
        fn = _Flaky(fail_times=1, exc=ValueError("nope"))
        policy = _policy(retry_on=["OSError"])
        with pytest.raises(ValueError, match="nope"):
            asyncio.run(aretry_call(fn.async_call, policy))
        assert fn.calls == 1


class _FakeHTTPMapper(Mapping):
    def __init__(self, errors: list[Exception | None]):
        self.errors = errors
        self.calls = 0

    def get(self, key, default=None):
        error = self.errors[min(self.calls, len(self.errors) - 1)]
        self.calls += 1
        if error is not None:
            raise error
        return b"{}"

    def __getitem__(self, key):
        raise KeyError(key)

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 0


def _client_error(status: int) -> Exception:
    from aiohttp import ClientResponseError, RequestInfo
    from multidict import CIMultiDict, CIMultiDictProxy
    from yarl import URL

    request_info = RequestInfo(
        url=URL("http://x"),
        method="GET",
        headers=CIMultiDictProxy(CIMultiDict()),
        real_url=URL("http://x"),
    )
    return ClientResponseError(request_info=request_info, history=(), status=status)


class TestFractalProbeRetry:
    @pytest.fixture(autouse=True)
    def _no_sleep(self, monkeypatch):
        monkeypatch.setattr(retry_mod, "_sleep", lambda _: None)

    def test_transient_http_error_retried(self, monkeypatch):
        from ngio.utils._fractal_fsspec_store import _probe_key

        config = NgioConfig()
        config.io_retry = _policy(retry_on=["ClientResponseError"])
        monkeypatch.setattr(retry_mod, "get_config", lambda: config)
        mapper = _FakeHTTPMapper([_client_error(503), None])
        assert _probe_key(mapper, "http://x", ".zgroup", None) == b"{}"
        assert mapper.calls == 2

    def test_auth_error_never_retried_even_in_blanket_mode(self, monkeypatch):
        from ngio.utils._fractal_fsspec_store import _probe_key

        config = NgioConfig()
        with pytest.warns(UserWarning):
            config.io_retry = _policy(retry_all_errors=True)
        monkeypatch.setattr(retry_mod, "get_config", lambda: config)
        mapper = _FakeHTTPMapper([_client_error(401)])
        with pytest.raises(NgioValueError, match="fractal_token"):
            _probe_key(mapper, "http://x", ".zgroup", None)
        assert mapper.calls == 1


class TestRetryIODecorator:
    def test_default_config_calls_once(self, monkeypatch):
        monkeypatch.setattr(retry_mod, "get_config", NgioConfig)
        fn = _Flaky(fail_times=1)
        wrapped = retry_io(fn.__call__)
        with pytest.raises(OSError, match="flaky"):
            wrapped()
        assert fn.calls == 1

    def test_reads_config_at_call_time(self, monkeypatch):
        monkeypatch.setattr(retry_mod, "_sleep", lambda _: None)
        config = NgioConfig()
        monkeypatch.setattr(retry_mod, "get_config", lambda: config)
        fn = _Flaky(fail_times=2)
        wrapped = retry_io(fn.__call__)
        config.io_retry = RetryConfig(
            max_retries=3, retry_on=["OSError"], backoff=_NO_BACKOFF
        )
        assert wrapped() == "ok"
        assert fn.calls == 3
