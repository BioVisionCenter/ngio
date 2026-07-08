import asyncio

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
from ngio.utils import NgioValueError, retry_io
from ngio.utils._retry import aretry_call, is_retryable, retry_call

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
