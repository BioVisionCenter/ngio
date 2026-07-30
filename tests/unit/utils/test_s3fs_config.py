import sys
import warnings

import pytest
import s3fs

from ngio.config import NgioConfig
from ngio.utils import NgioUserWarning, refresh_s3fs_config


def test_refresh_s3fs_config_apply_and_reset():
    original_handler = s3fs.core.CUSTOM_ERROR_HANDLER
    try:
        assert original_handler is None or not original_handler(
            Exception("boom: RequestTimeTooSkewed")
        )

        config = NgioConfig(s3fs={"custom_retry_markers": ["RequestTimeTooSkewed"]})
        refresh_s3fs_config(config)

        handler = s3fs.core.CUSTOM_ERROR_HANDLER
        assert handler is not original_handler
        assert handler(Exception("boom: RequestTimeTooSkewed")) is True
        assert handler(Exception("unrelated error")) is False

        config.s3fs = None
        refresh_s3fs_config(config)

        reset_handler = s3fs.core.CUSTOM_ERROR_HANDLER
        assert reset_handler(Exception("boom: RequestTimeTooSkewed")) is False
    finally:
        s3fs.set_custom_error_handler(original_handler)


def test_refresh_s3fs_config_noop_when_s3fs_not_installed(monkeypatch):
    original_handler = s3fs.core.CUSTOM_ERROR_HANDLER
    monkeypatch.setitem(sys.modules, "s3fs", None)

    config = NgioConfig(s3fs={"custom_retry_markers": ["RequestTimeTooSkewed"]})
    refresh_s3fs_config(config)

    assert s3fs.core.CUSTOM_ERROR_HANDLER is original_handler


def test_refresh_s3fs_config_warns_on_old_s3fs(monkeypatch):
    """An s3fs predating `set_custom_error_handler` must not break `import ngio`.

    s3fs is not an ngio dependency, and this runs at import scope.
    """
    monkeypatch.delattr(s3fs, "set_custom_error_handler", raising=True)
    monkeypatch.setattr(s3fs, "__version__", "2025.9.0", raising=False)

    config = NgioConfig(s3fs={"custom_retry_markers": ["RequestTimeTooSkewed"]})
    with pytest.warns(NgioUserWarning, match="s3fs>=2026.2.0"):
        refresh_s3fs_config(config)


def test_refresh_s3fs_config_silent_on_old_s3fs_without_markers(monkeypatch):
    """Nothing was requested, so there is nothing to warn about."""
    monkeypatch.delattr(s3fs, "set_custom_error_handler", raising=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        refresh_s3fs_config(NgioConfig())
