"""The codec-pipeline probe reads zarr internals; these fail when they move."""

import warnings

import pytest
import zarr
from zarr.storage import LocalStore, MemoryStore

from ngio.utils import NgioUserWarning
from ngio.utils._codec_pipeline import (
    _REPORTED,
    active_codec_pipeline,
    warn_on_codec_pipeline_fallback,
)
from ngio.utils._store import NgioStore


@pytest.fixture(autouse=True)
def _forget_reports():
    _REPORTED.clear()
    yield
    _REPORTED.clear()


def _array(store):
    return zarr.create_array(
        store=store, shape=(8, 8), chunks=(4, 4), dtype="uint16", zarr_format=3
    )


def test_active_codec_pipeline_resolves(tmp_path):
    """Canary: `_async_array.codec_pipeline` is private zarr API.

    If a zarr bump renames it the probe degrades to `None` and silently stops
    warning, which is the failure this test exists to make loud.
    """
    assert active_codec_pipeline(_array(LocalStore(tmp_path))) is not None


def test_no_warning_when_the_pipeline_is_the_configured_one(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_on_codec_pipeline_fallback(_array(LocalStore(tmp_path)))


def test_warns_once_on_a_silent_fallback(monkeypatch):
    """A store the configured pipeline rejects must not fail quietly.

    zarr catches the pipeline's `NotImplementedError` and builds its own
    `BatchedCodecPipeline`, while `zarr.config` keeps reporting the configured
    one -- so the configuration is not evidence of what ran.
    """
    monkeypatch.setattr(
        "ngio.utils._codec_pipeline._configured_codec_pipeline",
        lambda: "SomeOtherCodecPipeline",
    )
    array = _array(NgioStore(MemoryStore()))

    with pytest.warns(NgioUserWarning, match="SomeOtherCodecPipeline"):
        warn_on_codec_pipeline_fallback(array)

    # Deduplicated: one array per pyramid level per image would otherwise
    # repeat a single fact thousands of times on a plate.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_on_codec_pipeline_fallback(array)


def test_silent_when_no_pipeline_is_configured(monkeypatch):
    monkeypatch.setattr(
        "ngio.utils._codec_pipeline._configured_codec_pipeline", lambda: None
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_on_codec_pipeline_fallback(_array(MemoryStore()))
