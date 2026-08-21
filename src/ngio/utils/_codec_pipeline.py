"""Detection for zarr's silent codec-pipeline fallback.

`zarr.core.array.create_codec_pipeline` asks the configured pipeline to build
itself for a given store, and swallows the `NotImplementedError` a pipeline
raises when it does not support that store type — falling back to zarr's own
`BatchedCodecPipeline` without a word. `zarr.config` keeps reporting the
configured pipeline either way, so the configuration is not evidence of what
ran. This module asks the array instead.
"""

import warnings

import zarr

from ngio.utils._warnings import NgioUserWarning

#: (configured, actual, store class) triples already reported. One array is
#: opened per level per image, so without this a plate would emit thousands of
#: copies of one fact.
_REPORTED: set[tuple[str, str, str]] = set()


def active_codec_pipeline(array: zarr.Array) -> str | None:
    """The name of the codec pipeline class zarr actually built for `array`.

    Returns `None` if zarr's internals moved; reading them is best-effort and
    must never be the reason ngio raises.
    """
    try:
        return type(array._async_array.codec_pipeline).__name__
    except AttributeError:
        return None


def _configured_codec_pipeline() -> str | None:
    try:
        from zarr.core.config import config

        path = config.get("codec_pipeline.path", None)
    except Exception:
        # Deliberately broad: this is a diagnostic probe over a config surface
        # ngio does not own, and it must never be the reason a read fails.
        return None
    if not isinstance(path, str) or not path:
        return None
    return path.rsplit(".", 1)[-1]


def warn_on_codec_pipeline_fallback(array: zarr.Array) -> None:
    """Warn once if zarr silently ignored the configured codec pipeline."""
    configured = _configured_codec_pipeline()
    if configured is None:
        return
    actual = active_codec_pipeline(array)
    if actual is None or actual == configured:
        return

    store_cls = type(array.store).__name__
    key = (configured, actual, store_cls)
    if key in _REPORTED:
        return
    _REPORTED.add(key)
    warnings.warn(
        f"zarr is configured to use {configured}, but built {actual} for this "
        f"array: {configured} does not support {store_cls} stores, and zarr "
        "falls back without reporting it. Reads and writes are correct but "
        f"lose whatever {configured} was chosen for. Note zarrs currently "
        "accepts only a plain LocalStore.",
        NgioUserWarning,
        stacklevel=2,
    )
