"""Synthetic datasets the benchmarks run against.

Fixtures are generated, never downloaded: the Zenodo datasets are
network-dependent and up to 30 GB, which neither CI nor a reproducible baseline
can afford. `compressors=None` throughout, so `bytes.read`/`bytes.written` are
exact and a numcodecs bump cannot move the baseline. Compression behaviour is
the timing layer's concern.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping

from ngio.images._create_synt_container import create_synthetic_ome_zarr

__all__ = ["FIXTURES", "FixtureSpec", "build_fixture", "spec_hash"]

#: Bump to invalidate every prepared fixture regardless of its own spec.
FIXTURE_FORMAT_VERSION = 1


@dataclass(frozen=True)
class FixtureSpec:
    """A reproducible description of one synthetic dataset."""

    name: str
    kind: Literal["image"]
    params: Mapping[str, Any] = field(default_factory=dict)
    tier: Literal["ci", "large"] = "ci"


def spec_hash(spec: FixtureSpec) -> str:
    """Return a stable content hash used to detect a stale prepared fixture."""
    payload = json.dumps(
        {"format": FIXTURE_FORMAT_VERSION, **asdict(spec)},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()[:16]


_IMAGE_BASE: dict[str, Any] = {
    "shape": (2, 8, 512, 512),
    "axes_names": ("c", "z", "y", "x"),
    "channels_meta": ("Channel 1", "Channel 2"),
    "levels": 3,
    "chunks": (1, 1, 256, 256),
    "compressors": None,
}

FIXTURES: dict[str, FixtureSpec] = {
    "img_v05": FixtureSpec(
        name="img_v05",
        kind="image",
        params={**_IMAGE_BASE, "ngff_version": "0.5"},
    ),
    # Same bytes, older metadata. The pair isolates the cost of the decoder
    # fall-through in `get_ngio_meta`, which tries v0.4 first and must fail a
    # full pydantic validation before reaching the v0.5 decoder.
    "img_v04": FixtureSpec(
        name="img_v04",
        kind="image",
        params={**_IMAGE_BASE, "ngff_version": "0.4"},
    ),
}


def build_fixture(spec: FixtureSpec, store: Any) -> None:
    """Create the dataset described by `spec` in `store`."""
    if spec.kind != "image":
        raise NotImplementedError(f"unsupported fixture kind: {spec.kind!r}")
    params = dict(spec.params)
    create_synthetic_ome_zarr(
        store=store,
        shape=tuple(params.pop("shape")),
        axes_names=list(params.pop("axes_names")),
        channels_meta=list(params.pop("channels_meta")),
        overwrite=True,
        **params,
    )
