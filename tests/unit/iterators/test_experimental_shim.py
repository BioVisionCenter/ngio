"""The `ngio.experimental.iterators` shim forwards to `ngio.iterators`."""

import pytest

import ngio
import ngio.experimental.iterators as experimental_iterators
from ngio import iterators
from ngio.utils import NgioDeprecationWarning


@pytest.mark.parametrize("name", sorted(experimental_iterators.__all__))
def test_shim_warns_and_forwards(name: str):
    with pytest.warns(NgioDeprecationWarning, match=f"ngio.iterators.{name}") as record:
        obj = getattr(experimental_iterators, name)
    assert obj is getattr(iterators, name)
    # The promised removal version is the point of the warning; pin it.
    assert "ngio=1.1" in str(record[0].message)


def test_shim_unknown_attribute():
    with pytest.raises(AttributeError, match="NotAnIterator"):
        _ = experimental_iterators.NotAnIterator


@pytest.mark.parametrize("name", sorted(iterators.__all__))
def test_reexported_at_top_level(name: str):
    assert getattr(ngio, name) is getattr(iterators, name)
    assert name in ngio.__all__
