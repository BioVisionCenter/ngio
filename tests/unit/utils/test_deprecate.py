import asyncio
import warnings
from contextlib import contextmanager

import pytest

from ngio.utils import NgioDeprecationWarning, NgioValueError, deprecated
from ngio.utils import deprecated_alias as alias


@contextmanager
def warnings_are_errors():
    with warnings.catch_warnings():
        warnings.simplefilter("error", NgioDeprecationWarning)
        yield


@alias(old_name="new_name")
def _renamed(new_name: int = 0, other: int = 0) -> tuple[int, int]:
    return new_name, other


@alias(removed_in="2.0", first_old="first", second_old="second")
def _two_renamed(first: int = 0, second: int = 0) -> tuple[int, int]:
    return first, second


@deprecated(replacement="new_thing()")
def _old_thing(value: int) -> int:
    return value * 2


@deprecated(replacement="gather_things(max_workers=...)", removed_in="1.5")
async def _old_thing_async(value: int) -> int:
    return value * 3


class _Holder:
    @alias(old_name="new_name")
    def method(self, new_name: int = 0) -> int:
        return new_name


def test_new_name_does_not_warn():
    with warnings_are_errors():
        assert _renamed(new_name=3, other=1) == (3, 1)


def test_positional_args_are_untouched():
    with warnings_are_errors():
        assert _renamed(3, 1) == (3, 1)


def test_old_name_forwards_and_warns():
    with pytest.warns(NgioDeprecationWarning, match="'old_name'.*deprecated") as record:
        assert _renamed(old_name=3, other=1) == (3, 1)
    message = str(record[0].message)
    assert "ngio=1.1" in message
    assert "'new_name'" in message


def test_both_spellings_raise():
    with pytest.raises(NgioValueError, match="both 'old_name' and 'new_name'"):
        _renamed(old_name=1, new_name=2)


def test_custom_removal_version_and_multiple_aliases():
    with pytest.warns(NgioDeprecationWarning) as record:
        assert _two_renamed(first_old=1, second_old=2) == (1, 2)
    assert len(record) == 2
    assert all("ngio=2.0" in str(w.message) for w in record)


def test_works_on_methods():
    with pytest.warns(NgioDeprecationWarning, match="_Holder.method"):
        assert _Holder().method(old_name=7) == 7


def test_deprecated_callable_warns_and_forwards():
    with pytest.warns(NgioDeprecationWarning, match=r"new_thing\(\)") as record:
        assert _old_thing(4) == 8
    assert "ngio=1.1" in str(record[0].message)


def test_deprecated_async_warns_at_call_not_await():
    with pytest.warns(NgioDeprecationWarning, match="ngio=1.5"):
        coro = _old_thing_async(4)
    assert asyncio.run(coro) == 12


@pytest.mark.parametrize(
    "call",
    [lambda: _renamed(old_name=1), lambda: _old_thing(1)],
    ids=["deprecated_alias", "deprecated"],
)
def test_warning_points_at_the_caller(call):
    with pytest.warns(NgioDeprecationWarning) as record:
        call()
    assert record[0].filename == __file__


def test_metadata_is_preserved():
    assert _renamed.__name__ == "_renamed"
    assert _old_thing.__name__ == "_old_thing"
