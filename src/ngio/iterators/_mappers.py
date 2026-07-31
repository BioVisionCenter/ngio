"""Mappers for iterators.

Mappers are classes that can be passed to the `map` method of iterators to
transform the items yielded by the iterator.

"""

from collections.abc import Callable, Iterable
from typing import Generic, Protocol, TypeVar

from ngio.io_pipes._io_pipes_types import DataGetterProtocol, DataSetterProtocol
from ngio.utils import NgioValueError

T = TypeVar("T")


class MapperProtocol(Protocol[T]):
    """Protocol for mappers."""

    def __call__(
        self,
        func: Callable[[T], T],
        getters: Iterable[DataGetterProtocol[T]],
        setters: Iterable[DataSetterProtocol[T] | None],
    ) -> None:
        """Map an item to another item."""
        ...


class BasicMapper(Generic[T]):
    """A basic mapper that simply applies a function to the data."""

    def __call__(
        self,
        func: Callable[[T], T],
        getters: Iterable[DataGetterProtocol[T]],
        setters: Iterable[DataSetterProtocol[T] | None],
    ) -> None:
        """Map an item to another item."""
        for getter, setter in zip(getters, setters, strict=True):
            data = getter()
            data = func(data)
            if setter is None:
                raise NgioValueError(
                    "Error in BasicMapper: setter is None, "
                    "this iterator is read-only, mapping is not possible."
                )
            setter(data)
