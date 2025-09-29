"""Module for joining multiple numpy getter pipes."""

import numpy as np

from ngio.io_pipes._io_pipes import DataGetter


class NumpyJoinedGetters:
    """A class to join multiple numpy getter pipes.

    This class allows you to combine multiple getter pipes into a single interface.
    All getter pipes should return numpy arrays with
    compatible shapes for the operations you intend to perform.
    """

    def __init__(
        self,
        match_dimensions: bool = True,
        broadcast: bool = True,
        pixel_tolerance: int = 1,
    ) -> None:
        """Initialize the NumpyJoinedGetters.

        Args:
            match_dimensions (bool): If True, ensure that all arrays have the same
                shape in all common dimensions. If the shape mismatches more than
                `pixel_tolerance`, an error is raised.
            broadcast (bool): If True, we broadcast single-dimension axes to match
                the shape of the other arrays. If False single-dimension will
                be left as is.
            pixel_tolerance (int): The maximum number of pixels by which dimensions
                can differ when `match_dimensions` is True.
        """
        self._getter_pipes: list[DataGetter] = []
        self._match_dimensions = match_dimensions
        self._broadcast = broadcast
        self._pixel_tolerance = pixel_tolerance

    def add_getter_pipe(
        self,
        getter_pipe: DataGetter[np.ndarray],
    ) -> None:
        """Add a getter pipe to the joined pipes."""
        self._getter_pipes.append(getter_pipe)

    def joined_get(self) -> list[np.ndarray]:
        """Load data from all getter pipes."""
        if len(self._getter_pipes) == 0:
            raise ValueError("No getter pipes have been added.")

        ref_getter, *other_getters = self._getter_pipes
        ref_array = ref_getter.get()
        data = [ref_array]
        for getter in other_getters:
            array = getter.get()
            # Match shapes and broadcast if necessary
            data.append(array)
        return data
