"""Transforms and pre-post-processors for label images."""

# def make_unique_label_np(x: np.ndarray, p: int, n: int) -> np.ndarray:
#    """Make a unique label for the patch."""
#    x = np.where(x > 0, (1 + p - n) + x * n, 0)
#    return x
#
#
# def make_unique_label_da(x: da.Array, p: int, n: int) -> da.Array:
#    """Make a unique label for the patch."""
#    x = da.where(x > 0, (1 + p - n) + x * n, 0)
#    return x
