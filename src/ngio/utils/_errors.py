"""Error classes raised by ngio.

Every ngio error subclasses `NgioError`, so `except NgioError` catches all of
them. Each one also subclasses the builtin it stands in for, so code that
catches `ValueError`, `KeyError`, `FileNotFoundError` or `FileExistsError`
keeps working when ngio raises.
"""


class NgioError(Exception):
    """Base class for all ngio errors."""


class NgioFileNotFoundError(NgioError, FileNotFoundError):
    """Error raised when a file is not found."""


class NgioFileExistsError(NgioError, FileExistsError):
    """Error raised when a file already exists."""


class NgioValidationError(NgioError, ValueError):
    """Error raised when stored data does not match the spec ngio expects.

    Use for on-disk state: malformed OME-Zarr metadata, an unparsable config
    file, a group whose contents contradict its attributes. For a bad argument
    from the caller use `NgioValueError` instead.
    """


class NgioTableValidationError(NgioValidationError):
    """Error raised when a table does not match the spec ngio expects.

    The table-specific case of `NgioValidationError`.
    """


class NgioValueError(NgioError, ValueError):
    """Error raised when an argument fails a run time check.

    Use for values supplied by the caller: an unknown mode, a negative shape,
    two mutually exclusive arguments passed together. For data read off disk
    use `NgioValidationError` instead.
    """


class NgioKeyError(NgioError, KeyError):
    """Error raised when a named image, label, table or key does not exist."""

    def __str__(self) -> str:
        """Return the message unquoted, unlike `KeyError`."""
        return super(KeyError, self).__str__()
