import pytest

from ngio.utils import (
    NgioError,
    NgioFileExistsError,
    NgioFileNotFoundError,
    NgioKeyError,
    NgioTableValidationError,
    NgioValidationError,
    NgioValueError,
)

ALL_ERRORS = [
    NgioFileExistsError,
    NgioFileNotFoundError,
    NgioKeyError,
    NgioTableValidationError,
    NgioValidationError,
    NgioValueError,
]

BUILTIN_FOR = {
    NgioFileExistsError: FileExistsError,
    NgioFileNotFoundError: FileNotFoundError,
    NgioKeyError: KeyError,
    NgioTableValidationError: ValueError,
    NgioValidationError: ValueError,
    NgioValueError: ValueError,
}


@pytest.mark.parametrize("error_cls", ALL_ERRORS)
def test_every_error_is_an_ngio_error(error_cls):
    assert issubclass(error_cls, NgioError)


@pytest.mark.parametrize("error_cls", ALL_ERRORS)
def test_every_error_subclasses_its_builtin(error_cls):
    """No ngio error may be invisible to code catching the builtin."""
    assert issubclass(error_cls, BUILTIN_FOR[error_cls])


def test_table_validation_error_is_a_validation_error():
    assert issubclass(NgioTableValidationError, NgioValidationError)
    assert not issubclass(NgioValidationError, NgioTableValidationError)


def test_key_error_message_is_not_quoted():
    """`KeyError.__str__` reprs its argument; ngio messages must read plainly."""
    message = "Image 'a' not found. Available: ['b']"
    assert str(NgioKeyError(message)) == message
