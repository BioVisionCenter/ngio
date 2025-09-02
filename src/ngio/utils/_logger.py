import logging
import time
from functools import cache

from ngio.utils._errors import NgioValueError

# Configure the logger
ngio_logger = logging.getLogger("NgioLogger")
ngio_logger.setLevel(logging.ERROR)

# Set up a console handler with a custom format
console_handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(name)s - "
    "[%(module)s.%(funcName)s:%(lineno)d]: %(message)s"
)
console_handler.setFormatter(formatter)

# Add the handler to the logger
ngio_logger.addHandler(console_handler)


def set_logger_level(level: str) -> None:
    """Set the logger level.

    Args:
        level: The level to set the logger to.
            Must be one of "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL".
    """
    if level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        raise NgioValueError(f"Invalid log level: {level}")

    ngio_logger.setLevel(level)


@cache
def _warn(message: str, ttl_hash: int) -> None:
    """Log a warning message with a time-to-live (TTL) hash."""
    ngio_logger.warning(message, stacklevel=3)


def ngio_warn(message: str, cooldown: int = 2) -> None:
    """Log a warning message.

    Args:
        message: The warning message to log.
        cooldown: The cooldown period in seconds to avoid repeated logging.
    """
    ttl_hash = time.time() // cooldown
    _warn(message, ttl_hash)
