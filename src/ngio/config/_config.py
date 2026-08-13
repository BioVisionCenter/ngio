import json
import os
import random
import warnings
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ngio.utils._errors import NgioValidationError
from ngio.utils._warnings import NgioUserWarning

_ENV_VAR = "NGIO_CONFIG_PATH"
_DEFAULT_CONFIG_PATH = Path.home() / ".ngio" / "ngio_config.json"


class S3FSConfig(BaseModel):
    custom_retry_markers: list[str] = Field(default_factory=list)


class _BackoffBase(BaseModel):
    """Shared attributes and delay computation for backoff strategies."""

    delay_s: float = Field(default=0.1, ge=0)
    max_delay_s: float = Field(default=10.0, ge=0)
    jitter: bool = True
    model_config = ConfigDict(validate_assignment=True)

    def _scale(self, attempt: int) -> float:
        raise NotImplementedError

    def compute_delay(self, attempt: int) -> float:
        """Return the delay in seconds before retry number `attempt` (1-based).

        The delay is capped at `max_delay_s`, before and after jitter
        (a uniform factor in [0.5, 1.5]) is applied.
        """
        delay = min(self.delay_s * self._scale(attempt), self.max_delay_s)
        if self.jitter:
            delay = min(delay * random.uniform(0.5, 1.5), self.max_delay_s)
        return delay


class ConstantBackoff(_BackoffBase):
    """Wait `delay_s` between every retry."""

    strategy: Literal["constant"] = "constant"

    def _scale(self, attempt: int) -> float:
        return 1.0


class LinearBackoff(_BackoffBase):
    """Wait `delay_s * attempt`, growing linearly with each retry."""

    strategy: Literal["linear"] = "linear"

    def _scale(self, attempt: int) -> float:
        return float(attempt)


class ExponentialBackoff(_BackoffBase):
    """Wait `delay_s * 2 ** (attempt - 1)`, doubling with each retry."""

    strategy: Literal["exponential"] = "exponential"

    def _scale(self, attempt: int) -> float:
        return 2.0 ** (attempt - 1)


BackoffStrategy = Annotated[
    ConstantBackoff | LinearBackoff | ExponentialBackoff,
    Field(discriminator="strategy"),
]


class RetryConfig(BaseModel):
    """Retry policy for ngio IO operations.

    With the default `max_retries=0` no IO operation is ever retried.
    An error is retried only if any entry of `retry_on` is a substring of
    `f"{type(error).__name__}: {error}"` (so both exception class names and
    message fragments can be matched), or if `retry_all_errors` is enabled
    (mutually exclusive with `retry_on`).
    Ngio's own errors (`NgioError` subclasses) are never retried.

    Example:
        ```python
        RetryConfig(
            max_retries=3,
            backoff=LinearBackoff(delay_s=0.5),
            retry_on=["RequestTimeTooSkewed"],
        )
        ```
    """

    max_retries: int = Field(default=0, ge=0)
    backoff: BackoffStrategy = Field(default_factory=ExponentialBackoff)
    retry_on: list[str] = Field(default_factory=list)
    retry_all_errors: bool = False
    model_config = ConfigDict(validate_assignment=True)

    @model_validator(mode="after")
    def _check_error_matching(self) -> "RetryConfig":
        if self.retry_all_errors and self.retry_on:
            raise ValueError(
                "io_retry.retry_on and io_retry.retry_all_errors are mutually "
                "exclusive: remove the retry_on markers or disable "
                "retry_all_errors."
            )
        if self.retry_all_errors:
            warnings.warn(
                "io_retry.retry_all_errors is enabled: every IO error will be "
                "retried, including non-transient ones. Prefer listing specific "
                "error markers in io_retry.retry_on.",
                NgioUserWarning,
                stacklevel=2,
            )
        return self


class ConsolidationConfig(BaseModel):
    """When `consolidate(mode="auto")` may build a pyramid in memory.

    `mode="numpy"` holds a whole level at once and is 3-5x faster than the
    chunked path for it; `mode="dask"` stays chunk-bounded whatever the size.
    `"auto"` takes the fast one only below `numpy_max_bytes`, measured against
    the source level rather than the whole pyramid -- the chain never holds more
    than two adjacent levels, and peak is around 1.6x the source.

    Size is necessary but not sufficient. `"auto"` also declines whenever the
    two paths would not agree exactly: `dask` zooms per block with no halo, so
    it matches the whole-array zoom only for an integral-ratio downsample at
    `order` in `{"nearest", "linear"}`. Outside that envelope a size threshold
    would silently pick between two different answers.

    Example:
        ```python
        ConsolidationConfig(numpy_max_bytes=0)  # never build in memory
        ```
    """

    numpy_max_bytes: int = Field(default=256 * 2**20, ge=0)
    model_config = ConfigDict(validate_assignment=True)


class NgioConfig(BaseModel):
    """Global configuration for ngio."""

    s3fs: S3FSConfig | None = None
    io_retry: RetryConfig = Field(default_factory=RetryConfig)
    consolidation: ConsolidationConfig = Field(default_factory=ConsolidationConfig)
    model_config = ConfigDict(validate_assignment=True)


def _resolve_config_path() -> Path:
    if env_path := os.environ.get(_ENV_VAR):
        return Path(env_path)
    return _DEFAULT_CONFIG_PATH


def _load_config_data() -> dict[str, Any]:
    path = _resolve_config_path()
    if not path.exists():
        return {}

    if path.suffix != ".json":
        raise NgioValidationError(
            f"Unsupported ngio config file extension '{path.suffix}' "
            f"for {path}. Use a '.json' file."
        )

    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise NgioValidationError(
            f"Failed to parse ngio config file {path}: {e}"
        ) from e


_config: NgioConfig | None = None


def get_config() -> NgioConfig:
    """Return the global ngio configuration singleton.

    The singleton is built on first call rather than at import of this module.
    In practice `import ngio` already triggers that first call (see
    `ngio.utils._zarr_utils`), so `NGIO_CONFIG_PATH` must be set *before*
    importing ngio.

    Note:
        Use `_reset_config()` to force a reload, e.g. in tests.
    """
    global _config
    if _config is None:
        _config = NgioConfig.model_validate(_load_config_data())
    return _config


def _reset_config() -> None:
    """Drop the cached configuration so the next `get_config` reloads it."""
    global _config
    _config = None
