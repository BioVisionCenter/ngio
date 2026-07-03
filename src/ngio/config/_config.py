import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ngio.utils._errors import NgioValidationError

_ENV_VAR = "NGIO_CONFIG_PATH"
_DEFAULT_CONFIG_PATH = Path.home() / ".ngio_config.json"


class S3FSConfig(BaseModel):
    custom_retry_markers: list[str] = Field(default_factory=list)


class NgioConfig(BaseModel):
    """Global configuration for ngio."""

    s3fs: S3FSConfig | None = None
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


_config = NgioConfig.model_validate(_load_config_data())


def get_config() -> NgioConfig:
    """Return the global ngio configuration singleton."""
    return _config
