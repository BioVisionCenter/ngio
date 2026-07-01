import json
import os
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from ngio.utils import NgioValidationError

_ENV_VAR = "NGIO_CONFIG_PATH"
_DEFAULT_CONFIG_PATH = Path.home() / ".ngio_config.json"


class NgioConfig(BaseModel):
    """Global configuration for ngio."""

    model_config = ConfigDict(validate_assignment=True)


def _resolve_config_path() -> Path:
    if env_path := os.environ.get(_ENV_VAR):
        return Path(env_path)
    return _DEFAULT_CONFIG_PATH


def _load_config_data() -> dict[str, Any]:
    path = _resolve_config_path()
    if not path.exists():
        return {}

    try:
        if path.suffix == ".json":
            return json.loads(path.read_text())
        elif path.suffix == ".toml":
            return tomllib.loads(path.read_text())
        else:
            raise NgioValidationError(
                f"Unsupported ngio config file extension '{path.suffix}' "
                f"for {path}. Use a '.json' or '.toml' file."
            )
    except (json.JSONDecodeError, tomllib.TOMLDecodeError) as e:
        raise NgioValidationError(
            f"Failed to parse ngio config file {path}: {e}"
        ) from e


_config = NgioConfig.model_validate(_load_config_data())


def get_config() -> NgioConfig:
    """Return the global ngio configuration singleton."""
    return _config
