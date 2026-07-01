import pytest
from pydantic import ValidationError

from ngio.config import NgioConfig, get_config
from ngio.config._config import (
    _DEFAULT_CONFIG_PATH,
    _ENV_VAR,
    S3FSConfig,
    _load_config_data,
    _resolve_config_path,
)
from ngio.utils import NgioValidationError


def test_load_config_data_missing_file_returns_empty_dict(monkeypatch, tmp_path):
    monkeypatch.setenv(_ENV_VAR, str(tmp_path / "nope.json"))
    assert _load_config_data() == {}


def test_load_config_data_valid_json(monkeypatch, tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text('{"s3fs": {"skew_retry_marker": ["Foo"]}}')
    monkeypatch.setenv(_ENV_VAR, str(path))
    assert _load_config_data() == {"s3fs": {"skew_retry_marker": ["Foo"]}}


def test_load_config_data_valid_toml(monkeypatch, tmp_path):
    path = tmp_path / "cfg.toml"
    path.write_text('[s3fs]\nskew_retry_marker = ["Foo"]\n')
    monkeypatch.setenv(_ENV_VAR, str(path))
    assert _load_config_data() == {"s3fs": {"skew_retry_marker": ["Foo"]}}


def test_load_config_data_unsupported_extension(monkeypatch, tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("s3fs: {}")
    monkeypatch.setenv(_ENV_VAR, str(path))
    with pytest.raises(
        NgioValidationError, match="Unsupported ngio config file extension"
    ):
        _load_config_data()


def test_load_config_data_malformed_json(monkeypatch, tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text("{not valid json")
    monkeypatch.setenv(_ENV_VAR, str(path))
    with pytest.raises(NgioValidationError, match="Failed to parse ngio config file"):
        _load_config_data()


def test_load_config_data_malformed_toml(monkeypatch, tmp_path):
    path = tmp_path / "cfg.toml"
    path.write_text("not = [valid")
    monkeypatch.setenv(_ENV_VAR, str(path))
    with pytest.raises(NgioValidationError, match="Failed to parse ngio config file"):
        _load_config_data()


def test_resolve_config_path_env_var_set(monkeypatch, tmp_path):
    path = tmp_path / "x.json"
    monkeypatch.setenv(_ENV_VAR, str(path))
    assert _resolve_config_path() == path


def test_resolve_config_path_default(monkeypatch):
    monkeypatch.delenv(_ENV_VAR, raising=False)
    assert _resolve_config_path() == _DEFAULT_CONFIG_PATH


def test_s3fs_config_defaults():
    assert S3FSConfig().skew_retry_marker == []


def test_ngio_config_defaults():
    assert NgioConfig().s3fs is None


def test_ngio_config_validate_assignment_rejects_invalid_type():
    config = NgioConfig()
    with pytest.raises(ValidationError):
        config.s3fs = "not-a-dict"


def test_get_config_singleton_identity():
    assert get_config() is get_config()
