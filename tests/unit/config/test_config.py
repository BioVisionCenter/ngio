import pytest
from pydantic import ValidationError

from ngio.config import (
    ConstantBackoff,
    ExponentialBackoff,
    LinearBackoff,
    NgioConfig,
    RetryConfig,
    get_config,
)
from ngio.config._config import (
    _DEFAULT_CONFIG_PATH,
    _ENV_VAR,
    S3FSConfig,
    _load_config_data,
    _resolve_config_path,
)
from ngio.utils import NgioUserWarning, NgioValidationError


def test_load_config_data_missing_file_returns_empty_dict(monkeypatch, tmp_path):
    monkeypatch.setenv(_ENV_VAR, str(tmp_path / "nope.json"))
    assert _load_config_data() == {}


def test_load_config_data_valid_json(monkeypatch, tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text('{"s3fs": {"custom_retry_markers": ["Foo"]}}')
    monkeypatch.setenv(_ENV_VAR, str(path))
    assert _load_config_data() == {"s3fs": {"custom_retry_markers": ["Foo"]}}


def test_load_config_data_toml_now_unsupported(monkeypatch, tmp_path):
    path = tmp_path / "cfg.toml"
    path.write_text('[s3fs]\ncustom_retry_markers = ["Foo"]\n')
    monkeypatch.setenv(_ENV_VAR, str(path))
    with pytest.raises(
        NgioValidationError, match="Unsupported ngio config file extension"
    ):
        _load_config_data()


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


def test_resolve_config_path_env_var_set(monkeypatch, tmp_path):
    path = tmp_path / "x.json"
    monkeypatch.setenv(_ENV_VAR, str(path))
    assert _resolve_config_path() == path


def test_resolve_config_path_default(monkeypatch):
    monkeypatch.delenv(_ENV_VAR, raising=False)
    assert _resolve_config_path() == _DEFAULT_CONFIG_PATH


def test_s3fs_config_defaults():
    assert S3FSConfig().custom_retry_markers == []


def test_ngio_config_defaults():
    assert NgioConfig().s3fs is None


def test_ngio_config_validate_assignment_rejects_invalid_type():
    config = NgioConfig()
    with pytest.raises(ValidationError):
        config.s3fs = "not-a-dict"


def test_get_config_singleton_identity():
    assert get_config() is get_config()


def test_retry_config_defaults():
    retry = RetryConfig()
    assert retry.max_retries == 0
    assert retry.backoff == ExponentialBackoff()
    assert retry.retry_on == []
    assert retry.retry_all_errors is False


def test_ngio_config_default_io_retry():
    assert NgioConfig().io_retry == RetryConfig()


def test_retry_config_from_json(monkeypatch, tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text(
        '{"io_retry": {"max_retries": 3, '
        '"backoff": {"strategy": "linear", "delay_s": 0.5}, '
        '"retry_on": ["OSError", "Timeout"]}}'
    )
    monkeypatch.setenv(_ENV_VAR, str(path))
    config = NgioConfig.model_validate(_load_config_data())
    assert config.io_retry.max_retries == 3
    assert config.io_retry.backoff == LinearBackoff(delay_s=0.5)
    assert config.io_retry.retry_on == ["OSError", "Timeout"]


def test_retry_config_rejects_negative_max_retries():
    with pytest.raises(ValidationError):
        RetryConfig(max_retries=-1)


@pytest.mark.parametrize("field", ["delay_s", "max_delay_s"])
@pytest.mark.parametrize(
    "backoff_cls", [ConstantBackoff, LinearBackoff, ExponentialBackoff]
)
def test_backoff_rejects_negative_delays(backoff_cls, field):
    with pytest.raises(ValidationError):
        backoff_cls(**{field: -0.1})


def test_backoff_unknown_strategy_rejected():
    with pytest.raises(ValidationError):
        RetryConfig.model_validate({"backoff": {"strategy": "fibonacci"}})


def test_retry_config_blanket_mode_warns():
    with pytest.warns(NgioUserWarning, match="retry_all_errors"):
        RetryConfig(retry_all_errors=True)


def test_retry_on_and_blanket_mode_mutually_exclusive():
    with pytest.raises(ValidationError, match="mutually exclusive"):
        RetryConfig(retry_on=["OSError"], retry_all_errors=True)


def test_retry_on_assignment_rejected_in_blanket_mode():
    with pytest.warns(NgioUserWarning, match="retry_all_errors"):
        retry = RetryConfig(retry_all_errors=True)
    with pytest.raises(ValidationError, match="mutually exclusive"):
        retry.retry_on = ["OSError"]


def test_retry_config_blanket_mode_warns_on_assignment():
    retry = RetryConfig()
    with pytest.warns(NgioUserWarning, match="retry_all_errors"):
        retry.retry_all_errors = True


def test_retry_config_no_warning_by_default(recwarn):
    RetryConfig()
    RetryConfig(max_retries=5, retry_on=["OSError"])
    assert len(recwarn) == 0
