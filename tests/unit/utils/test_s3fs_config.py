import s3fs

from ngio.config import NgioConfig
from ngio.utils import refresh_s3fs_config


def test_refresh_s3fs_config_apply_and_reset():
    original_handler = s3fs.core.CUSTOM_ERROR_HANDLER
    try:
        config = NgioConfig(s3fs={"skew_retry_marker": ["RequestTimeTooSkewed"]})
        refresh_s3fs_config(config)

        handler = s3fs.core.CUSTOM_ERROR_HANDLER
        assert handler(Exception("boom: RequestTimeTooSkewed")) is True
        assert handler(Exception("unrelated error")) is False

        config.s3fs = None
        refresh_s3fs_config(config)

        reset_handler = s3fs.core.CUSTOM_ERROR_HANDLER
        assert reset_handler(Exception("boom: RequestTimeTooSkewed")) is False
    finally:
        s3fs.set_custom_error_handler(original_handler)
