---
description: Configure ngio via ngio_config.json, including the io_retry policy.
---

# 7. Configuration

**Tune the cross-cutting IO behaviour.**

ngio has a small global configuration object, read from a JSON file and also reachable programmatically. It is loaded once, during `import ngio`, and cached for the life of the process.

## The config file

By default ngio looks for `~/.ngio/ngio_config.json`. You can point it somewhere else with the `NGIO_CONFIG_PATH` environment variable — set it **before** you import ngio, since the file is read during import. Only `.json` files are supported. A missing file means all defaults.

```json
{
    "io_retry": {
        "max_retries": 3,
        "backoff": {"strategy": "exponential", "delay_s": 0.1, "max_delay_s": 10.0},
        "retry_on": ["RequestTimeTooSkewed", "ServerTimeoutError"]
    },
    "s3fs": {
        "custom_retry_markers": ["RequestTimeTooSkewed"]
    }
}
```

`get_config()` returns that object — an `NgioConfig` — so you can inspect or change the configuration at runtime:

```python
from ngio import get_config
from ngio.config import RetryConfig, LinearBackoff

config = get_config()
config.io_retry = RetryConfig(
    max_retries=3,
    backoff=LinearBackoff(delay_s=0.5),
    retry_on=["ServerTimeoutError"],
)
```

## IO retry (`io_retry`)

By default ngio never retries a failed IO operation (`max_retries=0`). When enabled, the policy applies to **all** ngio IO: zarr metadata and pixel data (including lazy dask reads/writes executed on workers), non-zarr table IO (parquet/CSV via pyarrow, AnnData writes), and remote store probes.

Fields:

- `max_retries`: how many times a failed operation is retried (`0` disables retry; `3` means up to 4 total attempts).
- `backoff`: one of three strategies, each with the same attributes (`delay_s`, `max_delay_s`, `jitter`):
    - `constant`: wait `delay_s` between retries.
    - `linear`: wait `delay_s * attempt`.
    - `exponential` (default): wait `delay_s * 2 ** (attempt - 1)`.
    `jitter` multiplies the delay by a random factor in `[0.5, 1.5]`; the result is capped at `max_delay_s` both before and after jitter is applied.
- `retry_on`: a list of substrings matched against `"ExceptionName: message"`. An error is retried only if at least one marker matches, so you can match either an exception class name (`"TimeoutError"`) or a message fragment (`"RequestTimeTooSkewed"`).
- `retry_all_errors`: retry every error. This is **discouraged** — it also retries errors that will never succeed (permissions, missing keys, bugs), multiplying the time to failure. Enabling it emits an `NgioUserWarning`, and it is mutually exclusive with `retry_on`. Prefer narrowing `retry_on` to the specific transient errors you observe.

ngio's own errors (`NgioError` subclasses, e.g. validation errors) are never retried, in any mode.

### Semantics worth knowing

- **Zarr IO snapshots the policy at open time.** Every group ngio opens is backed by a store that copies the current `io_retry` at construction. The snapshot travels with the store — including into pickled dask task graphs, so workers retry with the policy that was active on the driver. Changing `get_config().io_retry` afterwards does not affect already-open containers.
- **Non-zarr IO reads the policy at call time.** The table backends and store probes check the current global config on every call, so runtime changes apply immediately there.
- Retries are logged as warnings, including the error, attempt count, and sleep time. ngio names its loggers `ngio:<module>` (here `ngio:ngio.utils._retry`) — note the colon, which means they are not children of a `ngio` logger in Python's dot-separated hierarchy, so attach handlers to the full name.

## s3fs retry markers (`s3fs`)

`s3fs.custom_retry_markers` is a separate, lower-level mechanism: it registers an error handler inside `s3fs` itself, making s3fs's internal request loop retry any botocore error whose message contains one of the markers (the motivating case is AWS clock-skew `RequestTimeTooSkewed` errors). Apply changes at runtime with `ngio.utils.refresh_s3fs_config(get_config())`.

The two mechanisms are complementary and independent: `s3fs` retries individual S3 requests inside a single ngio IO call, while `io_retry` retries the whole ngio IO call. If both are enabled and their triggers overlap, an error can be retried at both layers, so the effective number of attempts is multiplicative — keep the two configurations narrow.

## Next steps

- [Quickstart](0_quickstart.md) — if you have not opened a container yet.
- [Contributing](../contributing.md) — set up a development environment.
