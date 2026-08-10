"""An experiment as a file: everything `--config` can say.

A benchmark invocation is an experiment, and an experiment that lives only in
shell history cannot be committed next to the CSV it produced, diffed against
last month's, or pasted into an issue. This is that file.

It is deliberately **not** a second way to declare axes. A config entry lowers
to exactly the same override a `--axis` flag produces and goes through the same
`apply_overrides`, so the two cannot drift: `[axes.consolidate] z = [16, 64]`
means what `--axis consolidate.z=16,64` means. The blocks stay the only place
an axis is *declared*; a config can only ever replace one.

A config expresses a strict subset of what the flags do: every axis entry must
name its block. `--axis z=16` stays unqualified because it is a keystroke in an
investigation; a file is read later by someone who was not there.
"""

from __future__ import annotations

import tomllib
from typing import TYPE_CHECKING, Any, NamedTuple

from benchmarks._measure import Override
from benchmarks.blocks import BLOCKS

if TYPE_CHECKING:
    from pathlib import Path

#: Every key a config file may set. Anything else raises rather than being
#: ignored: a typo'd key that silently ran the defaults would be discovered
#: only after the numbers had been believed.
KEYS = ("blocks", "axes", "envs", "csv", "repeats", "warmup", "keep", "quiet")


class Experiment(NamedTuple):
    """A parsed config file. A field is `None` when the file omits it."""

    blocks: list[str] | None = None
    overrides: tuple[Override, ...] = ()
    envs: list[str] | None = None
    csv: Path | None = None
    repeats: int | None = None
    warmup: int | None = None
    keep: bool | None = None
    quiet: bool | None = None


def _tokens(where: str, values: Any) -> list[str]:
    """Render one axis' values as the tokens `--axis` would have produced.

    Stringifying typed TOML back into tokens looks like a loss, but it is what
    keeps a single code path: `apply_overrides` re-parses an open axis against
    the type its block declared, so `z = [16]` and `--axis z=16` arrive
    identically, and a closed axis takes labels either way. Round-tripping an
    int, float, bool or string through `str` is exact.

    A comma is rejected because these tokens are re-joined with commas into an
    `--axis` flag when `--env` spawns a child, and a value containing one would
    split wrongly three layers away from here.

    The scalar types are a whitelist, not a `list`/`dict` blacklist: TOML also
    has bare dates, and `z = 2026-08-10` would otherwise stringify happily,
    fail to parse as the int its axis declared, and sweep the leftover string
    without complaint.
    """
    if not isinstance(values, list) or not values:
        raise SystemExit(f"{where} must be a non-empty list of values")
    tokens = []
    for value in values:
        if not isinstance(value, bool | int | float | str):
            raise SystemExit(
                f"{where} takes numbers, strings or booleans, got "
                f"{type(value).__name__} ({value!r})"
            )
        token = str(value)
        if "," in token:
            raise SystemExit(
                f"{where}: a value may not contain a comma ({token!r}). "
                "Overrides are re-serialised as `--axis` flags for `--env` "
                "children, and a comma would split the value there."
            )
        tokens.append(token)
    return tokens


def _overrides(axes: Any) -> tuple[Override, ...]:
    """Lower an `[axes]` table into `(block, name, tokens)` triples.

    Every entry must name its block: one `[axes.<block>]` table per block,
    whose keys are that block's axes. The unqualified form `--axis z=16`
    accepts at the command line has no config equivalent on purpose -- a
    config is a committed record, and "whichever blocks happen to have a `z`"
    is a statement about the suite's shape at the moment it was written, not
    about the experiment.

    Being able to say only one thing also removes two subtleties: there is no
    general-versus-specific precedence to define, and the `layout` collision
    disappears, because `layout` under `[axes]` can only ever be the block.
    """
    if not isinstance(axes, dict):
        raise SystemExit("`axes` must be a table, e.g. [axes.consolidate]")
    lowered: list[Override] = []
    for key, value in axes.items():
        if not isinstance(value, dict):
            raise SystemExit(
                f"axes.{key} is a bare axis, and every config entry must name "
                f"its block. Write a [axes.<block>] table with `{key} = ...` "
                f"inside it; blocks are: {', '.join(sorted(BLOCKS))}"
            )
        if key not in BLOCKS:
            raise SystemExit(
                f"[axes.{key}] is not a block; choose from: "
                f"{', '.join(sorted(BLOCKS))}"
            )
        for axis, values in value.items():
            lowered.append((key, axis, _tokens(f"axes.{key}.{axis}", values)))
    return tuple(lowered)


def _typed(data: dict[str, Any], key: str, kind: type, where: Path) -> Any:
    """Fetch `key` from `data`, checking its type.

    `bool` is a subclass of `int`, so a plain `isinstance` check would let
    `repeats = true` through and then run some number of repeats nobody asked
    for. Checked explicitly rather than coerced -- a config is a record of an
    experiment, and quietly reinterpreting it defeats the point.
    """
    value = data.get(key)
    mistyped = not isinstance(value, kind) or (
        kind is int and isinstance(value, bool)
    )
    if value is not None and mistyped:
        raise SystemExit(f"{where}: `{key}` must be {kind.__name__}, got {value!r}")
    return value


def load_config(path: Path) -> Experiment:
    """Read and validate a TOML experiment file.

    A relative `csv` resolves against the config file's own directory rather
    than the working directory, so a committed experiment writes its results
    beside itself no matter where it is invoked from.
    """
    try:
        data = tomllib.loads(path.read_text())
    except FileNotFoundError:
        raise SystemExit(f"no such config file: {path}") from None
    except tomllib.TOMLDecodeError as error:
        raise SystemExit(f"{path} is not valid TOML: {error}") from None

    unknown = sorted(set(data) - set(KEYS))
    if unknown:
        raise SystemExit(
            f"{path}: unknown key {unknown[0]!r}; valid keys are {', '.join(KEYS)}"
        )

    blocks = _typed(data, "blocks", list, path)
    if blocks is not None:
        missing = [b for b in blocks if b not in BLOCKS]
        if missing:
            raise SystemExit(
                f"{path}: no block {missing[0]!r}; "
                f"choose from: {', '.join(sorted(BLOCKS))}"
            )

    csv = _typed(data, "csv", str, path)
    return Experiment(
        blocks=blocks,
        overrides=_overrides(data.get("axes", {})),
        envs=_typed(data, "envs", list, path),
        csv=(path.parent / csv) if csv else None,
        repeats=_typed(data, "repeats", int, path),
        warmup=_typed(data, "warmup", int, path),
        keep=_typed(data, "keep", bool, path),
        quiet=_typed(data, "quiet", bool, path),
    )
