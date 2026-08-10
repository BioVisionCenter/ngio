"""Where results go: the CSV file and the two ASCII tables."""

from __future__ import annotations

import csv
import platform
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from benchmarks._measure import Result

#: Axis names across every block, as a literal.
#:
#: Deliberately *not* computed by importing the blocks: under `--env` a block
#: is allowed to fail to import, which would give a child a narrower header
#: than its parent and a merge that KeyErrors. Six names against four blocks
#: is cheap to maintain, a stale entry costs one empty column, and `--list`
#: makes a mismatch obvious.
AXIS_FIELDS = ("alignment", "kernel", "layout", "mode", "n", "size", "z")

CSV_FIELDS = (
    "env",
    "ngio_version",
    "python",
    "zarr",
    "platform",
    "block",
    "case",
    *AXIS_FIELDS,
    "seconds",
    "peak_mb",
    "note",
)


def environment() -> dict[str, str]:
    """Describe the interpreter and libraries these numbers came from.

    Recorded on every row, not once per file. Installing an older ngio also
    resolves *its* dependency versions, so a cross-environment difference can
    come from zarr rather than from ngio. These columns are what make a
    surprise attributable instead of a mystery.
    """
    import zarr

    import ngio

    return {
        "ngio_version": ngio.__version__,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "zarr": zarr.__version__,
        "platform": platform.system().lower(),
    }


def as_row(result: Result, env: dict[str, str], label: str = "") -> dict[str, str]:
    """Flatten one result into a CSV row.

    Axis columns hold *labels*, not values: `layout` is the string `sharded`,
    never a repr of a kwargs dict. Labels are unique per axis and are already
    what `--axis` speaks, so a CSV cell and a CLI token are the same token.
    """
    return {
        **{field: "" for field in CSV_FIELDS},
        **env,
        "env": label,
        "block": result.case.block,
        "case": result.case.label,
        **{k: v for k, v in result.case.labels.items() if k in AXIS_FIELDS},
        "seconds": f"{result.seconds:.6f}",
        "peak_mb": f"{result.peak_mb:.3f}",
        "note": result.note,
    }


def check_csv(path: Path) -> bool:
    """Return whether `path` is an appendable results file; raise if it clashes.

    A file whose header does not match raises rather than being appended to.
    Adding an axis anywhere in the suite changes the header, and silently
    misaligning an existing file's rows is the kind of failure nobody notices
    until the numbers have already been believed.

    Called once before any measuring starts, so an unusable `--csv` costs
    nothing rather than being discovered after a half-hour run.
    """
    if not (path.exists() and path.stat().st_size > 0):
        path.parent.mkdir(parents=True, exist_ok=True)
        return False
    with path.open(newline="") as handle:
        existing = next(csv.reader(handle), [])
    if tuple(existing) != CSV_FIELDS:
        raise SystemExit(
            f"{path} was written with a different set of columns "
            f"({', '.join(existing)}).\nAppending would misalign it -- "
            "write to a new file instead."
        )
    return True


def write_csv(path: Path, rows: Iterable[dict[str, str]]) -> None:
    """Append rows to `path`, writing a header only for a new file."""
    appending = check_csv(path)
    with path.open("a", newline="") as handle:
        # A child running an older ngio may emit a narrower row than this
        # parent knows about; `restval` merges it instead of raising.
        writer = csv.DictWriter(
            handle, fieldnames=CSV_FIELDS, restval="", extrasaction="ignore"
        )
        if not appending:
            writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read back a results CSV."""
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def report(results: list[Result]) -> None:
    """Print one table per block."""
    if not results:
        print("\nno results\n")
        return
    width = max(len(r.case.label) for r in results)
    current = None
    for r in results:
        if r.case.block != current:
            current = r.case.block
            print(f"\n{current}")
            print(f"  {'case':<{width}} {'median':>11} {'peak RAM':>13}  note")
            print(f"  {'-' * width} {'-' * 11} {'-' * 13}  ----")
        seconds = f"{r.seconds * 1000:>8.1f} ms" if r.seconds == r.seconds else " " * 11
        peak = f"{r.peak_mb:>10.1f} MB" if r.peak_mb == r.peak_mb else " " * 13
        print(f"  {r.case.label:<{width}} {seconds} {peak}  {r.note}")
    print()


#: Width of one environment column, matching the data cell rendered below:
#: 10 chars + " ms" + space + 7 chars + " MB".
_COL = 24


def _short(label: str) -> str:
    """Shorten a label for display; the CSV always keeps the full string."""
    return label if len(label) <= _COL else label[: _COL - 1] + "…"


def report_envs(rows: list[dict[str, str]]) -> None:
    """Print one table per block, a column per environment, plus a ratio."""
    envs, blocks = [], []
    for row in rows:
        if row["env"] not in envs:
            envs.append(row["env"])
        if row["block"] not in blocks and row["block"] != "-":
            blocks.append(row["block"])
    if not blocks:
        print("\nno results: every environment failed to run\n")
        return

    index = {(r["env"], r["block"], r["case"]): r for r in rows}
    # The first environment that actually produced data. A failed one still
    # has a row (block "-") recording the failure, so it must be excluded here
    # or ratios would be taken against an environment that never ran.
    measured = [e for e in envs if any(k[0] == e and k[1] != "-" for k in index)]
    base = measured[0] if measured else envs[0]
    last = measured[-1] if measured else None

    for block in blocks:
        cases = []
        for row in rows:
            if row["block"] == block and row["case"] not in cases:
                cases.append(row["case"])
        width = max(len(c) for c in cases)
        head = f"vs {_short(base)}"

        print(f"\n{block}")
        print(
            f"  {'case':<{width}}"
            + "".join(f" {_short(e):>{_COL}}" for e in envs)
            + f" {head:>12}"
        )
        print(
            f"  {'-' * width}" + "".join(f" {'-' * _COL}" for _ in envs) + f" {'-' * 12}"
        )

        for case in cases:
            line = f"  {case:<{width}}"
            seconds: dict[str, float] = {}
            for env in envs:
                row = index.get((env, block, case))
                if row is None:
                    line += f" {'—':>{_COL}}"
                    continue
                seconds[env] = float(row["seconds"])
                line += (
                    f" {seconds[env] * 1000:>10.1f} ms {float(row['peak_mb']):>7.1f} MB"
                )
            if base in seconds and last in seconds and last != base:
                line += f" {seconds[last] / seconds[base]:>11.2f}x"
            print(line)
    print()
