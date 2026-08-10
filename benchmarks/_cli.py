"""The entry point. Knows about blocks and flags; knows nothing about ngio.

    python -m benchmarks
    python -m benchmarks --list
    python -m benchmarks --only consolidate --axis z=16,64,256,1024
    python -m benchmarks --env current --env 'ngio==1.0.0'
    python -m benchmarks --config experiments/consolidate-ceiling.toml
"""

from __future__ import annotations

import argparse
import importlib
import shutil
import tempfile
from pathlib import Path
from typing import Any, NamedTuple

from benchmarks._config import Experiment, load_config
from benchmarks._envs import run_envs
from benchmarks._measure import (
    Override,
    Result,
    Skip,
    apply_overrides,
    cases,
    measure,
)
from benchmarks._output import (
    as_row,
    check_csv,
    environment,
    report,
    report_envs,
    write_csv,
)
from benchmarks.blocks import BLOCKS

#: Used when a block declares no `REPEATS` of its own.
DEFAULT_REPEATS = 5

class Settings(NamedTuple):
    """One run, after a config file and the command line have been merged.

    `main` reads only from this. Resolving once is what keeps `--env` honest:
    the parent used to rebuild the child passthrough from the raw `--axis`
    flags, which would have silently dropped everything a config supplied.
    """

    blocks: list[str] | None  # None means every block
    overrides: list[Override]
    envs: list[str]
    csv: Path | None
    repeats: int | None
    warmup: int
    keep: bool
    quiet: bool


def resolve(args: argparse.Namespace, config: Experiment) -> Settings:
    """Merge a config file with the command line; the command line wins.

    Config overrides go *first* in the list so a later `--axis` for the same
    axis replaces them -- `apply_overrides` applies in order.
    """
    return Settings(
        blocks=args.only or config.blocks,
        overrides=[*config.overrides, *(parse_override(r) for r in args.axis or [])],
        envs=args.env or config.envs or [],
        csv=args.csv or config.csv,
        repeats=args.repeats or config.repeats,
        warmup=_first(args.warmup, config.warmup, 1),
        keep=_first(args.keep, config.keep, False),
        quiet=_first(args.quiet, config.quiet, False),
    )


def _first(*values: Any) -> Any:
    """The first value that was actually set."""
    return next((v for v in values if v is not None), None)


def passthrough(settings: Settings) -> list[str]:
    """Lower resolved settings back to the flags a `--env` child understands.

    The config file itself is never forwarded: it may set `envs`, and a child
    reading it would recurse. Lowering also means a child is told exactly one
    kind of thing, the same kind it was already told before configs existed.
    """
    flags: list[str] = []
    for name in settings.blocks or []:
        flags += ["--only", name]
    for target, name, tokens in settings.overrides:
        key = f"{target}.{name}" if target else name
        flags += ["--axis", f"{key}={','.join(tokens)}"]
    if settings.repeats:
        flags += ["--repeats", str(settings.repeats)]
    if settings.warmup != 1:
        flags += ["--warmup", str(settings.warmup)]
    return flags


def parse_override(raw: str) -> Override:
    """Parse `[block.]axis=v1,v2,...` into its three parts."""
    key, sep, values = raw.partition("=")
    if not sep:
        raise SystemExit(f"--axis wants NAME=VALUES, got {raw!r}")
    block, dot, name = key.partition(".")
    if dot and block in BLOCKS:
        return block, name, values.split(",")
    return None, key, values.split(",")


def load(selected: list[str]) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    """Import the selected block modules; collect the ones that would not.

    A block that cannot import must not take discovery down with it -- under
    `--env` an older ngio is expected to be missing things -- so this returns
    the failures rather than raising them.
    """
    modules: dict[str, Any] = {}
    unavailable: list[tuple[str, str]] = []
    for name in selected:
        try:
            modules[name] = importlib.import_module(BLOCKS[name])
        except ImportError as error:
            unavailable.append((name, str(error)))
    return modules, unavailable


def check_overrides(overrides: list[Override], modules: dict[str, Any]) -> None:
    """Reject an axis override that no selected block would ever read.

    An unqualified override is deliberately tolerant per block so that
    `--axis z=512` can reach several at once -- but tolerant across *all* of
    them would let a typo silently measure the defaults, so it must name an
    axis somewhere.

    Every override is then resolved eagerly, before any measuring starts: a
    bad closed-axis label should not surface after the first block has already
    run.

    Overrides arrive from `--axis` and from a config file's `[axes]`, so the
    messages below name neither.
    """
    for target, name, _ in overrides:
        if target is not None:
            if target not in modules:
                raise SystemExit(
                    f"axis override names block {target!r}, which is not selected"
                )
            continue
        if not any(name in getattr(m, "AXES", {}) for m in modules.values()):
            known = sorted(
                {a for m in modules.values() for a in getattr(m, "AXES", {})}
            )
            raise SystemExit(
                f"no selected block has an axis {name!r}; "
                f"available: {', '.join(known) or 'none'}"
            )
    for name, module in modules.items():
        apply_overrides(name, getattr(module, "AXES", {}), overrides)


def print_list(
    modules: dict[str, Any],
    unavailable: list[tuple[str, str]],
    settings: Settings,
) -> None:
    """Print every selected block and the axes it would actually sweep.

    Overrides are applied first, so `--list` is a dry run of the experiment
    rather than a catalogue of the defaults. With `--config` that is the
    difference between confirming a file does what you meant and being told
    what the blocks happen to declare.
    """
    for name, module in modules.items():
        declared = getattr(module, "AXES", {})
        axes = apply_overrides(name, declared, settings.overrides)
        repeats = settings.repeats or getattr(module, "REPEATS", DEFAULT_REPEATS)
        count = 1
        for values in axes.values():
            count *= len(values)
        print(f"\n{name}  ({count} cases, {repeats} repeats)")
        summary = (module.__doc__ or "").strip().splitlines()
        if summary:
            print(f"  {summary[0]}")
        width = max((len(a) for a in axes), default=0)
        for axis, values in axes.items():
            kind = "closed" if isinstance(declared[axis], dict) else "open"
            print(f"    {axis:<{width}}  [{kind}]  {', '.join(str(v) for v in values)}")
    for name, error in unavailable:
        print(f"\n{name}\n    unavailable: {error}")
    print()


def run_blocks(
    modules: dict[str, Any],
    overrides: list[Override],
    root: Path,
    *,
    repeats: int | None,
    warmup: int,
    quiet: bool,
) -> tuple[list[Result], int]:
    """Measure every case of every module; return the results and skip count."""
    results: list[Result] = []
    skipped = 0
    for name, module in modules.items():
        if not quiet:
            print(f"running {name} ...", flush=True)
        axes = apply_overrides(name, getattr(module, "AXES", {}), overrides)
        count = repeats or getattr(module, "REPEATS", DEFAULT_REPEATS)
        for case in cases(name, axes):
            try:
                measured = module.run(root, **case.values)
            except Skip:
                skipped += 1
                continue
            except ImportError as error:
                # Private paths this block needs are gone: degrade the block,
                # not the run, and stop retrying it case by case.
                results.append(
                    Result(case, float("nan"), float("nan"), f"unavailable: {error}")
                )
                break
            seconds, peak = measure(measured.fn, repeats=count, warmup=warmup)
            results.append(Result(case, seconds, peak, measured.note))
    return results, skipped


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks", description=(__doc__ or "").splitlines()[0]
    )
    parser.add_argument(
        "--only", choices=sorted(BLOCKS), action="append", help="run only this block"
    )
    parser.add_argument(
        "--axis",
        action="append",
        metavar="[BLOCK.]NAME=V1,V2",
        help="replace an axis, e.g. --axis consolidate.z=16,1024 or --axis mode=numpy",
    )
    parser.add_argument(
        "--list", action="store_true", help="print every block and its axes, then exit"
    )
    parser.add_argument(
        "--config",
        type=Path,
        metavar="FILE",
        help="read the experiment from a TOML file; any flag here overrides it",
    )
    parser.add_argument("--repeats", type=int, help="override every block's repeats")
    parser.add_argument("--warmup", type=int, help="untimed runs per case (default 1)")
    # Optional-boolean rather than store_true: without a `--no-` form, a `true`
    # set in a config file could not be switched off from the command line.
    parser.add_argument(
        "--keep",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="reuse/keep generated data in benchmarks/.data instead of a temp dir",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="append results to this CSV (header written only for a new file)",
    )
    parser.add_argument(
        "--env",
        action="append",
        metavar="SPEC",
        help=(
            "run in an isolated uv environment and compare; a comma-separated "
            "requirement list where 'current' means this working tree, "
            "e.g. --env current --env 'ngio==1.0.0,zarr==3.0.6'"
        ),
    )
    parser.add_argument(
        "--quiet",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="suppress the per-run table",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the selected blocks, print the tables."""
    args = build_parser().parse_args(argv)
    settings = resolve(args, load_config(args.config) if args.config else Experiment())
    selected = settings.blocks or sorted(BLOCKS)

    if args.list:
        # Before the env branch, not after: `envs` set in a config file is
        # ambient, so a `--list` that fell through would install and measure
        # every environment instead of printing a listing.
        modules, unavailable = load(selected)
        check_overrides(settings.overrides, modules)
        print_list(modules, unavailable, settings)
        return 0

    if settings.envs:
        # Parent form. The child is invoked without --env, which is also what
        # stops this recursing.
        if settings.keep:
            # Say so rather than ignoring it. `keep` is deliberately not passed
            # to children (a fixture written by one ngio and read by another is
            # a different experiment), so here it can only be inert -- and a
            # config that asks for it should not look like it got it.
            print("note: `keep` does not apply across environments; ignoring it")
        if settings.csv:
            check_csv(settings.csv)
        rows = run_envs(settings.envs, passthrough(settings))
        if settings.csv:
            write_csv(settings.csv, rows)
            print(f"wrote {settings.csv}")
        report_envs(rows)
        return 0

    modules, unavailable = load(selected)
    check_overrides(settings.overrides, modules)
    if settings.csv:
        check_csv(settings.csv)

    if settings.keep:
        root = Path(__file__).parent / ".data"
        root.mkdir(parents=True, exist_ok=True)
        cleanup = None
    else:
        cleanup = tempfile.mkdtemp(prefix="ngio-bench-")
        root = Path(cleanup)

    try:
        results, skipped = run_blocks(
            modules,
            settings.overrides,
            root,
            repeats=settings.repeats,
            warmup=settings.warmup,
            quiet=settings.quiet,
        )
        results += [
            Result(
                _empty_case(name), float("nan"), float("nan"), f"unavailable: {error}"
            )
            for name, error in unavailable
        ]
        if settings.csv:
            env = environment()
            write_csv(settings.csv, [as_row(r, env) for r in results])
        if not settings.quiet:
            # Worth printing because a config's relative `csv` resolves against
            # the config file, which is not where you are standing. Suppressed
            # under --quiet so an `--env` child does not announce the temp file
            # its parent is about to merge and delete.
            if settings.csv:
                print(f"wrote {settings.csv}")
            report(results)
            if skipped:
                # Never let a bounded sweep read as a complete one.
                print(f"({skipped} cases skipped: not measured at that size)\n")
    finally:
        if cleanup is not None:
            shutil.rmtree(cleanup, ignore_errors=True)
    return 0


def _empty_case(block: str):
    from benchmarks._measure import Case

    return Case(block, {}, {})
