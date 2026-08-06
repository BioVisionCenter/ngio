"""Committed op-count baselines and their diffs.

The baseline is the whole point of the gate: an exact expected tally per
scenario, so a change that multiplies store reads shows up in review as
`-"get.meta": 12 / +"get.meta": 47` rather than as a percentage nobody can act
on. There are no thresholds — a differing count is a failure until someone
regenerates the baseline deliberately.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = [
    "BASELINE_DIR",
    "baseline_path",
    "load_baseline",
    "render_diff",
    "save_baseline",
]

BASELINE_DIR = Path(__file__).parent / "baselines"


def baseline_path(profile: str) -> Path:
    """Return the baseline file for a store profile."""
    return BASELINE_DIR / f"{profile}.json"


def load_baseline(profile: str) -> dict[str, Any] | None:
    """Return the parsed baseline, or `None` if it has never been generated."""
    path = baseline_path(profile)
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def save_baseline(
    profile: str, benchmarks: dict[str, dict[str, int]], env: dict[str, str]
) -> Path:
    """Write a baseline file, sorted so diffs stay readable."""
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    path = baseline_path(profile)
    payload = {
        "schema": 1,
        "profile": profile,
        "generated_with": env,
        "benchmarks": {name: benchmarks[name] for name in sorted(benchmarks)},
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    return path


def render_diff(name: str, expected: dict[str, int], observed: dict[str, int]) -> str:
    """Describe every counter that moved, for a failure message."""
    lines = [f"op counts changed for {name!r}:", ""]
    width = max(len(k) for k in set(expected) | set(observed))
    for key in sorted(set(expected) | set(observed)):
        was, now = expected.get(key, 0), observed.get(key, 0)
        if was == now:
            continue
        delta = now - was
        pct = f"{delta / was:+.0%}" if was else "new"
        lines.append(f"  {key:<{width}}  {was:>10} -> {now:>10}  ({delta:+d}, {pct})")
    lines += [
        "",
        "If this change is intended, regenerate the baseline with:",
        "  pixi run -e test11 pytest tests/benchmarks --bench-update-baseline",
        "and include the resulting diff in the pull request.",
    ]
    return "\n".join(lines)
