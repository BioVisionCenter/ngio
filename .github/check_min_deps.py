"""Verify the minimum-dependency CI leg is actually testing the declared floors.

Two ways this job could silently pass without testing anything:

1. `.github/min-constraints.txt` drifts from the floors in `pyproject.toml`, so the
   job pins versions nobody declares.
2. A pin does not take effect — the package is absent, or resolution landed on a
   different version — so the suite runs against something newer than the floor.

Both are checked here. Run from the repo root after installing with the
constraints file.
"""

from __future__ import annotations

import re
import sys
import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from packaging.version import Version

ROOT = Path(__file__).resolve().parent.parent
CONSTRAINTS = ROOT / ".github" / "min-constraints.txt"
PYPROJECT = ROOT / "pyproject.toml"

# Distribution name -> import-time metadata name, where they differ.
_REQUIREMENT = re.compile(r"^\s*([A-Za-z0-9._-]+)\s*(\[[^\]]*\])?\s*(.*)$")


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _declared_floors() -> dict[str, str]:
    """Return {normalized name: floor} from pyproject's runtime dependencies."""
    data = tomllib.loads(PYPROJECT.read_text())
    floors: dict[str, str] = {}
    for raw in data["project"]["dependencies"]:
        match = _REQUIREMENT.match(raw)
        if match is None:
            sys.exit(f"could not parse dependency: {raw!r}")
        name, _extras, specifiers = match.groups()
        found = re.search(r">=\s*([0-9][0-9A-Za-z.*+!-]*)", specifiers or "")
        if found is None:
            sys.exit(
                f"{name} has no `>=` lower bound in pyproject.toml. Every runtime "
                "dependency needs a tested floor."
            )
        floors[_normalize(name)] = found.group(1)
    return floors


def _pinned() -> dict[str, str]:
    """Return {normalized name: pinned version} from the constraints file."""
    pins: dict[str, str] = {}
    for line in CONSTRAINTS.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        name, _, pin = line.partition("==")
        if not pin:
            sys.exit(f"constraint must use `==`: {line!r}")
        pins[_normalize(name)] = pin.strip()
    return pins


def main() -> int:
    floors, pins = _declared_floors(), _pinned()
    problems: list[str] = []

    for name, floor in sorted(floors.items()):
        pin = pins.get(name)
        if pin is None:
            problems.append(
                f"{name}: floor >={floor} in pyproject.toml but no pin in "
                f"{CONSTRAINTS.name}"
            )
            continue
        # `>=3.9` and `==3.9.0` denote the same release, so compare as versions.
        if Version(pin) != Version(floor):
            problems.append(
                f"{name}: pyproject floor >={floor} but {CONSTRAINTS.name} pins "
                f"=={pin}"
            )
            continue
        try:
            installed = version(name)
        except PackageNotFoundError:
            problems.append(f"{name}: pinned =={pin} but not installed")
            continue
        if Version(installed) != Version(pin):
            problems.append(
                f"{name}: pinned =={pin} but {installed} is installed, so the "
                "floor is not being tested"
            )

    for name in sorted(set(pins) - set(floors)):
        problems.append(
            f"{name}: pinned in {CONSTRAINTS.name} but not a runtime dependency"
        )

    if problems:
        print("Minimum-dependency check failed:\n")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print(f"All {len(floors)} declared floors are pinned and installed:")
    for name, floor in sorted(floors.items()):
        print(f"  {name}=={floor}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
