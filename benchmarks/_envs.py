"""The dependency axis: run the same blocks against different installs.

One product; the outer factors are realized by re-exec, the inner ones
in-process. An ngio version cannot be varied inside a running interpreter, so
the parent spawns one child per environment and each child enumerates the
in-process product normally. The unification already shows up in the output:
`ngio_version` and `zarr` have always been result columns, which is to say the
dependency axis was already an axis of the schema and only needed naming.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from benchmarks._output import read_csv

#: The repo root, which is also the working directory every child runs in.
#: `python -m benchmarks` needs it on `sys.path`; because ngio uses a `src/`
#: layout, putting it there does *not* shadow the pinned ngio the child just
#: installed. Under a flat layout this would silently measure the working tree
#: in every environment.
REPO_ROOT = Path(__file__).resolve().parent.parent


def uv_args(spec: str) -> list[str]:
    """Translate one `--env` spec into `uv run` arguments.

    A spec is a comma-separated requirement list; the token `current` means
    this working tree. `--env 'current,zarr==3.0.6'` is therefore the working
    tree against a pinned zarr, which is how a difference blamed on ngio gets
    checked against the possibility that it was zarr all along.
    """
    args: list[str] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if token == "current":
            args += ["--with-editable", str(REPO_ROOT)]
        else:
            args += ["--with", token]
    return args


def run_envs(specs: list[str], passthrough: list[str]) -> list[dict[str, str]]:
    """Run the selected blocks once per environment and return merged rows.

    `--no-project` is load-bearing: `uv run` inside a directory containing a
    `pyproject.toml` installs *that* project by default, which would quietly
    install the working-tree ngio and make every environment measure the same
    code.

    An environment that fails to install or import is reported and skipped;
    the rest still run.
    """
    if shutil.which("uv") is None:
        raise SystemExit(
            "--env needs `uv` on PATH (https://docs.astral.sh/uv/). "
            "Install it, or run the suite once per environment by hand."
        )

    rows: list[dict[str, str]] = []
    python = f"{sys.version_info.major}.{sys.version_info.minor}"
    for spec in specs:
        with tempfile.TemporaryDirectory(prefix="ngio-bench-") as tmp:
            out = Path(tmp) / "out.csv"
            command = [
                "uv", "run", "--no-project", "--python", python,
                *uv_args(spec),
                "python", "-m", "benchmarks",
                "--csv", str(out), "--quiet",
                *passthrough,
            ]  # fmt: skip
            print(f"=> {spec}", flush=True)
            # `--keep` is never forwarded. A fixture written by ngio 1.0.0 and
            # read back by the working tree is a different experiment from the
            # one that was asked for, so every child gets its own temp root.
            completed = subprocess.run(command, check=False, cwd=REPO_ROOT)
            if completed.returncode != 0 or not out.exists():
                print(f"   {spec}: FAILED (exit {completed.returncode})")
                rows.append(
                    {
                        "env": spec,
                        "ngio_version": "",
                        "python": python,
                        "zarr": "",
                        "platform": platform.system().lower(),
                        "block": "-",
                        "case": "-",
                        "seconds": "nan",
                        "peak_mb": "nan",
                        "note": f"run failed (exit {completed.returncode})",
                    }
                )
                continue
            # The child does not know which environment it is; the parent does.
            for row in read_csv(out):
                rows.append({**row, "env": spec})
    return rows
