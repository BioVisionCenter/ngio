"""Performance benchmarks for ngio.

Two layers share one set of scenario declarations:

- **Op counters** (`benchmarks._counting`) are deterministic and gate CI from
  `tests/benchmarks/`. They catch the algorithmic regressions that matter —
  metadata re-parsed per call, one group opened per well — and, because counts
  are backend-independent, a local measurement predicts remote cost.
- **Wall clock** (`benchmarks/timing/`) is a manual, non-gating layer.

See `benchmarks/README.md` for the workflow.
"""

from benchmarks._counting import CountingNgioStore, count, counting_store, zero_fill
from benchmarks._registry import Benchmark, benchmark, registry

__all__ = [
    "Benchmark",
    "CountingNgioStore",
    "benchmark",
    "count",
    "counting_store",
    "registry",
    "zero_fill",
]
