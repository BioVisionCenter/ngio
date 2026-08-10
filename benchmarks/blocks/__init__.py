"""The measured blocks.

A plain dict of module *paths*, not imported modules. Under `--env` a block
may fail to import against an older ngio, and that has to degrade one block
rather than the whole run -- so the CLI imports each selected block lazily.
Same reasoning as `tests/performance/scenarios.py`: one consumer, so a
registry would be a dict with extra steps.

A block earns its place by informing a decision -- *which option should I
choose*, or *what does this cost at scale*. Blocks that merely produce a
number do not. Keep this list short.
"""

BLOCKS = {
    "consolidate": "benchmarks.blocks.consolidate",
    "layout": "benchmarks.blocks.layout",
    "roi": "benchmarks.blocks.roi",
    "algorithms": "benchmarks.blocks.algorithms",
}
