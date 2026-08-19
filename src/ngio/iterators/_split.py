"""Deterministic partitioning of iterator units into independent jobs.

`split(n_jobs)` promises embarrassing independence: no write unit (a chunk, or
a shard when the output is sharded) is ever shared between two jobs, so jobs
need no locks, no barriers, and no channel to one another — the topology SLURM
array tasks on a shared filesystem actually offer. The partition is built from
the write-conflict *components* (`write_conflict_components`): units whose
footprints conflict travel together, and a component is never divided.
"""


def partition_components(
    components: list[list[int]], n_jobs: int
) -> list[list[int]]:
    """Assign conflict components to `n_jobs` jobs, balanced and deterministic.

    Greedy multiway partitioning: components sorted by (size descending,
    smallest index ascending), each placed on the currently lightest job,
    ties broken by the lowest job index. Every input list stays whole — that
    is the independence guarantee — and each job's indices come back sorted.
    Deterministic by construction: a pure function of `components` and
    `n_jobs`, so every process that computes the same components computes the
    same partition. Jobs may come back empty when `n_jobs` exceeds the
    component count.
    """
    jobs: list[list[int]] = [[] for _ in range(n_jobs)]
    loads = [0] * n_jobs
    for component in sorted(components, key=lambda c: (-len(c), c[0])):
        lightest = min(range(n_jobs), key=lambda job: (loads[job], job))
        jobs[lightest].extend(component)
        loads[lightest] += len(component)
    return [sorted(job) for job in jobs]
