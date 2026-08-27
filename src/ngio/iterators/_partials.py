"""Per-job partial tables for distributed read-only iterators.

Read-only iterators (features, detection) end in a *global join* — one
coalesce, one NMS pass — that independent jobs cannot reproduce piecewise.
Their distributed form therefore stores each job's **raw pre-join records**
here, in a scratch group beside the multiscale levels, and the gather step
(`finalize()`) reconstructs the full record set and runs the single global
join: bit-for-bit the serial answer, custom joins included.

The scratch group (`_ngio_partials`) plays the role the stitch scratch plays
for stitched segmentations: transient per-job evidence, invisible to
`list_tables`, created race-free by the init step (`prepare_jobs`, which
always wipes leftovers from earlier runs first), written one subgroup per job
(sibling creation touches no shared attributes — the `tables` list is never
involved, so there is no read-modify-write race), and deleted by the merge.
Partial tables are written through ngio's own table machinery
(`GenericTable` + the table backends on a `ZarrGroupHandler`), so every store
type and the retry policy are inherited.

Schema fidelity: concatenating per-unit frames unions their columns and
NaN-fills the gaps (upcasting ints to floats), so each job also records every
unit's own column list and dtypes in its attrs, and the gather subsets and
`astype`s each unit's rows back to exactly what the function returned. The
one residual is backend-level: the anndata backend packs mixed float columns
into one `X` matrix, so a `float32` column comes back `float64`.
"""

from typing import Any, NamedTuple

import pandas as pd

from ngio.tables.v1._generic_table import GenericTable
from ngio.utils import NgioFileNotFoundError, NgioValueError, ZarrGroupHandler

_PARTIALS_GROUP = "_ngio_partials"

#: Column carrying each record's global unit index inside a partial table.
#: Reserved: a function whose result already carries it is refused.
INDEX_COLUMN = "_ngio_index"


def delete_partials_root(handler: ZarrGroupHandler) -> None:
    """Remove the partials group, if it is there."""
    try:
        handler.delete_group(_PARTIALS_GROUP)
    except (KeyError, FileNotFoundError):
        pass


def create_partials_root(handler: ZarrGroupHandler, attrs: dict[str, Any]) -> None:
    """Create the partials group fresh, wiping any leftovers first.

    The one moment a distributed run may destroy partials: stale or corrupted
    temp tables from a crashed, interrupted, or superseded run are wiped
    here, so nothing older than this call can ever reach a merge.
    """
    delete_partials_root(handler)
    root = handler.get_handler(path=_PARTIALS_GROUP, overwrite=True)
    root.write_attrs(attrs)


def read_partials_root_attrs(handler: ZarrGroupHandler) -> dict[str, Any] | None:
    """The partials group's attributes, or `None` when no group exists."""
    try:
        root = handler.get_handler(path=_PARTIALS_GROUP, create_mode=False)
    except NgioFileNotFoundError:
        # Only "no group" means "not prepared"; any other failure (an
        # unreadable store, a group that is an array) must surface.
        return None
    return root.load_attrs()


def prepare_partials(iterator: Any, handler: ZarrGroupHandler, n_jobs: int) -> None:
    """The init-step setup for a read-only distributed run."""
    create_partials_root(
        handler,
        attrs={
            "fingerprint": iterator._plan_fingerprint(n_jobs),
            "n_jobs": n_jobs,
        },
    )


def validate_partials_context(
    iterator: Any, handler: ZarrGroupHandler, n_jobs: int
) -> None:
    """Require a prepared, matching partials root before a job may run."""
    attrs = read_partials_root_attrs(handler)
    if attrs is None:
        raise NgioValueError(
            "A distributed run of a read-only iterator needs "
            "`prepare_jobs(n_jobs)` before `for_job`: the init step creates "
            "the partials group the jobs write into."
        )
    if attrs.get("n_jobs") != n_jobs or attrs.get(
        "fingerprint"
    ) != iterator._plan_fingerprint(n_jobs):
        raise NgioValueError(
            "The partials group was prepared for a different plan: the "
            "tiling, halo, or n_jobs changed since `prepare_jobs`. Re-run "
            "`prepare_jobs(n_jobs)` with the current iterator, then resubmit "
            "the jobs."
        )


def write_partial(
    handler: ZarrGroupHandler,
    job_index: int,
    frame: pd.DataFrame | None,
    attrs: dict[str, Any],
) -> None:
    """Store one job's raw records under its own subgroup.

    `frame` is the job's records tagged with `INDEX_COLUMN` (or `None` when
    the job produced no rows — stored as attributes only, sidestepping the
    empty-table edge cases of the backends). Re-running a job overwrites its
    own partial idempotently; no other job's subgroup is touched.
    """
    job = handler.get_handler(path=f"{_PARTIALS_GROUP}/job_{job_index}", overwrite=True)
    if frame is not None and len(frame):
        table = GenericTable(table_data=frame.reset_index(drop=True))
        table.set_backend(handler=job)
        table.consolidate()
        job.write_attrs({**attrs, "empty": False})
    else:
        job.write_attrs({**attrs, "empty": True})


def read_partial(
    handler: ZarrGroupHandler, job_index: int
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    """One job's attributes and records; records are `None` for an empty job."""
    job = handler.get_handler(
        path=f"{_PARTIALS_GROUP}/job_{job_index}", create_mode=False
    )
    attrs = job.load_attrs()
    if attrs.get("empty", False):
        return attrs, None
    table = GenericTable.from_handler(handler=job)
    return attrs, table.dataframe


class MergedPartials(NamedTuple):
    """A complete distributed run's merged records plus per-unit schemas."""

    #: Every job's records concatenated, or `None` when every job was empty.
    frame: pd.DataFrame | None
    #: Per unit index: the (column list, {column: dtype}) the function's
    #: normalized frame actually had, as recorded by the job that ran it.
    columns: dict[int, tuple[list[str], dict[str, str]]]


def unit_columns_record(index: int, frame: pd.DataFrame) -> list[Any]:
    """One unit's schema entry for the job attrs (JSON: a plain list)."""
    return [
        index,
        list(frame.columns),
        {column: str(dtype) for column, dtype in frame.dtypes.items()},
    ]


def merge_partial_frames(
    iterator: Any,
    handler: ZarrGroupHandler,
    *,
    job_verb: str,
) -> MergedPartials:
    """Validate a complete distributed run and return its merged records.

    Refuses loudly when the merge could silently lie: no partials at all
    (the init step or the jobs never ran), a fingerprint that no longer
    matches the iterator (the tiling changed between phases), or a missing
    job (a half-finished run must error, not produce a plausible-looking,
    silently incomplete table). Extra subgroups outside the expected job set
    can only be empty-partition writes and are ignored.

    Returns the concatenation of every job's records, stably sorted by
    `INDEX_COLUMN` (intra-unit row order preserved) — `None` when every job
    was empty — together with the per-unit schema records (see the module
    docstring). `job_verb` is the caller's per-job topic verb, used in
    error messages.
    """
    from ngio.iterators._mappers import write_conflict_components
    from ngio.iterators._split import partition_components

    root_attrs = read_partials_root_attrs(handler)
    if root_attrs is None:
        raise NgioValueError(
            "No partials to merge: run `prepare_jobs(n_jobs)`, then "
            f"`for_job(**args).{job_verb}(func)` per job, before "
            "`finalize()`."
        )
    n_jobs = root_attrs.get("n_jobs")
    if not isinstance(n_jobs, int) or root_attrs.get(
        "fingerprint"
    ) != iterator._plan_fingerprint(n_jobs):
        raise NgioValueError(
            "The partials were prepared for a different plan: the tiling, "
            "halo, or n_jobs changed since `prepare_jobs`. Re-run "
            "`prepare_jobs(n_jobs)` and the jobs with the current iterator."
        )

    units = list(iterator._numpy_units_generator())
    partition = partition_components(write_conflict_components(units), n_jobs)
    expected = [index for index, indices in enumerate(partition) if indices]

    frames: list[pd.DataFrame] = []
    columns: dict[int, tuple[list[str], dict[str, str]]] = {}
    missing: list[int] = []
    for job_index in expected:
        try:
            attrs, frame = read_partial(handler, job_index)
        except NgioFileNotFoundError:
            # Only an absent job subgroup counts as "not run yet"; a partial
            # that exists but cannot be read is corruption and must raise.
            missing.append(job_index)
            continue
        if attrs.get("fingerprint") != root_attrs.get("fingerprint"):
            missing.append(job_index)
            continue
        for index, cols, dtypes in attrs.get("columns", []):
            columns[int(index)] = (list(cols), dict(dtypes))
        if frame is not None:
            frames.append(frame)
    if missing:
        raise NgioValueError(
            f"The distributed run is incomplete: job(s) {missing} of "
            f"{n_jobs} have no (matching) partial. Merging now would return "
            "a silently incomplete table — run the missing jobs first."
        )

    if not frames:
        return MergedPartials(None, columns)
    merged = pd.concat(frames, ignore_index=True)
    return MergedPartials(
        merged.sort_values(INDEX_COLUMN, kind="stable", ignore_index=True), columns
    )
