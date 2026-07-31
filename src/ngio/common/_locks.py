"""Locks shared across ngio's write paths."""

from dask.utils import SerializableLock

# Serialises the store flushes of every `da.store` call in ngio.
#
# `da.store(..., lock=False)` gives each dask block its own write task. A block
# whose footprint does not align with the target's write unit — the chunk, or
# the shard when the array is sharded — makes zarr read-modify-write that unit,
# so two such blocks racing on the same key silently lose one update. Holding
# this lock around the flush serialises those writes; block compute stays
# parallel.
#
# `SerializableLock` rather than `threading.Lock` so the graph still pickles
# under a distributed client.
DASK_STORE_LOCK = SerializableLock()
