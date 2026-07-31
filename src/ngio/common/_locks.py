from dask.utils import SerializableLock

# Serialises zarr chunk read-modify-writes issued by da.store across all ngio
# dask write sites; block compute stays parallel. SerializableLock rather than
# threading.Lock so graphs survive pickling under a distributed client.
DASK_STORE_LOCK = SerializableLock()
