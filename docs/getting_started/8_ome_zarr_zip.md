---
description: "OME-Zarr zip files (.ozx, RFC-9): create, edit in place, check, repack, read remotely, and move hierarchies in and out of an archive."
---

# 8. OME-Zarr zip files

**One file instead of thousands.**

An OME-Zarr image is a directory tree with one file per chunk, which is the right
shape for object storage and parallel writers and the wrong shape for an email
attachment, a shared drive or a filesystem with an inode quota. [OME-NGFF
RFC-9](https://ngff.openmicroscopy.org/rfc/9/) packs the whole tree, uncompressed,
into a single ZIP archive with the extension `.ozx`. Readers find every entry through
the ZIP central directory, so a chunk is read with one seek, locally or over HTTP
range requests.

ngio works on `.ozx` files natively: anywhere a path is accepted, a path ending in
`.ozx` opens the archive. Nothing else in a script has to know it is a zip.

!!! note "OME-Zarr 0.5 only"
    RFC-9 requires the root `zarr.json` at the archive root, which only Zarr v3
    (OME-Zarr 0.5 and later) has. Creating a 0.4 image inside an `.ozx` raises, so pass
    `ngff_version="0.5"` to the creation functions.

## Create and edit in place

Point a creation function at an `.ozx` path:

```python exec="true" source="material-block" session="ozx"
--8<-- "docs/snippets/getting_started/ome_zarr_zip.py:create"
```

Reopen it and edit, exactly as you would a directory. Metadata changes, new labels,
deleted tables and `overwrite=True` all work in place:

```python exec="true" source="material-block" session="ozx"
--8<-- "docs/snippets/getting_started/ome_zarr_zip.py:edit"
```

### What happens on disk

A ZIP archive is a sequence of entries followed by a central directory that lists them.
ngio appends new entries and, when an entry is rewritten, drops the old record from the
directory so no duplicate names remain. The directory itself is rewritten at the end of
every call that writes (metadata, a group created or deleted, `set_array`, `set_roi`,
`consolidate`, a table), so whenever your code is between ngio calls the file on disk is
a complete, readable archive, in this process and in any other. The one window where it
is not is the write itself: appending to a zip overwrites the directory, so a process
that dies in the middle of a `set_array` leaves an archive without one, and ngio cannot
recover it. Keep a copy of anything precious, and for heavy or parallel writes work on a
directory or in memory and pack the result with `to_ozx`.

An archive has one writer per process. Opening a path that another container in the
same process already holds for writing does not open the file again: the new container
joins the live session (read-only if you asked for `mode="r"`), and `overwrite=True`
empties it in place. Create, then reopen by path, keeps working as it does for a
directory.

Overwritten entries leave their old bytes behind: that is the "obsolete entries" notice
in the report below, and `repack` (further down) is the remedy. `check_ozx` reports the
state of an archive against the RFC:

```python exec="true" source="material-block" session="ozx"
--8<-- "docs/snippets/getting_started/ome_zarr_zip.py:check"
```

## An explicit session

For heavy writes, or when you need to know exactly when the archive is finished, open
the store yourself. The `with` block is the writing session: everything inside shares
one file handle, code that takes the path joins the same session, and the archive is
finalized on exit. A session cannot start on an archive that containers opened by path
still hold, so open it first or on a file of its own.

```python exec="true" source="material-block" session="ozx"
--8<-- "docs/snippets/getting_started/ome_zarr_zip.py:session"
```

After the block, every container opened inside it is closed with the store. Inside it,
`store.flush()` publishes what has been written so far to readers with their own
handle; ngio calls it at the end of every write verb, so you only need it around raw
zarr writes. A second explicit writer on the same archive raises, and so does pickling a
writable store, so an `.ozx` cannot be the target of a dask cluster. For parallel writes
work on a directory or in memory and pack at the end (below).

!!! tip "Shard when the archive is the destination"
    The RFC recommends the sharding codec for zip files: with one entry per chunk a
    large image means a central directory of tens of thousands of records, which every
    reader parses before the first byte of data. Pass `shards=` to the creation
    functions when an `.ozx` is the target.

### Reclaim dead space

Rewriting an entry appends the new version and abandons the old bytes, so editing the
pixels of an existing level grows the file by the rewritten size until the next
`repack`, which rewrites the archive with only its live entries:

```python exec="true" source="material-block" session="ozx"
--8<-- "docs/snippets/getting_started/ome_zarr_zip.py:repack"
```

## Pack, unpack, and work in memory

`to_memory`, `from_memory`, `to_ozx` and `from_ozx` copy a whole hierarchy between
stores, tables included. This is the explicit extract, modify, repack workflow the RFC
recommends for workloads that are a poor fit for in-place editing: load the archive into
RAM, work on it with any store-agnostic code, and write it back in one pass. `to_memory`
holds every level and every table in RAM, and for a URL it downloads the whole archive
first. A derive followed by a full pyramid build is cheaper on a directory or in memory
than in place, since every rewritten level stays in the file until `repack`; build it
there and `to_ozx` at the end.

```python exec="true" source="material-block" session="ozx"
--8<-- "docs/snippets/getting_started/ome_zarr_zip.py:transfer"
```

`to_ozx` writes a closed, conformant archive with no dead space, so it is also the way
to produce a clean file from a directory at the end of a pipeline.

## Remote archives

A remote `.ozx` opens read-only through a URL. Only the central directory and the
chunks you read are fetched, with HTTP range requests:

```python
from ngio import open_ome_zarr_container

ome_zarr = open_ome_zarr_container(
    "https://static.webknossos.org/misc/6001240.ozx", mode="r"
)
```

A zip cannot be modified in place on an object store, so a writable mode on a URL
raises. To publish an archive, pack it and upload: `to_ozx(src, "s3://bucket/image.ozx")`
packs into a local temporary file and uploads it.

## Next steps

- [OME-Zarr containers](1_ome_zarr_containers.md) — everything above applies unchanged.
- [HCS plates](5_hcs.md) — a plate can live in an `.ozx` too.
