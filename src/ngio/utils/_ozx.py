"""OME-Zarr zip files (`.ozx`, OME-NGFF RFC-9).

One OME-Zarr hierarchy stored in a single ZIP archive. `OzxStore` is a zarr
`ZipStore` that keeps the archive conformant while ngio writes to it in place:

- a key never has two central directory records: the previous record is dropped
  before each write, so a rewritten `zarr.json` has exactly one record;
- deletes work: records are dropped, their bytes stay as dead space until
  `repack`;
- whenever the central directory is written (`flush`, `close`, garbage
  collection) every `zarr.json` record precedes every other record, root first
  then breadth-first, and the archive comment says so (`jsonFirst: true`).

The unit of work is the writing session. Appends and the central directory
rewrite cost O(entries) per `flush`/`close`; reclaiming dead space (`repack`)
costs a copy of the archive and is never automatic. One writer per archive:
a second writable store on the same path, or a pickled writable store, raises.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import struct
import threading
import time
import warnings
import weakref
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from zarr.core.buffer import Buffer
from zarr.storage import ZipStore

from ngio.utils._errors import NgioValueError
from ngio.utils._warnings import NgioUserWarning

if TYPE_CHECKING:
    import builtins
    from collections.abc import AsyncIterator, Iterable

    from zarr.abc.store import ByteRequest
    from zarr.core.buffer import BufferPrototype

logger = logging.getLogger(f"ngio:{__name__}")

OzxAccessMode = Literal["r", "r+", "w", "w-", "a"]

OZX_SUFFIX = ".ozx"
DEFAULT_OME_VERSION = "0.5"

_ZIP_MODES: dict[str, str] = {"r": "r", "r+": "a", "a": "a", "w": "w", "w-": "x"}
_META_NAME = "zarr.json"
_ALLOWED_EXTRA_IDS = {
    0x0001: "ZIP64",
    0x5455: "extended timestamp",
    0x7875: "Info-ZIP New Unix",
}


def _is_zarr_json(name: str) -> bool:
    return name == _META_NAME or name.endswith("/" + _META_NAME)


def _record_sort_key(name: str) -> tuple[int, int, str]:
    """Central directory order: root `zarr.json`, then breadth-first, then the rest."""
    return (0 if _is_zarr_json(name) else 1, name.count("/"), name)


def rfc9_comment(version: str = DEFAULT_OME_VERSION) -> bytes:
    """Return the RFC-9 archive comment for a `zarr.json`-first central directory."""
    doc = {
        "ome": {
            "version": version,
            "zipFile": {"centralDirectory": {"jsonFirst": True}},
        }
    }
    return json.dumps(doc, separators=(",", ":")).encode("utf-8")


class _OzxZipFile(zipfile.ZipFile):
    """`ZipFile` whose central directory is RFC-9 shaped whenever it is written.

    The ordering and the comment are applied in `close` rather than by the
    store, so an archive that is only closed by garbage collection (zipfile's
    `__del__`) is still conformant.
    """

    def ome_version(self) -> str:
        info = self.NameToInfo.get(_META_NAME)
        if info is None:
            return DEFAULT_OME_VERSION
        try:
            with self.open(info) as f:
                version = json.loads(f.read())["attributes"]["ome"]["version"]
        except (KeyError, TypeError, ValueError, zipfile.BadZipFile):
            return DEFAULT_OME_VERSION
        return str(version)

    def tombstone(self, name: str) -> bool:
        """Drop `name` from the directory. Its bytes stay until `repack`."""
        info = self.NameToInfo.pop(name, None)
        if info is None:
            return False
        self.filelist.remove(info)
        self._didModify = True
        return True

    def drop_duplicates(self) -> int:
        """Keep one record per name: the one a reader resolves (`NameToInfo`)."""
        live = [i for i in self.filelist if self.NameToInfo.get(i.filename) is i]
        dropped = len(self.filelist) - len(live)
        if dropped:
            self.filelist[:] = live
            self._didModify = True
        return dropped

    def _prepare_directory(self) -> None:
        self.filelist.sort(key=lambda info: _record_sort_key(info.filename))
        comment = rfc9_comment(self.ome_version())
        # Only when it differs: the setter flips `_didModify`, which would turn
        # every open/close of an untouched archive into a directory rewrite.
        if self.comment != comment:
            self.comment = comment

    def flush(self) -> None:
        """Write the central directory now and keep the archive open.

        Mirrors what `close` does: seek to the end of the last entry, write
        the directory and the end records, truncate. Later writes overwrite
        the directory again, exactly as the first append of a session does.
        """
        if self.mode not in ("w", "x", "a"):
            return
        if not self._didModify:
            return
        if self._writing:  # ty: ignore[unresolved-attribute]
            raise ValueError("Can't flush the ZIP file while a writing handle is open.")
        self._prepare_directory()
        fp = self.fp
        assert fp is not None
        with self._lock:  # ty: ignore[unresolved-attribute]
            fp.seek(self.start_dir)
            self._write_end_record()  # ty: ignore[unresolved-attribute]
            # `_write_end_record` truncates in mode "a" only. A directory that
            # shrank after deletes in mode "w" would otherwise leave a stale end
            # record behind the live one, and readers scan for it from the end.
            fp.truncate()
            fp.flush()
            os.fsync(fp.fileno())

    def close(self) -> None:
        if self.fp is not None and self.mode in ("w", "x", "a") and self._didModify:
            self._prepare_directory()
        super().close()


# ----------------------------------------------------------------------
# One writer per archive, and a safety net for stores nobody closes
# ----------------------------------------------------------------------

_open_writable: weakref.WeakValueDictionary[str, OzxStore] = (
    weakref.WeakValueDictionary()
)
_registry_lock = threading.Lock()


def _registry_key(path: Path) -> str:
    return os.fspath(path.resolve())


def _register_writable(store: OzxStore) -> None:
    key = _registry_key(store.path)
    with _registry_lock:
        other = _open_writable.get(key)
        if other is not None and other is not store:
            raise NgioValueError(
                f"{store.path} is already open for writing in this process. "
                "An OME-Zarr zip file has one writer: close() the other store "
                "(or the container holding it) first."
            )
        _open_writable[key] = store


def _unregister_writable(store: OzxStore) -> None:
    key = _registry_key(store.path)
    with _registry_lock:
        if _open_writable.get(key) is store:
            del _open_writable[key]


def _close_abandoned(zf: _OzxZipFile, path: Path) -> None:
    """Finalizer for a writable store nobody closed.

    Runs when the store is garbage-collected and, through `weakref.finalize`,
    at interpreter exit. This is the normal end of a session for code that
    opened an `.ozx` path without knowing it was one (a generic script or a
    Fractal task), so it is logged, not warned.
    """
    if zf.fp is None:
        return
    zf.close()
    logger.debug(f"Finalized the OME-Zarr zip file {path} on garbage collection.")


def _check_archive(zf: zipfile.ZipFile) -> None:
    """Refuse an existing archive that violates an RFC-9 MUST we rely on."""
    infos = zf.filelist
    if not infos:
        return
    problems = []
    compressed = [i.filename for i in infos if i.compress_type != zipfile.ZIP_STORED]
    if compressed:
        problems.append(
            f"{len(compressed)} entries are compressed (e.g. {compressed[0]!r}); "
            "all entries must use the STORE method"
        )
    encrypted = [i.filename for i in infos if i.flag_bits & 0x1]
    if encrypted:
        problems.append(f"{len(encrypted)} entries are encrypted")
    # The RFC bans data before the first local file header (prepended stubs).
    # An obsolete header at offset 0 still satisfies that: the RFC allows
    # headers the central directory no longer references, and rewriting the
    # root `zarr.json` leaves exactly such a header behind.
    fp = zf.fp
    assert fp is not None
    fp.seek(0)
    if fp.read(4) != b"PK\x03\x04":
        problems.append("no local file header at offset 0")
    if problems:
        raise NgioValueError(
            f"{zf.filename} is not an OME-Zarr zip file (RFC-9): "
            + "; ".join(problems)
            + ". Use `zarr.storage.ZipStore` directly for arbitrary zip archives."
        )


class OzxStore(ZipStore):
    """A zarr store for an OME-Zarr zip file (`.ozx`, RFC-9).

    Modes follow ngio's: `"r"` read-only, `"r+"`/`"a"` append to an existing
    archive (created if missing), `"w"` truncate, `"w-"` create or fail.

    The archive on disk is a complete, readable OME-Zarr zip file after every
    `flush` and after `close`. ngio flushes whenever the hierarchy changes
    (metadata written, groups created or deleted); array chunks written since
    the last flush become visible at the next flush or at `close`. Dead space
    left by overwrites and deletes is only reclaimed by `repack`.

    Raises:
        NgioValueError: On a second writable store for the same archive in
            this process, on pickling a writable store, on IO after `close`,
            and when an existing archive breaks an RFC-9 MUST (compression,
            encryption, data before the first entry).
    """

    supports_deletes: bool = True

    _zf: _OzxZipFile
    _ngio_mode: OzxAccessMode
    _closed: bool
    _view_of: OzxStore | None
    _finalizer: weakref.finalize | None

    def __init__(self, path: Path | str, *, mode: OzxAccessMode = "r") -> None:
        if mode not in _ZIP_MODES:
            raise NgioValueError(
                f"Unsupported mode {mode!r}; expected one of {sorted(_ZIP_MODES)}."
            )
        path = Path(path)
        if path.suffix != OZX_SUFFIX:
            warnings.warn(
                f"OME-Zarr zip files should end with '{OZX_SUFFIX}' (RFC-9), "
                f"got {path.name!r}.",
                NgioUserWarning,
                stacklevel=2,
            )
        super().__init__(
            path,
            mode=cast("Any", _ZIP_MODES[mode]),
            read_only=mode == "r",
            compression=zipfile.ZIP_STORED,
            allowZip64=True,
        )
        self._ngio_mode = mode
        self._closed = False
        self._view_of = None
        self._finalizer = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def mode(self) -> OzxAccessMode:
        """Return the ngio access mode the store was opened with."""
        return self._ngio_mode

    @property
    def closed(self) -> bool:
        """Whether `close` has run on this store (or the store it views)."""
        return (self._view_of or self)._closed

    def _sync_open(self) -> None:
        if self._is_open:
            raise ValueError("store is already open")
        if self._closed:
            # Stock `ZipStore` would silently start a new writing session here.
            raise NgioValueError(
                f"The store for {self.path} is closed. Open the archive again "
                "instead of reusing a closed store."
            )
        if not self.read_only:
            _register_writable(self)
        self._lock = threading.RLock()
        try:
            self._zf = _OzxZipFile(
                self.path,
                mode=cast("Any", self._zmode),
                compression=self.compression,
                allowZip64=self.allowZip64,
            )
        except BaseException:
            if not self.read_only:
                _unregister_writable(self)
            raise
        if self._zmode == "x":
            # Exclusive creation happened; everything from here on is an append.
            self._zmode = "a"
        _check_archive(self._zf)
        if not self.read_only:
            # An archive written by a stock `ZipStore` may carry duplicate
            # records; readers resolve the last one, keep exactly that.
            self._zf.drop_duplicates()
            self._finalizer = weakref.finalize(
                self, _close_abandoned, self._zf, self.path
            )
        self._is_open = True

    def _ensure_live(self) -> None:
        root = self._view_of or self
        if root._closed:
            raise NgioValueError(
                f"The store for {self.path} is closed. Open the archive again "
                "instead of reusing a closed store."
            )
        if not root._is_open:
            root._sync_open()
        if root is not self:
            self._zf = root._zf
            self._lock = root._lock
            self._is_open = True

    def flush(self) -> None:
        """Write the central directory, keeping the writing session open.

        A no-op on read-only stores, views, closed stores and when nothing
        was written since the last flush.
        """
        if self._view_of is not None or self.read_only or self._closed:
            return
        if not self._is_open:
            return
        with self._lock:
            self._zf.flush()

    def close(self) -> None:
        """End the writing session: write the central directory and close.

        Idempotent. Views (`with_read_only`) never own the archive and ignore
        `close`. After `close`, every IO call on this store raises.
        """
        if self._view_of is not None or self._closed:
            return
        self._closed = True
        if self._finalizer is not None:
            self._finalizer.detach()
            self._finalizer = None
        if self._is_open:
            self._is_open = False
            with self._lock:
                self._zf.close()
        if not self.read_only:
            _unregister_writable(self)

    def repack(self) -> None:
        """Rewrite the archive with only its live records, reclaiming dead space.

        Costs a full copy of the archive (written beside it, then renamed over
        it). The writing session continues afterwards.
        """
        self._check_writable()
        self._ensure_live()
        with self._lock:
            self._zf.flush()
            tmp = self.path.with_name(self.path.name + ".repack")
            try:
                with _OzxZipFile(
                    tmp, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True
                ) as dst:
                    for info in list(self._zf.filelist):
                        new = zipfile.ZipInfo(info.filename, date_time=info.date_time)
                        new.compress_type = zipfile.ZIP_STORED
                        new.external_attr = info.external_attr
                        # Known up front, so zipfile picks ZIP64 for big entries.
                        new.file_size = info.file_size
                        with self._zf.open(info) as src, dst.open(new, mode="w") as out:
                            shutil.copyfileobj(src, out, 1024 * 1024)
                self._zf.close()
                os.replace(tmp, self.path)
            except BaseException:
                tmp.unlink(missing_ok=True)
                raise
            self._zmode = "a"
            self._zf = _OzxZipFile(
                self.path,
                mode="a",
                compression=zipfile.ZIP_STORED,
                allowZip64=True,
            )
            if self._finalizer is not None:
                self._finalizer.detach()
            self._finalizer = weakref.finalize(
                self, _close_abandoned, self._zf, self.path
            )

    def with_read_only(self, read_only: bool = False) -> OzxStore:
        """Return a read-only view sharing this store's open archive.

        A second file handle would not see entries appended in the live
        session (the central directory is only on disk after a flush), so
        the view shares the handle instead. zarr calls this whenever a group
        on a writable store is reopened with `mode="r"`.
        """
        if read_only == self.read_only:
            return self
        if not read_only:
            raise NgioValueError(
                "Cannot make a read-only .ozx store writable; open the archive "
                "with mode='r+' instead."
            )
        self._ensure_live()
        # Not `copy.copy`: that goes through `__getstate__`, which refuses
        # writable stores on purpose.
        view = type(self).__new__(type(self))
        view.__dict__.update(self.__dict__)
        view._read_only = True
        view._view_of = self._view_of or self
        view._finalizer = None
        return view

    async def clear(self) -> None:
        self._check_writable()
        self._ensure_live()
        with self._lock:
            self._zf.close()
            os.remove(self.path)
            self._zmode = "w"
            self._zf = _OzxZipFile(
                self.path, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True
            )
            if self._finalizer is not None:
                self._finalizer.detach()
            self._finalizer = weakref.finalize(
                self, _close_abandoned, self._zf, self.path
            )

    async def move(self, path: Path | str) -> None:
        raise NgioValueError(
            "OzxStore.move is not supported: close() the store and move the file."
        )

    # ------------------------------------------------------------------
    # Pickling: a writable store is one writing session, it cannot be copied
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict[str, Any]:
        if not self.read_only or self._view_of is not None:
            raise NgioValueError(
                f"A writable OzxStore ({self.path}) cannot be pickled: every "
                "unpickled copy would open its own writing session on the archive "
                "and the last one to close would discard the others. Open the "
                "archive read-only for parallel reads, and write from one process."
            )
        state = super().__getstate__()
        state["_finalizer"] = None
        state["_view_of"] = None
        return state

    def __repr__(self) -> str:
        return f"OzxStore('{self.path}', mode='{self._ngio_mode}')"

    # ------------------------------------------------------------------
    # IO
    # ------------------------------------------------------------------

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        self._ensure_live()
        with self._lock:
            return self._get(key, prototype=prototype, byte_range=byte_range)

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> builtins.list[Buffer | None]:
        self._ensure_live()
        with self._lock:
            return [
                self._get(key, prototype=prototype, byte_range=byte_range)
                for key, byte_range in key_ranges
            ]

    def _set(self, key: str, value: Buffer) -> None:
        with self._lock:
            self._zf.tombstone(key)
            info = zipfile.ZipInfo(
                filename=key, date_time=time.localtime(time.time())[:6]
            )
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o644 << 16
            self._zf.writestr(info, value.to_bytes())

    async def set(self, key: str, value: Buffer) -> None:
        self._check_writable()
        self._ensure_live()
        if not isinstance(value, Buffer):
            raise TypeError(
                "OzxStore.set(): `value` must be a Buffer instance. "
                f"Got an instance of {type(value)} instead."
            )
        with self._lock:
            self._set(key, value)

    async def set_if_not_exists(self, key: str, value: Buffer) -> None:
        self._check_writable()
        self._ensure_live()
        with self._lock:
            if key not in self._zf.NameToInfo:
                self._set(key, value)

    async def delete(self, key: str) -> None:
        self._check_writable()
        self._ensure_live()
        with self._lock:
            self._zf.tombstone(key)

    async def delete_dir(self, prefix: str) -> None:
        self._check_writable()
        self._ensure_live()
        if prefix != "" and not prefix.endswith("/"):
            prefix += "/"
        with self._lock:
            for name in [n for n in self._zf.NameToInfo if n.startswith(prefix)]:
                self._zf.tombstone(name)

    async def exists(self, key: str) -> bool:
        self._ensure_live()
        with self._lock:
            return key in self._zf.NameToInfo

    async def list(self) -> AsyncIterator[str]:
        self._ensure_live()
        with self._lock:
            keys = self._zf.namelist()
        for key in keys:
            yield key

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        async for key in self.list():
            if key.startswith(prefix):
                yield key

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        prefix = prefix.rstrip("/")
        self._ensure_live()
        with self._lock:
            keys = self._zf.namelist()
        seen: set[str] = set()
        for key in keys:
            if prefix == "":
                child = key.split("/")[0]
            elif key.startswith(prefix + "/") and key.strip("/") != prefix:
                child = key.removeprefix(prefix + "/").split("/")[0]
            else:
                continue
            if child not in seen:
                seen.add(child)
                yield child


# ----------------------------------------------------------------------
# Conformance check (central directory only, works over range requests)
# ----------------------------------------------------------------------


@dataclass
class OzxReport:
    """Result of `check_ozx`: RFC-9 MUST violations and SHOULD notices."""

    path: str
    must: list[str] = field(default_factory=list)
    should: list[str] = field(default_factory=list)
    entries: int = 0
    zip64: bool = False
    json_first: bool = False
    duplicates: list[str] = field(default_factory=list)
    comment: dict[str, Any] | None = None

    @property
    def conforming(self) -> bool:
        """Whether no MUST requirement is violated."""
        return not self.must

    def __str__(self) -> str:
        lines = [f"== {self.path}"]
        lines += [f"  MUST violated : {m}" for m in self.must]
        lines += [f"  SHOULD/notice : {s}" for s in self.should]
        lines.append(
            f"  info          : {self.entries} entries, zip64={self.zip64}, "
            f"jsonFirst={self.json_first}"
        )
        verdict = "NOT CONFORMING" if self.must else "no MUST violations found"
        lines.append(f"  RESULT        : {verdict}")
        return "\n".join(lines)


_EOCD = struct.Struct("<4s4H2LH")
_EOCD64_LOC = struct.Struct("<4sLQL")
_EOCD64 = struct.Struct("<4sQ2H2L4Q")
_CDIR = struct.Struct("<4s6H3L5H2L")


def _open_for_check(location: str) -> Any:
    if "://" in location:
        import fsspec

        return fsspec.open(location, mode="rb").open()
    return open(location, "rb")


def check_ozx(path: str | Path) -> OzxReport:
    """Check an OME-Zarr zip file against the RFC-9 MUST and SHOULD rules.

    Reads only the end of the file and the central directory, so a URL costs
    a few range requests, never the entries themselves. Mirrors the reference
    checker shipped with the RFC.

    Args:
        path: A local path or a URL (`https://`, `s3://`, ...).
    """
    location = os.fspath(path)
    report = OzxReport(path=location)
    with _open_for_check(location) as f:
        size = f.seek(0, os.SEEK_END)
        tail_len = min(size, 65536 + 22 + 20 + 56)
        f.seek(size - tail_len)
        tail = f.read(tail_len)
        i = tail.rfind(b"PK\x05\x06")
        if i < 0:
            report.must.append("no end of central directory record")
            return report
        _, disk, cd_disk, _, n_total, cd_size, cd_off, comment_len = _EOCD.unpack(
            tail[i : i + 22]
        )
        raw_comment = tail[i + 22 : i + 22 + comment_len]
        j = tail.rfind(b"PK\x06\x07", 0, i)
        report.zip64 = j >= 0
        if n_total == 0xFFFF or cd_size == 0xFFFFFFFF or cd_off == 0xFFFFFFFF:
            _, _, rec_off, _ = _EOCD64_LOC.unpack(tail[j : j + 20])
            f.seek(rec_off)
            rec = f.read(56)
            (_, _, _, _, disk, cd_disk, _, n_total, cd_size, cd_off) = _EOCD64.unpack(
                rec
            )
        f.seek(cd_off)
        cd = f.read(cd_size)
        f.seek(0)
        head = f.read(4)

    names: list[str] = []
    methods: set[int] = set()
    extras: dict[int, str] = {}
    encrypted = multi_disk = False
    lowest = None
    pos = 0
    while pos + 46 <= len(cd):
        fields = _CDIR.unpack_from(cd, pos)
        if fields[0] != b"PK\x01\x02":
            report.must.append("bad central directory record signature")
            break
        flags, method = fields[3], fields[4]
        csize, usize = fields[8], fields[9]
        nlen, elen, file_comment_len, dstart = (
            fields[10],
            fields[11],
            fields[12],
            fields[13],
        )
        off = fields[16]
        name = cd[pos + 46 : pos + 46 + nlen].decode("utf-8")
        ex = cd[pos + 46 + nlen : pos + 46 + nlen + elen]
        k = 0
        while k + 4 <= len(ex):
            hid, hsz = struct.unpack_from("<HH", ex, k)
            if hid == 0x0001 and off == 0xFFFFFFFF:
                p = k + 4 + (8 if usize == 0xFFFFFFFF else 0)
                p += 8 if csize == 0xFFFFFFFF else 0
                off = struct.unpack_from("<Q", ex, p)[0]
            if hid not in _ALLOWED_EXTRA_IDS:
                extras.setdefault(hid, name)
            k += 4 + hsz
        names.append(name)
        methods.add(method)
        encrypted = encrypted or bool(flags & 1)
        multi_disk = multi_disk or bool(dstart)
        lowest = off if lowest is None else min(lowest, off)
        pos += 46 + nlen + elen + file_comment_len
    report.entries = len(names)

    if _META_NAME not in names:
        report.must.append("root-level zarr.json is missing")
    if methods - {0}:
        report.must.append(f"entries are not STORE (methods {sorted(methods - {0})})")
    if encrypted:
        report.must.append("encrypted entries")
    for hid, name in extras.items():
        report.must.append(f"disallowed extra field 0x{hid:04x} (first seen on {name})")
    if disk or cd_disk or multi_disk:
        report.must.append("multi-volume archive")
    if head != b"PK\x03\x04":
        report.must.append("no local file header at offset 0")
    elif lowest:
        # Obsolete headers may stay behind after in-place modification; the
        # RFC's reference checker reports this as a MUST violation because it
        # only sees the central directory. `repack` removes them.
        report.should.append(
            f"{lowest} bytes of obsolete entries before the first live record "
            "(the RFC reference checker flags this; repack() reclaims it)"
        )

    if not report.zip64:
        report.should.append("ZIP64 end of central directory records are not present")
    seen: set[str] = set()
    dups: set[str] = set()
    for n in names:
        (dups if n in seen else seen).add(n)
    report.duplicates = sorted(dups)
    if dups:
        report.should.append(
            f"{len(dups)} duplicate names in the central directory "
            f"(e.g. {report.duplicates[0]!r})"
        )
    is_json = [_is_zarr_json(n) for n in names if not n.endswith("/")]
    ordered = all(is_json[k] >= is_json[k + 1] for k in range(len(is_json) - 1))
    report.json_first = ordered

    comment_ok = False
    claims_json_first = False
    try:
        doc = json.loads(raw_comment.decode("utf-8")) if raw_comment else None
    except ValueError:
        doc = None
    if isinstance(doc, dict):
        report.comment = doc
        ome = doc.get("ome")
        comment_ok = isinstance(ome, dict) and isinstance(ome.get("version"), str)
        if isinstance(ome, dict):
            cdir = (ome.get("zipFile") or {}).get("centralDirectory") or {}
            claims_json_first = bool(cdir.get("jsonFirst"))
    if not comment_ok:
        report.should.append(
            "archive comment is missing or is not JSON with ome.version"
        )
    if claims_json_first and not ordered:
        report.should.append(
            "comment says jsonFirst: true but zarr.json records do not all "
            "precede other records"
        )
    if not location.split("?")[0].endswith(OZX_SUFFIX):
        report.should.append(f"file name does not end with {OZX_SUFFIX}")
    return report
