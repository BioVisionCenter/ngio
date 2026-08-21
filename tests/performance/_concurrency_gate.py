"""Deterministic concurrency instrument for the performance gate.

Op counts are invariant to concurrency: 218 serial round-trips and 218
concurrent ones tally identically, so the op-count gate is structurally blind
to a parallelism regression. This module measures the missing axis without a
single wall-clock assertion: a rendezvous store parks every `get`/`set` on
zarr's IO event loop until `k` of them are in flight *together*, and a gauge
records the maximum overlap ever reached.

Why this is exact rather than probabilistic: every ngio worker thread reaches
the store through zarr's sync bridge, which submits the coroutine to one
global IO loop and blocks the calling thread until it returns. So one thread
has at most one op in flight, and k ops in flight ⇔ k concurrent submitters.
With working concurrency the k-th arrival releases everyone — fast, no sleeps.
With a serial regression the first op waits out one bounded timeout, the gate
gives up, and the gauge (stuck below k) fails the assertion legibly.

The probe is a module global gated by a context manager — the same design as
`_counting._ACTIVE`, for the same reasons (stores pickled into dask graphs
carry no state; worker threads all see one probe), plus one more:
`NgioStore._with_store` forwards only `retry`, so per-instance probe state
would silently vanish on the first `with_read_only()` reopen.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from typing import TYPE_CHECKING, Literal

from tests.performance._counting import CountingNgioStore, _kind

if TYPE_CHECKING:
    from collections.abc import Iterator

    from zarr.abc.store import ByteRequest
    from zarr.core.buffer import Buffer, BufferPrototype

#: Paid only when concurrency has regressed (or was never expected): the first
#: parked op waits this long, then the gate gives up and lets the run finish
#: serially so the gauge assertion can fail with the observed maximum.
_TIMEOUT_S = 2.0


class ConcurrencyProbe:
    """Gauge plus optional rendezvous over the store ops inside the block.

    `kind` selects which ops participate ("chunk", "meta", or "all"). This is
    not cosmetic: an operation's serial *prefix* — `get_as_numpy` reads two
    metadata documents before fanning out over chunks — would otherwise park
    alone, burn the one timeout, and disarm the rendezvous before the ops
    under test arrive. Scoping the probe to the kind whose overlap is being
    asserted keeps the measurement exact.
    """

    def __init__(self, rendezvous: int | None, kind: str) -> None:
        self.rendezvous = rendezvous
        self.kind = kind
        #: Highest number of ops ever in flight at once.
        self.max_in_flight = 0
        #: Total ops that entered the probe. Asserted `> 0` by every test: a
        #: zarr that reroutes IO through the 3.3 sync surface (executor
        #: threads, where nothing can park on the loop) must fail the test
        #: naming the surface move, not pass it vacuously.
        self.arrivals = 0
        self._current = 0
        self._lock = threading.Lock()
        # Created lazily: it must belong to zarr's I/O event loop.
        self._release: asyncio.Event | None = None

    def _enter(self) -> None:
        with self._lock:
            self.arrivals += 1
            self._current += 1
            self.max_in_flight = max(self.max_in_flight, self._current)

    def _exit(self) -> None:
        with self._lock:
            self._current -= 1

    async def _hold(self) -> None:
        """Park until `rendezvous` ops are in flight, once.

        After the rendezvous fires (or times out) the event stays set, so
        every later op passes straight through — the block measures whether
        the concurrency *exists*, not every moment of it.
        """
        if self.rendezvous is None:
            return
        if self._release is None:
            self._release = asyncio.Event()
        if self._release.is_set():
            return
        with self._lock:
            reached = self._current >= self.rendezvous
        if reached:
            self._release.set()
            return
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._release.wait(), timeout=_TIMEOUT_S)
        # Give up for everyone: the gauge already tells the story, and a
        # serial run must not pay one timeout per op.
        self._release.set()


_PROBE: ConcurrencyProbe | None = None


@contextlib.contextmanager
def concurrency_probe(
    rendezvous: int | None = None,
    kind: Literal["chunk", "meta", "all"] = "all",
) -> Iterator[ConcurrencyProbe]:
    """Activate a probe for the block.

    Args:
        rendezvous: Park participating ops until this many are in flight
            together, then release them all. `None` only gauges — a serial
            control costs nothing and asserts `max_in_flight == 1`
            structurally.
        kind: Which store keys participate; see `ConcurrencyProbe`.

    Raises:
        RuntimeError: If a probe is already active; nesting would attribute
            inner ops to both.
    """
    global _PROBE
    if _PROBE is not None:
        raise RuntimeError("concurrency_probe() blocks cannot be nested")
    probe = ConcurrencyProbe(rendezvous, kind)
    _PROBE = probe
    try:
        yield probe
    finally:
        _PROBE = None


class RendezvousNgioStore(CountingNgioStore):
    """A store whose `get`/`set` report to — and park on — the active probe.

    Subclasses `CountingNgioStore` so `from_any` normalization survival and
    the whole counted surface come for free; with no `count()` block active
    the counting hooks are no-ops, and with no probe active this class is
    behaviorally identical to its parent.
    """

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        probe = _PROBE
        if probe is None or probe.kind not in ("all", _kind(key)):
            return await super().get(key, prototype, byte_range)
        probe._enter()
        try:
            await probe._hold()
            return await super().get(key, prototype, byte_range)
        finally:
            probe._exit()

    async def set(self, key: str, value: Buffer) -> None:
        probe = _PROBE
        if probe is None or probe.kind not in ("all", _kind(key)):
            return await super().set(key, value)
        probe._enter()
        try:
            await probe._hold()
            return await super().set(key, value)
        finally:
            probe._exit()


def rendezvous_store(source) -> RendezvousNgioStore:
    """Wrap `source` the same way `counting_store` does."""
    store = RendezvousNgioStore.from_any(source)
    assert isinstance(store, RendezvousNgioStore)
    return store
