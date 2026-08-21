"""Disjoint-set union over arbitrary hashable items.

Used to resolve label equivalences discovered tile by tile: each `union` says
"these two ids are the same object", and `resolve` turns the accumulated claims
into one canonical id per connected group.
"""

from collections.abc import Hashable, Iterable
from typing import Generic, TypeVar

T = TypeVar("T", bound=Hashable)


class UnionFind(Generic[T]):
    """Union-find with path compression and union by size.

    Items are created on first mention, so there is no separate `add` step.

    Example:
        ```python
        uf = UnionFind[int]()
        uf.union(1, 2)
        uf.union(2, 3)
        uf.find(3)  # -> 1
        ```
    """

    def __init__(self, items: Iterable[T] | None = None) -> None:
        """Build the structure, optionally seeding it with singleton items."""
        self._parent: dict[T, T] = {}
        self._size: dict[T, int] = {}
        for item in items or ():
            self._ensure(item)

    def __len__(self) -> int:
        """The number of distinct items seen."""
        return len(self._parent)

    def __contains__(self, item: T) -> bool:
        """Whether `item` has been seen."""
        return item in self._parent

    def _ensure(self, item: T) -> None:
        if item not in self._parent:
            self._parent[item] = item
            self._size[item] = 1

    def find(self, item: T) -> T:
        """The canonical representative of `item`'s group."""
        self._ensure(item)
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        # Path compression: point everything on the way up straight at the root.
        while self._parent[item] != root:
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, a: T, b: T) -> None:
        """Merge the groups containing `a` and `b`."""
        root_a, root_b = self.find(a), self.find(b)
        if root_a == root_b:
            return
        # Union by size keeps the trees shallow.
        if self._size[root_a] < self._size[root_b]:
            root_a, root_b = root_b, root_a
        self._parent[root_b] = root_a
        self._size[root_a] += self._size[root_b]

    def resolve(self) -> dict[T, T]:
        """Map every item to its group's representative.

        The representative is the smallest item in the group when the items
        order, so the result is deterministic rather than dependent on the
        order the unions arrived in.
        """
        groups: dict[T, list[T]] = {}
        for item in self._parent:
            groups.setdefault(self.find(item), []).append(item)

        mapping: dict[T, T] = {}
        for members in groups.values():
            try:
                representative = min(members)  # type: ignore[type-var]
            except TypeError:
                representative = members[0]
            for member in members:
                mapping[member] = representative
        return mapping
