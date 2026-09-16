"""Canonical NS descendant basis shared by the exact and numerical modules.

Words use the Hadasz--Jaskolski--Suchanek convention (hep-th/0611266):
G modes first, with their positive indices increasing from left to right,
followed by L modes with nondecreasing positive indices. Mode indices are
stored doubled and negative. Vector order agrees with the displayed Gram
matrices in Belavin--Geiko (1806.09563), Appendix B; e.g. level two is
``(L_-1**2, L_-2, G_-1/2 G_-3/2)``.

At higher levels the ordering rule is fewer G modes first, then lexicographic
order in the positive mode indices. This orders vectors within a fixed level;
it introduces no normalization, fermion phase, or primary propagation power.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterator, Literal

Mode = tuple[Literal["L", "G"], int]
State = tuple[Mode, ...]
NS_PBW_BASIS_CONVENTION = "hjs-words-bg-appendix-b-vector-order-v1"


def _integer_partitions(total: int, maximum: int | None = None) -> Iterator[tuple[int, ...]]:
    if total == 0:
        yield ()
        return
    maximum = min(total, total if maximum is None else maximum)
    for first in range(maximum, 0, -1):
        for tail in _integer_partitions(total - first, first):
            yield (first,) + tail


def _strict_odd_partitions(total: int, maximum: int | None = None) -> Iterator[tuple[int, ...]]:
    if total == 0:
        yield ()
        return
    maximum = min(total, total if maximum is None else maximum)
    if maximum % 2 == 0:
        maximum -= 1
    for first in range(maximum, 0, -2):
        for tail in _strict_odd_partitions(total - first, first - 2):
            yield (first,) + tail


@lru_cache(maxsize=None)
def ns_pbw_basis(twice_level: int) -> tuple[State, ...]:
    """Return all NS descendants in the literature's displayed vector order."""
    if not isinstance(twice_level, int) or twice_level < 0:
        raise ValueError("twice_level must be a nonnegative integer")
    states: list[State] = []
    for g_twice_level in range(twice_level + 1):
        remainder = twice_level - g_twice_level
        if remainder % 2:
            continue
        for g_parts in _strict_odd_partitions(g_twice_level):
            for l_parts in _integer_partitions(remainder // 2):
                states.append(tuple(
                    [("G", -part) for part in reversed(g_parts)]
                    + [("L", -2 * part) for part in reversed(l_parts)]
                ))
    return tuple(sorted(states, key=lambda state: (
        sum(kind == "G" for kind, _ in state),
        tuple((kind, -index) for kind, index in state),
    )))
