#!/usr/bin/env python3
"""Direct NS Majorana sewing on the genus-two pants graphs.

This is an independent finite-level oracle for the fermionic factor used in
``ns_genus2_partition.py``.  It works entirely in the plumbing coordinates of
the two trivalent graphs and does not evaluate a period matrix, a Riemann theta
constant, or a Schottky primitive-word product.

The NS Fock basis is

    psi(-k_1) ... psi(-k_s) |0>,       1 <= k_1 < ... < k_s,

with twice-level ``sum_i (2 k_i - 1)``.  Three-point coefficients are Pfaffians
of the sphere two-point function, with the fields at infinity placed in the
BPZ-dual (reversed) order.  The theta/glasses sewing signs are supplied by the
same frame-derived orientation ledgers used by the generic NS block code.
Common ``q^{-c/24}`` edge powers are deliberately stripped.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import comb
from typing import Sequence


FermionState = tuple[int, ...]
TaggedMode = tuple[str, int]


@lru_cache(maxsize=None)
def ns_fermion_states_at_twice_level(twice_level: int) -> tuple[FermionState, ...]:
    """Return distinct-mode NS Fock states at a fixed twice-level."""

    twice_level = int(twice_level)
    if twice_level < 0:
        return ()

    out: list[FermionState] = []

    def visit(remaining: int, next_mode: int, chosen: tuple[int, ...]) -> None:
        if remaining == 0:
            out.append(chosen)
            return
        mode = int(next_mode)
        while 2 * mode - 1 <= remaining:
            visit(remaining - (2 * mode - 1), mode + 1, chosen + (mode,))
            mode += 1

    visit(twice_level, 1, ())
    return tuple(out)


def ns_fermion_twice_level(state: Sequence[int]) -> int:
    return sum(2 * int(mode) - 1 for mode in state)


def ns_fermion_parity(state: Sequence[int]) -> int:
    return len(tuple(state)) % 2


def _pair_contraction(left: TaggedMode, right: TaggedMode) -> int:
    """Sphere contraction for the ordered regions infinity, one, zero."""

    left_region, left_mode = left
    right_region, right_mode = right
    if left_region == right_region:
        # Each vertex operator is normal ordered.
        return 0
    if (left_region, right_region) == ("bra", "middle"):
        if left_mode < right_mode:
            return 0
        return comb(left_mode - 1, right_mode - 1)
    if (left_region, right_region) == ("bra", "ket"):
        return int(left_mode == right_mode)
    if (left_region, right_region) == ("middle", "ket"):
        sign = -1 if (left_mode - 1) % 2 else 1
        return sign * comb(left_mode + right_mode - 2, left_mode - 1)
    raise ValueError(f"unexpected fermion region order {(left_region, right_region)!r}")


@lru_cache(maxsize=None)
def _pfaffian_pairing_sum(fields: tuple[TaggedMode, ...]) -> int:
    """Return the exact Pfaffian by recursive expansion of its first row."""

    if not fields:
        return 1
    if len(fields) % 2:
        return 0
    first = fields[0]
    total = 0
    for partner_index in range(1, len(fields)):
        contraction = _pair_contraction(first, fields[partner_index])
        if contraction == 0:
            continue
        pfaffian_sign = -1 if partner_index % 2 == 0 else 1
        remainder = fields[1:partner_index] + fields[partner_index + 1 :]
        total += (
            pfaffian_sign
            * contraction
            * _pfaffian_pairing_sum(remainder)
        )
    return int(total)


@lru_cache(maxsize=None)
def majorana_three_point(
    bra: FermionState,
    middle: FermionState,
    ket: FermionState,
) -> int:
    r"""Return ``<bra|Y(middle,1)|ket>`` in the NS Majorana Fock basis."""

    bra = tuple(int(mode) for mode in bra)
    middle = tuple(int(mode) for mode in middle)
    ket = tuple(int(mode) for mode in ket)
    for state in (bra, middle, ket):
        if tuple(sorted(state)) != state or len(set(state)) != len(state):
            raise ValueError("Majorana modes must be positive, distinct, and increasing")
        if any(mode <= 0 for mode in state):
            raise ValueError("Majorana modes must be positive")

    # The BPZ dual reverses the annihilation operators.  This makes
    # majorana_three_point(state, (), state) exactly +1.
    fields = (
        tuple(("bra", mode) for mode in reversed(bra))
        + tuple(("middle", mode) for mode in middle)
        + tuple(("ket", mode) for mode in ket)
    )
    return _pfaffian_pairing_sum(fields)


@dataclass(frozen=True)
class MajoranaPlumbingResult:
    channel: str
    q_values: tuple[complex, complex, complex]
    max_total_twice_level: int
    chiral_value: complex
    last_shell: complex
    nonzero_level_triples: int


def _orientation_sign(channel: str, parities: tuple[int, int, int]) -> int:
    # Imported lazily so the elementary Pfaffian oracle remains usable without
    # the super-Virasoro implementation on PYTHONPATH.
    if channel == "theta":
        from ns_regular_block import THETA_ORIENTATION

        return int(THETA_ORIENTATION.sign(parities))
    if channel == "glasses":
        from ns_genus2_partition import GLASSES_ORIENTATION

        return int(GLASSES_ORIENTATION.sign(parities))
    raise ValueError("channel must be theta or glasses")


def theta_majorana_plumbing_partition(
    q_zero: complex,
    q_one: complex,
    q_infty: complex,
    *,
    max_total_twice_level: int,
    lifts: Sequence[int] = (1, 1, 1),
) -> MajoranaPlumbingResult:
    """Direct Majorana sewing sum in the theta graph."""

    q_values = (complex(q_zero), complex(q_one), complex(q_infty))
    lift_values = tuple(int(value) for value in lifts)
    if len(lift_values) != 3 or any(value not in (-1, 1) for value in lift_values):
        raise ValueError("three +/-1 plumbing lifts are required")
    cutoff = int(max_total_twice_level)
    states = tuple(ns_fermion_states_at_twice_level(level) for level in range(cutoff + 1))
    total = 0.0j
    last_shell = 0.0j
    nonzero = 0
    for level_infty in range(cutoff + 1):
        for level_one in range(cutoff + 1 - level_infty):
            for level_zero in range(cutoff + 1 - level_infty - level_one):
                coefficient = 0
                for state_infty in states[level_infty]:
                    for state_one in states[level_one]:
                        for state_zero in states[level_zero]:
                            rho = majorana_three_point(state_infty, state_one, state_zero)
                            if rho:
                                coefficient += rho * rho
                if coefficient == 0:
                    continue
                parities = (level_zero % 2, level_one % 2, level_infty % 2)
                sign = _orientation_sign("theta", parities)
                lift = (
                    lift_values[0] ** parities[0]
                    * lift_values[1] ** parities[1]
                    * lift_values[2] ** parities[2]
                )
                contribution = (
                    sign
                    * lift
                    * coefficient
                    * q_values[0] ** (0.5 * level_zero)
                    * q_values[1] ** (0.5 * level_one)
                    * q_values[2] ** (0.5 * level_infty)
                )
                total += contribution
                if level_zero + level_one + level_infty == cutoff:
                    last_shell += contribution
                nonzero += 1
    return MajoranaPlumbingResult(
        channel="theta",
        q_values=q_values,
        max_total_twice_level=cutoff,
        chiral_value=complex(total),
        last_shell=complex(last_shell),
        nonzero_level_triples=nonzero,
    )


def glasses_majorana_plumbing_partition(
    q_left: complex,
    q_right: complex,
    q_bridge: complex,
    *,
    max_total_twice_level: int,
    lifts: Sequence[int] = (1, 1, 1),
) -> MajoranaPlumbingResult:
    """Direct Majorana sewing sum in the two-self-loop glasses graph."""

    q_values = (complex(q_left), complex(q_right), complex(q_bridge))
    lift_values = tuple(int(value) for value in lifts)
    if len(lift_values) != 3 or any(value not in (-1, 1) for value in lift_values):
        raise ValueError("three +/-1 plumbing lifts are required")
    cutoff = int(max_total_twice_level)
    states = tuple(ns_fermion_states_at_twice_level(level) for level in range(cutoff + 1))
    total = 0.0j
    last_shell = 0.0j
    nonzero = 0
    for level_left in range(cutoff + 1):
        for level_right in range(cutoff + 1 - level_left):
            for level_bridge in range(cutoff + 1 - level_left - level_right):
                coefficient = 0
                for state_left in states[level_left]:
                    for state_right in states[level_right]:
                        for state_bridge in states[level_bridge]:
                            rho_left = majorana_three_point(
                                state_left, state_bridge, state_left
                            )
                            if rho_left == 0:
                                continue
                            rho_right = majorana_three_point(
                                state_right, state_bridge, state_right
                            )
                            if rho_right:
                                coefficient += rho_left * rho_right
                if coefficient == 0:
                    continue
                parities = (level_left % 2, level_right % 2, level_bridge % 2)
                sign = _orientation_sign("glasses", parities)
                lift = (
                    lift_values[0] ** parities[0]
                    * lift_values[1] ** parities[1]
                    * lift_values[2] ** parities[2]
                )
                contribution = (
                    sign
                    * lift
                    * coefficient
                    * q_values[0] ** (0.5 * level_left)
                    * q_values[1] ** (0.5 * level_right)
                    * q_values[2] ** (0.5 * level_bridge)
                )
                total += contribution
                if level_left + level_right + level_bridge == cutoff:
                    last_shell += contribution
                nonzero += 1
    return MajoranaPlumbingResult(
        channel="glasses",
        q_values=q_values,
        max_total_twice_level=cutoff,
        chiral_value=complex(total),
        last_shell=complex(last_shell),
        nonzero_level_triples=nonzero,
    )

