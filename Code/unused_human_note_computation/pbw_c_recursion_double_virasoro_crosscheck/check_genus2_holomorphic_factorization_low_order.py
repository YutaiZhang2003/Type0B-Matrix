#!/usr/bin/env python3
"""Exact low-order audit of genus-two holomorphic factorization signs.

The calculation follows the notation of ``Human Notes/SCblock.tex``.  The
holomorphic theta edges use ``(A,B)``, ``(C,D)``, and ``(E,F)``; barred
letters label an independent antiholomorphic NS Majorana module.  We insert
the full resolution of identity and sum the full tensor-product PBW/Fock
basis directly.

Nothing in the direct sum is formed by multiplying precomputed chiral block
coefficients.  The comparison with chiral products is performed only after
the full coefficient has been obtained.  The audit keeps separate:

* the algebraic dagger sign in every inverse Gram matrix;
* the two local signs that group each full trinion by chirality;
* the theta sign obtained by an explicit permutation from vertex order to
  edge-contraction order;
* the Pfaffian three-point coefficients in the human dagger convention.

The NS Majorana basis is finite at every twice-level, so all arithmetic is
exact over the integers.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product
from pathlib import Path
import sys
from typing import Iterable, Mapping, Sequence


REPOSITORY = Path(__file__).resolve().parents[2]
PYTHON_DIRECTORY = REPOSITORY / "Code" / "genus_2_cross_channel"
sys.path.insert(0, str(PYTHON_DIRECTORY))

from free_majorana_pair_of_pants import (  # noqa: E402
    FermionState,
    majorana_three_point,
    ns_fermion_parity,
    ns_fermion_states_at_twice_level,
)


Level = tuple[int, int, int]
StateTriple = tuple[FermionState, FermionState, FermionState]


def level_triples(cutoff: int) -> tuple[Level, ...]:
    """Return ``(A,C,E)`` twice-levels of total degree at most ``cutoff``."""

    return tuple(
        (a, c, e)
        for a in range(cutoff + 1)
        for c in range(cutoff + 1 - a)
        for e in range(cutoff + 1 - a - c)
    )


def state_triples(
    levels: Level,
    states_by_level: Sequence[tuple[FermionState, ...]],
) -> Iterable[StateTriple]:
    return product(*(states_by_level[level] for level in levels))


def koszul_permutation_exponent(
    source_order: Sequence[str],
    target_order: Sequence[str],
    parities: Mapping[str, int],
) -> int:
    """Count odd inversions between two orders of the same named objects."""

    source = tuple(source_order)
    target = tuple(target_order)
    if len(set(source)) != len(source) or set(source) != set(target):
        raise ValueError("source_order and target_order must be permutations")
    target_position = {name: position for position, name in enumerate(target)}
    exponent = 0
    for left in range(len(source)):
        for right in range(left + 1, len(source)):
            if target_position[source[left]] > target_position[source[right]]:
                exponent += (
                    int(parities[source[left]]) * int(parities[source[right]])
                )
    return exponent % 2


def sign(exponent: int) -> int:
    return -1 if int(exponent) % 2 else 1


def theta_orientation_exponent(edge_parities: Sequence[int]) -> int:
    """Derive the theta exponent by permuting trinion order to edge order.

    The source order is the concatenation of the two ordered trinions,
    ``(A,C,E,B,D,F)``.  The resolution of identity contracts in the order
    ``(A,B,C,D,E,F)``.  Equal parity at the two ends of an edge is imposed by
    the Gram matrix.
    """

    a, c, e = (int(value) % 2 for value in edge_parities)
    parities = {"A": a, "B": a, "C": c, "D": c, "E": e, "F": e}
    return koszul_permutation_exponent(
        ("A", "C", "E", "B", "D", "F"),
        ("A", "B", "C", "D", "E", "F"),
        parities,
    )


def local_chirality_exponent(
    left_parities: Sequence[int], right_parities: Sequence[int]
) -> int:
    """Group one full trinion from interleaved to chirality-first order."""

    a, c, e = (int(value) % 2 for value in left_parities)
    abar, cbar, ebar = (int(value) % 2 for value in right_parities)
    parities = {
        "A": a,
        "Abar": abar,
        "C": c,
        "Cbar": cbar,
        "E": e,
        "Ebar": ebar,
    }
    return koszul_permutation_exponent(
        ("A", "Abar", "C", "Cbar", "E", "Ebar"),
        ("A", "C", "E", "Abar", "Cbar", "Ebar"),
        parities,
    )


def cross_exponent(left_parities: Sequence[int], right_parities: Sequence[int]) -> int:
    """Return the quotient of full and separate theta orientation signs."""

    left = tuple(int(value) % 2 for value in left_parities)
    right = tuple(int(value) % 2 for value in right_parities)
    total = tuple(a ^ abar for a, abar in zip(left, right))
    return (
        theta_orientation_exponent(total)
        + theta_orientation_exponent(left)
        + theta_orientation_exponent(right)
    ) % 2


def human_majorana_three_point(states: StateTriple) -> int:
    """Majorana three-point form with the human algebraic dagger.

    ``majorana_three_point`` uses the reversed BPZ-dual bra word with positive
    unit norm.  The human convention ``psi_r^dagger=-psi_-r`` contributes one
    minus sign per fermion in the first (infinity) slot.
    """

    a_state, c_state, e_state = states
    dagger_sign = sign(len(a_state))
    return dagger_sign * majorana_three_point(a_state, c_state, e_state)


def inverse_gram_sign(states: StateTriple) -> int:
    """Product of the three inverse-Gram signs in the human Fock basis."""

    return sign(sum(len(state) for state in states))


def parity_triple(states: StateTriple) -> Level:
    return tuple(ns_fermion_parity(state) for state in states)  # type: ignore[return-value]


def chiral_direct_coefficient(
    levels: Level,
    states_by_level: Sequence[tuple[FermionState, ...]],
) -> int:
    """Compute the human chiral coefficient directly from its PBW sum."""

    theta_sign = sign(theta_orientation_exponent(tuple(level % 2 for level in levels)))
    coefficient = 0
    for states in state_triples(levels, states_by_level):
        rho_first = human_majorana_three_point(states)
        if rho_first == 0:
            continue
        # The second trinion has labels (B,D,F).  In the diagonal Fock basis
        # the resolution sets it equal to the same state triple.
        rho_second = human_majorana_three_point(states)
        coefficient += (
            theta_sign
            * inverse_gram_sign(states)
            * rho_first
            * rho_second
        )
    return int(coefficient)


def full_direct_coefficient(
    left_levels: Level,
    right_levels: Level,
    states_by_level: Sequence[tuple[FermionState, ...]],
) -> int:
    """Compute one full nonchiral coefficient before chiral factorization."""

    total_parities = tuple(
        (left_level + right_level) % 2
        for left_level, right_level in zip(left_levels, right_levels)
    )
    theta_sign = sign(theta_orientation_exponent(total_parities))
    coefficient = 0
    for left_states in state_triples(left_levels, states_by_level):
        rho_left_first = human_majorana_three_point(left_states)
        if rho_left_first == 0:
            continue
        rho_left_second = human_majorana_three_point(left_states)
        left_gram = inverse_gram_sign(left_states)
        left_parities = parity_triple(left_states)

        for right_states in state_triples(right_levels, states_by_level):
            rho_right_first = human_majorana_three_point(right_states)
            if rho_right_first == 0:
                continue
            rho_right_second = human_majorana_three_point(right_states)
            right_gram = inverse_gram_sign(right_states)
            right_parities = parity_triple(right_states)

            # These signs are computed independently for the two trinions.
            first_local = sign(
                local_chirality_exponent(left_parities, right_parities)
            )
            second_local = sign(
                local_chirality_exponent(left_parities, right_parities)
            )
            full_rho_first = (
                first_local * rho_left_first * rho_right_first
            )
            full_rho_second = (
                second_local * rho_left_second * rho_right_second
            )
            coefficient += (
                theta_sign
                * left_gram
                * right_gram
                * full_rho_first
                * full_rho_second
            )
    return int(coefficient)


@dataclass(frozen=True)
class Ledger:
    left_levels: Level
    right_levels: Level
    left_rho: int
    right_rho: int
    left_gram: int
    right_gram: int
    first_local: int
    second_local: int
    full_theta: int
    direct_full: int
    left_chiral: int
    right_chiral: int
    ordinary_product: int
    cross_sign: int
    graded_product: int


def one_state_ledger(
    left_levels: Level,
    right_levels: Level,
    states_by_level: Sequence[tuple[FermionState, ...]],
) -> Ledger:
    left_basis = tuple(state_triples(left_levels, states_by_level))
    right_basis = tuple(state_triples(right_levels, states_by_level))
    if len(left_basis) != 1 or len(right_basis) != 1:
        raise ValueError("the displayed ledger requires one state per level triple")
    left_states = left_basis[0]
    right_states = right_basis[0]
    left_parities = parity_triple(left_states)
    right_parities = parity_triple(right_states)
    left_rho = human_majorana_three_point(left_states)
    right_rho = human_majorana_three_point(right_states)
    left_gram = inverse_gram_sign(left_states)
    right_gram = inverse_gram_sign(right_states)
    first_local = sign(local_chirality_exponent(left_parities, right_parities))
    second_local = sign(local_chirality_exponent(left_parities, right_parities))
    total_parities = tuple(a ^ abar for a, abar in zip(left_parities, right_parities))
    full_theta = sign(theta_orientation_exponent(total_parities))
    direct_full = (
        full_theta
        * left_gram
        * right_gram
        * (first_local * left_rho * right_rho)
        * (second_local * left_rho * right_rho)
    )
    left_chiral = chiral_direct_coefficient(left_levels, states_by_level)
    right_chiral = chiral_direct_coefficient(right_levels, states_by_level)
    ordinary_product = left_chiral * right_chiral
    cross_sign_value = sign(cross_exponent(left_parities, right_parities))
    return Ledger(
        left_levels=left_levels,
        right_levels=right_levels,
        left_rho=left_rho,
        right_rho=right_rho,
        left_gram=left_gram,
        right_gram=right_gram,
        first_local=first_local,
        second_local=second_local,
        full_theta=full_theta,
        direct_full=direct_full,
        left_chiral=left_chiral,
        right_chiral=right_chiral,
        ordinary_product=ordinary_product,
        cross_sign=cross_sign_value,
        graded_product=cross_sign_value * ordinary_product,
    )


def print_ledger(label: str, ledger: Ledger) -> None:
    print(label)
    print(
        "  2(|A|,|C|,|E|)="
        f"{ledger.left_levels}, "
        "2(|Abar|,|Cbar|,|Ebar|)="
        f"{ledger.right_levels}"
    )
    print(f"  rho_left={ledger.left_rho:+d}, rho_right={ledger.right_rho:+d}")
    print(
        "  inverse-Gram signs: "
        f"left={ledger.left_gram:+d}, right={ledger.right_gram:+d}"
    )
    print(
        "  local chirality signs: "
        f"first trinion={ledger.first_local:+d}, "
        f"second trinion={ledger.second_local:+d}"
    )
    print(f"  full theta permutation sign={ledger.full_theta:+d}")
    print(f"  direct full coefficient={ledger.direct_full:+d}")
    print(
        "  chiral coefficients: "
        f"left={ledger.left_chiral:+d}, right={ledger.right_chiral:+d}"
    )
    print(f"  ordinary product={ledger.ordinary_product:+d}")
    print(f"  left-right cross sign={ledger.cross_sign:+d}")
    print(f"  graded product={ledger.graded_product:+d}")


def run(cutoff: int) -> None:
    states_by_level = tuple(
        ns_fermion_states_at_twice_level(level) for level in range(cutoff + 1)
    )

    # Sanity checks for the dagger and the literal permutation derivation.
    one_particle = (1,)
    vacuum: FermionState = ()
    if human_majorana_three_point((one_particle, one_particle, vacuum)) != -1:
        raise AssertionError("rho(f 1,f 1,1) does not match the human dagger")
    if human_majorana_three_point((one_particle, vacuum, one_particle)) != -1:
        raise AssertionError("rho(f 1,1,f 1) does not match the human dagger")
    if human_majorana_three_point((vacuum, one_particle, one_particle)) != 1:
        raise AssertionError("rho(1,f 1,f 1) does not match the human dagger")
    for parities in product((0, 1), repeat=3):
        a, c, e = parities
        if theta_orientation_exponent(parities) != (a * c + a * e + c * e) % 2:
            raise AssertionError("explicit theta permutation changed")

    triples = level_triples(cutoff)
    chiral = {
        levels: chiral_direct_coefficient(levels, states_by_level)
        for levels in triples
    }
    tested = 0
    nonzero_full = 0
    ordinary_mismatches = []
    for left_levels in triples:
        for right_levels in triples:
            if sum(left_levels) + sum(right_levels) > cutoff:
                continue
            direct = full_direct_coefficient(
                left_levels, right_levels, states_by_level
            )
            ordinary = chiral[left_levels] * chiral[right_levels]
            cross = sign(
                cross_exponent(
                    tuple(level % 2 for level in left_levels),
                    tuple(level % 2 for level in right_levels),
                )
            )
            graded = cross * ordinary
            if direct != graded:
                raise AssertionError(
                    "direct full sum disagrees with graded factorization at "
                    f"{left_levels}, {right_levels}: {direct} != {graded}"
                )
            if direct != ordinary:
                ordinary_mismatches.append(
                    (left_levels, right_levels, direct, ordinary, cross)
                )
            tested += 1
            nonzero_full += int(direct != 0)

    diagonal = one_state_ledger((1, 1, 0), (1, 1, 0), states_by_level)
    cross_sensitive = one_state_ledger((1, 1, 0), (0, 1, 1), states_by_level)
    print_ledger("Ledger 1: local signs are nontrivial but cancel", diagonal)
    print_ledger("Ledger 2: first left-right-sensitive coefficient", cross_sensitive)
    print(
        "PASS: direct full nonchiral PBW/Fock sewing agrees with the graded "
        f"holomorphic product through total twice-level {cutoff}."
    )
    print(f"  full level pairs tested: {tested}")
    print(f"  nonzero direct full coefficients: {nonzero_full}")
    print(
        "  nonzero coefficients missed by ordinary factorization: "
        f"{len(ordinary_mismatches)}"
    )
    for row in ordinary_mismatches[:8]:
        left_levels, right_levels, direct, ordinary, cross = row
        print(
            "    "
            f"{left_levels} x {right_levels}: "
            f"direct={direct:+d}, ordinary={ordinary:+d}, cross={cross:+d}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", type=int, default=8)
    arguments = parser.parse_args()
    if arguments.cutoff < 4:
        parser.error("cutoff must be at least 4 to include the first cross sign")
    run(arguments.cutoff)
