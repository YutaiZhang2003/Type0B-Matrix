#!/usr/bin/env python3
"""Direct rank-one Heisenberg-VOA sewing on genus-two pants graphs.

This module computes the free-boson oscillator partition function by the same
pair-of-pants sewing operation used for the CCY Virasoro blocks.  It does not
use Schottky primitive-word products internally.

The Heisenberg vacuum module has basis

    a_{-lambda_1} ... a_{-lambda_k} |0>,

labelled by integer partitions ``lambda``.  Its Gram matrix is diagonal with
norm ``z_lambda = product_n n^m_n m_n!``.  The pants coefficient is

    rho(lambda, mu, nu) = <lambda| Y(mu, 1) |nu>,

and is evaluated by Wick contractions of the current and its derivatives.
The sewing propagator is ``q^L0``: no ``-c/24`` shift is inserted.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable


Partition = tuple[int, ...]


def parse_complex(value: str) -> complex:
    return complex(value.replace("i", "j"))


def format_complex(value: complex) -> str:
    value = complex(value)
    return f"{value.real:+.12e}{value.imag:+.12e}j"


@lru_cache(maxsize=None)
def integer_partitions(total: int, max_part: int | None = None) -> tuple[Partition, ...]:
    """Return integer partitions of ``total`` in nonincreasing order."""

    total = int(total)
    if total < 0:
        return ()
    if total == 0:
        return ((),)
    if max_part is None:
        max_part = total
    out: list[Partition] = []
    for part in range(min(int(max_part), total), 0, -1):
        for tail in integer_partitions(total - part, part):
            out.append((part,) + tail)
    return tuple(out)


def partition_level(partition: Partition) -> int:
    return int(sum(partition))


@lru_cache(maxsize=None)
def heisenberg_gram_norm(partition: Partition) -> int:
    r"""Return ``z_lambda = product_n n^m_n m_n!``."""

    norm = 1
    for mode, multiplicity in Counter(partition).items():
        if mode <= 0:
            raise ValueError("Heisenberg creation modes must be positive")
        norm *= (int(mode) ** int(multiplicity)) * math.factorial(int(multiplicity))
    return int(norm)


def _remove_index(values: tuple[int, ...], index: int) -> tuple[int, ...]:
    return values[:index] + values[index + 1 :]


def _bra_middle_contraction(bra_mode: int, middle_mode: int) -> int:
    r"""Return ``<0|a_bra D_middle J(1)|0>``."""

    if bra_mode < middle_mode:
        return 0
    return int(bra_mode * math.comb(bra_mode - 1, middle_mode - 1))


def _middle_ket_contraction(middle_mode: int, ket_mode: int) -> int:
    r"""Return ``<0|D_middle J(1) a_-ket|0>``."""

    sign = -1 if (middle_mode - 1) % 2 else 1
    return int(sign * ket_mode * math.comb(ket_mode + middle_mode - 1, middle_mode - 1))


def _bra_ket_contraction(bra_mode: int, ket_mode: int) -> int:
    return int(bra_mode) if bra_mode == ket_mode else 0


@lru_cache(maxsize=None)
def _heisenberg_pairing_sum(
    bra: tuple[int, ...],
    middle: tuple[int, ...],
    ket: tuple[int, ...],
) -> int:
    """Sum complete Wick pairings with no same-insertion contractions."""

    field_count = len(bra) + len(middle) + len(ket)
    if field_count == 0:
        return 1
    if field_count % 2:
        return 0

    total = 0
    if bra:
        mode = bra[0]
        rest_bra = bra[1:]
        for index, partner in enumerate(middle):
            contraction = _bra_middle_contraction(mode, partner)
            if contraction:
                total += contraction * _heisenberg_pairing_sum(
                    rest_bra,
                    _remove_index(middle, index),
                    ket,
                )
        for index, partner in enumerate(ket):
            contraction = _bra_ket_contraction(mode, partner)
            if contraction:
                total += contraction * _heisenberg_pairing_sum(
                    rest_bra,
                    middle,
                    _remove_index(ket, index),
                )
        return int(total)

    if middle:
        mode = middle[0]
        rest_middle = middle[1:]
        for index, partner in enumerate(ket):
            contraction = _middle_ket_contraction(mode, partner)
            if contraction:
                total += contraction * _heisenberg_pairing_sum(
                    bra,
                    rest_middle,
                    _remove_index(ket, index),
                )
        return int(total)

    return 0


@lru_cache(maxsize=None)
def heisenberg_three_point(bra: Partition, middle: Partition, ket: Partition) -> int:
    r"""Return ``<bra|Y(middle,1)|ket>`` in the Heisenberg vacuum module.

    For ``middle=(n_1,...,n_k)``, the vertex operator is the normal-ordered
    product of ``(n_i-1)`` derivatives of the current.  Wick contractions
    among fields belonging to the same insertion are excluded.
    """

    bra = tuple(int(mode) for mode in bra)
    middle = tuple(int(mode) for mode in middle)
    ket = tuple(int(mode) for mode in ket)
    if any(mode <= 0 for mode in bra + middle + ket):
        raise ValueError("Heisenberg partitions must contain positive modes")
    return _heisenberg_pairing_sum(bra, middle, ket)


@dataclass(frozen=True)
class HeisenbergPlumbingResult:
    channel: str
    q_values: tuple[complex, complex, complex]
    max_total_level: int
    chiral_value: complex
    nonchiral_value: float
    level_contributions: dict[tuple[int, int, int], complex]


def _states_by_level(max_total_level: int) -> tuple[tuple[Partition, ...], ...]:
    if max_total_level < 0:
        raise ValueError("max_total_level must be non-negative")
    return tuple(integer_partitions(level) for level in range(int(max_total_level) + 1))


def theta_heisenberg_plumbing_partition(
    q_zero: complex,
    q_one: complex,
    q_infty: complex,
    *,
    max_total_level: int,
) -> HeisenbergPlumbingResult:
    r"""Direct Heisenberg sewing sum in the theta graph.

    The chiral quantity is

      sum_{lambda,mu,nu} q_infty^|lambda| q_one^|mu| q_zero^|nu|
        rho(lambda,mu,nu)^2 / (z_lambda z_mu z_nu),

    truncated by ``|lambda|+|mu|+|nu| <= max_total_level``.
    """

    q_values = (complex(q_zero), complex(q_one), complex(q_infty))
    states = _states_by_level(int(max_total_level))
    total = 0.0 + 0.0j
    level_contributions: dict[tuple[int, int, int], complex] = {}
    # rho is ordered as (infinity bra, insertion at one, zero ket), whereas
    # generators_for_theta and the public plumbing API use (zero, one, infinity).
    for level_infty in range(int(max_total_level) + 1):
        for level_one in range(int(max_total_level) + 1 - level_infty):
            for level_zero in range(int(max_total_level) + 1 - level_infty - level_one):
                coefficient = 0.0 + 0.0j
                for state_infty in states[level_infty]:
                    norm_infty = heisenberg_gram_norm(state_infty)
                    for state_one in states[level_one]:
                        norm_one = heisenberg_gram_norm(state_one)
                        for state_zero in states[level_zero]:
                            rho = heisenberg_three_point(state_infty, state_one, state_zero)
                            if rho == 0:
                                continue
                            norm_zero = heisenberg_gram_norm(state_zero)
                            coefficient += (rho * rho) / (norm_infty * norm_one * norm_zero)
                contribution = (
                    (q_values[0] ** level_zero)
                    * (q_values[1] ** level_one)
                    * (q_values[2] ** level_infty)
                    * coefficient
                )
                if contribution != 0.0:
                    level_contributions[(level_zero, level_one, level_infty)] = contribution
                    total += contribution
    return HeisenbergPlumbingResult(
        channel="theta",
        q_values=q_values,
        max_total_level=int(max_total_level),
        chiral_value=total,
        nonchiral_value=float(abs(total) ** 2),
        level_contributions=level_contributions,
    )


def glasses_heisenberg_plumbing_partition(
    q_left: complex,
    q_right: complex,
    q_bridge: complex,
    *,
    max_total_level: int,
) -> HeisenbergPlumbingResult:
    r"""Direct Heisenberg sewing sum in the separating glasses graph.

    The chiral quantity is

      sum_{lambda,mu,nu} qL^|lambda| qR^|mu| qB^|nu|
        rho(lambda,nu,lambda) rho(mu,nu,mu)
        / (z_lambda z_mu z_nu),

    truncated by total sewing level.
    """

    q_values = (complex(q_left), complex(q_right), complex(q_bridge))
    states = _states_by_level(int(max_total_level))
    total = 0.0 + 0.0j
    level_contributions: dict[tuple[int, int, int], complex] = {}
    for level_left in range(int(max_total_level) + 1):
        for level_right in range(int(max_total_level) + 1 - level_left):
            for level_bridge in range(int(max_total_level) + 1 - level_left - level_right):
                coefficient = 0.0 + 0.0j
                for state_left in states[level_left]:
                    norm_left = heisenberg_gram_norm(state_left)
                    for state_right in states[level_right]:
                        norm_right = heisenberg_gram_norm(state_right)
                        for state_bridge in states[level_bridge]:
                            rho_left = heisenberg_three_point(state_left, state_bridge, state_left)
                            if rho_left == 0:
                                continue
                            rho_right = heisenberg_three_point(state_right, state_bridge, state_right)
                            if rho_right == 0:
                                continue
                            norm_bridge = heisenberg_gram_norm(state_bridge)
                            coefficient += (rho_left * rho_right) / (
                                norm_left * norm_right * norm_bridge
                            )
                contribution = (
                    (q_values[0] ** level_left)
                    * (q_values[1] ** level_right)
                    * (q_values[2] ** level_bridge)
                    * coefficient
                )
                if contribution != 0.0:
                    level_contributions[(level_left, level_right, level_bridge)] = contribution
                    total += contribution
    return HeisenbergPlumbingResult(
        channel="glasses",
        q_values=q_values,
        max_total_level=int(max_total_level),
        chiral_value=total,
        nonchiral_value=float(abs(total) ** 2),
        level_contributions=level_contributions,
    )


def run(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Direct genus-two Heisenberg pair-of-pants sewing sum.")
    parser.add_argument("--channel", choices=("theta", "glasses"), required=True)
    parser.add_argument("--q", type=parse_complex, nargs=3, required=True)
    parser.add_argument("--max-total-level", type=int, default=6)
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.channel == "theta":
        result = theta_heisenberg_plumbing_partition(*args.q, max_total_level=args.max_total_level)
    else:
        result = glasses_heisenberg_plumbing_partition(*args.q, max_total_level=args.max_total_level)

    print(f"Heisenberg pair-of-pants plumbing: {result.channel}")
    print(f"  q = ({', '.join(format_complex(value) for value in result.q_values)})")
    print(f"  max total level = {result.max_total_level}")
    print(f"  chiral value = {format_complex(result.chiral_value)}")
    print(f"  nonchiral value = {result.nonchiral_value:.16e}")
    print(f"  nonzero level triples = {len(result.level_contributions)}")


if __name__ == "__main__":
    run()
