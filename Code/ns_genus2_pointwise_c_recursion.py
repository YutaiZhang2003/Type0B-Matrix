#!/usr/bin/env python3
"""Pointwise all-NS genus-two c-recursion in the human CCY theta frame.

Unlike the coefficient audit, this evaluator inserts ``q``, the three lifts,
and the NS sector before recursing.  Its order limits only the accumulated
twice-level of Kac residues.  Every recursion leaf is the fully resummed
vacuum/global regular block.

The regular block is assembled in the theta-polarized ``star`` algebra.  For
a fixed global parity ``p``, the cocycle can be absorbed into a parity-
dependent flip of the vacuum lifts, so only four Schottky-vacuum evaluations
are needed for one sector rather than a coefficient table.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import product
import math
from typing import Sequence

from ns_genus12_finite_c_check import theta_residue_prefactor
from ns_genus2_partition import resummed_theta_global_component
from ns_human_convention import (
    normalize_parity_triple,
    theta_primary_parity_rephasing,
)
from ns_regular_block import THETA_ORIENTATION
from ns_vacuum_schottky import (
    ccy_theta_generators,
    ns_schottky_vacuum_block,
    theta_lift_signs,
)


Parity = tuple[int, int, int]
PARITIES: tuple[Parity, ...] = tuple(product((0, 1), repeat=3))


def _character(lifts: Sequence[int], parity: Sequence[int]) -> int:
    return math.prod(int(lift) ** int(bit) for lift, bit in zip(lifts, parity))


def _unit(edge: int) -> Parity:
    return tuple(int(index == int(edge)) for index in range(3))  # type: ignore[return-value]


def _transport_human_lifts(
    lifts: Sequence[int], edge: int
) -> tuple[int, int, int]:
    basis = _unit(edge)
    return tuple(
        int(lifts[target])
        * (-1) ** THETA_ORIENTATION.polarized_exponent(
            basis, _unit(target)
        )
        for target in range(3)
    )  # type: ignore[return-value]


class PointwiseHumanThetaCRecursion:
    """Functional NS c-recursion with the literal human theta convention.

    ``sector`` is always the note's relative label ``a=A+C+E mod 2``.
    Intrinsic primary parities are independent metadata and enter through
    ``eta_i^(A_i+p_i)`` and ``Q(A+p_1,C+p_2,E+p_3)``.
    """

    def __init__(
        self,
        *,
        q_values: Sequence[complex],
        global_tolerance: float = 1.0e-14,
        global_max_total_occupation: int = 24,
        vacuum_word_length: int = 8,
        vacuum_max_mode: int = 50,
    ) -> None:
        if len(q_values) != 3 or any(abs(complex(q)) >= 1 for q in q_values):
            raise ValueError("three plumbing coordinates with modulus below one are required")
        if global_tolerance <= 0 or global_max_total_occupation < 0:
            raise ValueError("invalid global-block convergence controls")
        self.q_values = tuple(complex(q) for q in q_values)
        self.global_tolerance = float(global_tolerance)
        self.global_max_total_occupation = int(global_max_total_occupation)
        self.vacuum_word_length = int(vacuum_word_length)
        self.vacuum_max_mode = int(vacuum_max_mode)
        self.block_calls = 0
        self.maximum_global_occupation_used = 0
        self.maximum_global_last_shell_relative = 0.0

    @lru_cache(maxsize=None)
    def _vacuum(self, human_lifts: tuple[int, int, int]) -> complex:
        # The vacuum table and this API both use literal human-note lifts;
        # ``theta_lift_signs`` performs the documented conversion to the raw
        # determinant-one Schottky representatives at the backend boundary.
        return ns_schottky_vacuum_block(
            ccy_theta_generators(*self.q_values),
            theta_lift_signs(human_lifts),
            max_word_length=self.vacuum_word_length,
            max_mode=self.vacuum_max_mode,
        ).value

    @lru_cache(maxsize=None)
    def _global_component(
        self,
        weights: tuple[complex, complex, complex],
        parity: Parity,
    ) -> complex:
        # The production resummation accepts geometric (zero,one,infinity)
        # labels and internally reverses them into CCY trinion slots.  Reverse
        # here so its trinion order is the human-note order used by the exact
        # coefficient audit.  Its lifts already are the literal human-note
        # lifts, so the unit lift tuple is passed unchanged.
        diagnostics = resummed_theta_global_component(
            weights=weights[::-1],
            q_values=self.q_values[::-1],
            fermions=parity[::-1],
            lifts=(1, 1, 1),
            tolerance=self.global_tolerance,
            max_total_endpoint_occupation=self.global_max_total_occupation,
        )
        self.maximum_global_occupation_used = max(
            self.maximum_global_occupation_used,
            diagnostics.max_total_occupation,
        )
        relative_shell = abs(diagnostics.last_shell) / max(
            1.0, abs(diagnostics.value)
        )
        self.maximum_global_last_shell_relative = max(
            self.maximum_global_last_shell_relative,
            float(relative_shell),
        )
        if not diagnostics.converged:
            raise RuntimeError(
                "pointwise theta global component failed to converge: "
                f"weights={weights!r}, parity={parity!r}, "
                f"last_shell={diagnostics.last_shell!r}"
            )
        return diagnostics.value

    def regular_block(
        self,
        *,
        weights: Sequence[complex],
        sector: int,
        lifts: Sequence[int],
        primary_parities: Sequence[int] = (0, 0, 0),
    ) -> complex:
        """Evaluate the resummed vacuum ``star`` global regular block."""

        if sector not in (0, 1):
            raise ValueError("sector must be zero or one")
        weight_tuple = tuple(complex(value) for value in weights)
        lift_tuple = tuple(int(value) for value in lifts)
        if len(weight_tuple) != 3 or len(lift_tuple) != 3:
            raise ValueError("three weights and three lifts are required")
        if any(value not in (-1, 1) for value in lift_tuple):
            raise ValueError("lifts must be +/-1")
        primaries = normalize_parity_triple(
            primary_parities, name="primary_parities"
        )
        if any(primaries):
            prefactor, effective_lifts = theta_primary_parity_rephasing(
                lift_tuple, primaries
            )
            return prefactor * self.regular_block(
                weights=weight_tuple,
                sector=sector,
                lifts=effective_lifts,
                primary_parities=(0, 0, 0),
            )

        total = 0.0 + 0.0j
        for parity in PARITIES:
            if sum(parity) % 2 != sector:
                continue
            # For B(r,p)=sum_{i<j}(r_i p_j+p_i r_j), fixing p makes
            # (-1)^B a character in r.  Absorb that character by flipping
            # vacuum lift i by (-1)^(sum_{j != i} p_j).
            vacuum_lifts = tuple(
                lift_tuple[edge]
                * (-1) ** sum(parity[other] for other in range(3) if other != edge)
                for edge in range(3)
            )
            total += (
                _character(lift_tuple, parity)
                * self._global_component(weight_tuple, parity)
                * self._vacuum(vacuum_lifts)
            )
        return total

    def precompute_vacuum_for(
        self,
        *,
        sector: int,
        lifts: Sequence[int],
        primary_parities: Sequence[int] = (0, 0, 0),
    ) -> None:
        """Cache the geometry-only vacuum factors needed by one block."""

        lift_tuple = tuple(int(value) for value in lifts)
        if sector not in (0, 1) or len(lift_tuple) != 3:
            raise ValueError("invalid sector or lift tuple")
        _prefactor, lift_tuple = theta_primary_parity_rephasing(
            lift_tuple,
            normalize_parity_triple(
                primary_parities, name="primary_parities"
            ),
        )
        for parity in PARITIES:
            if sum(parity) % 2 != sector:
                continue
            vacuum_lifts = tuple(
                lift_tuple[edge]
                * (-1) ** sum(
                    parity[other]
                    for other in range(3)
                    if other != edge
                )
                for edge in range(3)
            )
            self._vacuum(vacuum_lifts)

    def block(
        self,
        *,
        central_charge: complex,
        weights: Sequence[complex],
        sector: int,
        recursion_order: int,
        lifts: Sequence[int],
        primary_parities: Sequence[int] = (0, 0, 0),
    ) -> complex:
        """Insert all numerical data and evaluate the functional recursion."""

        order = int(recursion_order)
        if order < 0 or order > 16:
            raise ValueError("this comparison evaluator supports recursion orders 0 through 16")
        weight_tuple = tuple(complex(value) for value in weights)
        human_lifts = tuple(int(value) for value in lifts)
        if len(weight_tuple) != 3 or len(human_lifts) != 3:
            raise ValueError("three weights and three lifts are required")
        if sector not in (0, 1) or any(value not in (-1, 1) for value in human_lifts):
            raise ValueError("invalid sector or lift")
        primaries = normalize_parity_triple(
            primary_parities, name="primary_parities"
        )
        primary_prefactor, effective_lifts = theta_primary_parity_rephasing(
            human_lifts, primaries
        )

        @lru_cache(maxsize=None)
        def recurse(
            remaining: int,
            current_c: complex,
            current_weights: tuple[complex, complex, complex],
            current_sector: int,
            current_lifts: tuple[int, int, int],
        ) -> complex:
            self.block_calls += 1
            total = self.regular_block(
                weights=current_weights,
                sector=current_sector,
                lifts=current_lifts,
            )
            for edge in range(3):
                for r in range(2, remaining + 1):
                    for s in range(1, remaining // r + 1):
                        rs = r * s
                        if rs > remaining or (r + s) % 2:
                            continue
                        pole_c, residue = theta_residue_prefactor(
                            r=r,
                            s=s,
                            edge=edge,
                            weights=current_weights,
                            sectors=(current_sector, current_sector),
                        )
                        denominator = current_c - pole_c
                        if abs(denominator) < 1.0e-12:
                            raise ZeroDivisionError(
                                "the scalar pointwise benchmark hit a confluent c-pole"
                            )
                        shifted = list(current_weights)
                        shifted[edge] += rs / 2.0
                        child_sector = current_sector ^ (rs % 2)
                        if rs % 2:
                            child_lifts = _transport_human_lifts(
                                current_lifts, edge
                            )
                            orientation_constant = (-1) ** THETA_ORIENTATION.exponent(
                                _unit(edge)
                            )
                        else:
                            child_lifts = current_lifts
                            orientation_constant = 1
                        total += (
                            residue
                            / denominator
                            * self.q_values[edge] ** (rs / 2.0)
                            * current_lifts[edge] ** (rs % 2)
                            * orientation_constant
                            * recurse(
                                remaining - rs,
                                complex(pole_c),
                                tuple(shifted),
                                child_sector,
                                tuple(child_lifts),
                            )
                        )
            return total

        return primary_prefactor * recurse(
            order,
            complex(central_charge),
            weight_tuple,
            int(sector),
            effective_lifts,
        )
