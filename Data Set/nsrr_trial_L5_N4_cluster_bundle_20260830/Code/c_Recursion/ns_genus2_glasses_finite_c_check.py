#!/usr/bin/env python3
"""Independent finite-c Ward/Gram sewing oracle for the NS glasses graph.

The direct side of this check uses only NS PBW bases, super-Virasoro Gram
matrices, and descendant Ward identities.  At the left trinion it traces the
two incidences of the left handle with one inverse Gram matrix; it does the
same at the right trinion, and then sews the two resulting bridge vectors with
the bridge inverse Gram matrix.  No c-recursion residue or fusion polynomial
enters this contraction.

The comparison side evaluates the production functional c-recursion at small
but Schottky-stable plumbing coordinates.  Direct series are shown at several
total twice-level cutoffs; their difference from the resummed recursion must
decrease with the cutoff.  This is a low-level graph/orientation oracle, not a
replacement for the integrated momentum and recursion convergence scan.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from ns_genus12_finite_c_check import (
    NSDescendantThreeForm,
    NSVacuumModule,
    NumericNSVermaModule,
    State,
    level_tuples,
)
from ns_genus2_partition import (
    GLASSES_ORIENTATION,
    NSGenus2CRecursion,
)
from ns_global_osp_block import osp_norm, osp_three_point


def glasses_orientation_sign(twice_levels: Sequence[int]) -> int:
    """Return the glasses Koszul orientation for one level triple."""

    if len(twice_levels) != 3:
        raise ValueError("glasses graph has three edge levels")
    fermions = tuple(int(level) % 2 for level in twice_levels)
    return (-1) ** GLASSES_ORIENTATION.exponent(fermions)


def global_glasses_coefficient(
    *,
    weights: Sequence[complex],
    twice_levels: Sequence[int],
    sector: int,
    lifts: Sequence[int] = (1, 1, 1),
) -> complex:
    """One coefficient of the analytic global glasses network."""

    if len(weights) != 3 or len(twice_levels) != 3 or len(lifts) != 3:
        raise ValueError("glasses sewing requires three weights, levels, and lifts")
    levels = tuple(int(value) for value in twice_levels)
    if sector not in (0, 1):
        raise ValueError("sector must be zero or one")
    if levels[2] % 2 != sector:
        return 0.0 + 0.0j
    h_left, h_right, h_bridge = (complex(value) for value in weights)
    occupations = tuple(level // 2 for level in levels)
    fermions = tuple(level % 2 for level in levels)
    n_left, n_right, n_bridge = occupations
    e_left, e_right, e_bridge = fermions
    left = osp_three_point(
        n1=n_left,
        n2=n_bridge,
        n3=n_left,
        epsilon1=e_left,
        epsilon2=e_bridge,
        epsilon3=e_left,
        d1=h_left,
        d2=h_bridge,
        d3=h_left,
    )
    right = osp_three_point(
        n1=n_right,
        n2=n_bridge,
        n3=n_right,
        epsilon1=e_right,
        epsilon2=e_bridge,
        epsilon3=e_right,
        d1=h_right,
        d2=h_bridge,
        d3=h_right,
    )
    denominator = (
        osp_norm(h_left, n_left, e_left)
        * osp_norm(h_right, n_right, e_right)
        * osp_norm(h_bridge, n_bridge, e_bridge)
    )
    lift = math.prod(
        int(sign) ** fermion for sign, fermion in zip(lifts, fermions)
    )
    return complex(
        glasses_orientation_sign(levels) * lift * left * right / denominator
    )


class DirectGlassesOracle:
    """Direct finite-c PBW/Gram/Ward sewing for the glasses graph."""

    def __init__(
        self,
        *,
        c: complex,
        weights: Sequence[complex],
        vacuum: bool = False,
    ) -> None:
        if len(weights) != 3:
            raise ValueError("glasses sewing requires (h_left,h_right,h_bridge)")
        self.c = complex(c)
        self.vacuum = bool(vacuum)
        self.weights = tuple(complex(value) for value in weights)
        h_left, h_right, h_bridge = self.weights
        if self.vacuum:
            if any(weight != 0.0 for weight in self.weights):
                raise ValueError("vacuum quotient requires zero weights")
            self.modules = tuple(NSVacuumModule(c=self.c) for _ in range(3))
        else:
            self.modules = tuple(
                NumericNSVermaModule(c=self.c, weight=weight)
                for weight in self.weights
            )
        self.forms = (
            NSDescendantThreeForm(
                c=self.c,
                bra_weight=h_left,
                middle_weight=h_bridge,
                ket_weight=h_left,
                vacuum=self.vacuum,
            ),
            NSDescendantThreeForm(
                c=self.c,
                bra_weight=h_right,
                middle_weight=h_bridge,
                ket_weight=h_right,
                vacuum=self.vacuum,
            ),
        )

    @lru_cache(maxsize=None)
    def basis(self, edge: int, twice_level: int) -> tuple[State, ...]:
        return tuple(self.modules[int(edge)].basis(int(twice_level)))

    @lru_cache(maxsize=None)
    def inverse_gram(self, edge: int, twice_level: int) -> np.ndarray:
        module = self.modules[int(edge)]
        if isinstance(module, NumericNSVermaModule):
            return module.numeric_inverse_gram(int(twice_level))
        return np.linalg.inv(
            np.asarray(module.gram_matrix(int(twice_level)), dtype=np.complex128)
        )

    @lru_cache(maxsize=None)
    def _handle_bridge_vector(
        self,
        side: int,
        handle_twice_level: int,
        bridge_twice_level: int,
    ) -> np.ndarray:
        """Trace one handle and leave its bridge incidence uncontracted."""

        side = int(side)
        if side not in (0, 1):
            raise ValueError("side must be zero (left) or one (right)")
        handle_edge = side
        handle_basis = self.basis(handle_edge, int(handle_twice_level))
        bridge_basis = self.basis(2, int(bridge_twice_level))
        if not handle_basis or not bridge_basis:
            return np.zeros(len(bridge_basis), dtype=np.complex128)
        tensor = np.empty(
            (len(handle_basis), len(bridge_basis), len(handle_basis)),
            dtype=np.complex128,
        )
        form = self.forms[side]
        for bra_index, bra in enumerate(handle_basis):
            for bridge_index, bridge in enumerate(bridge_basis):
                for ket_index, ket in enumerate(handle_basis):
                    tensor[bra_index, bridge_index, ket_index] = form.value(
                        bra, bridge, ket
                    )
        # This is the same self-loop trace as G^{AB} rho(B,external,A)
        # in the independently tested torus one-point oracle.
        return np.einsum(
            "ad,dba->b",
            self.inverse_gram(handle_edge, int(handle_twice_level)),
            tensor,
            optimize=True,
        )

    @lru_cache(maxsize=None)
    def coefficient(
        self,
        twice_levels: tuple[int, int, int],
        sector: int,
        lifts: tuple[int, int, int] = (1, 1, 1),
    ) -> complex:
        """Directly sew one finite-c glasses coefficient.

        Edge order is ``(left handle,right handle,bridge)``.  The handle
        incidences occur twice at a vertex, so the three-form sector is fixed
        solely by the bridge parity.
        """

        levels = tuple(int(value) for value in twice_levels)
        lift_values = tuple(int(value) for value in lifts)
        if len(levels) != 3 or len(lift_values) != 3:
            raise ValueError("glasses sewing requires three levels and lifts")
        if sector not in (0, 1):
            raise ValueError("sector must be zero or one")
        if any(value not in (-1, 1) for value in lift_values):
            raise ValueError("lifts must be +/-1")
        if self.vacuum and sector != 0:
            return 0.0 + 0.0j
        if levels[2] % 2 != sector:
            return 0.0 + 0.0j
        left = self._handle_bridge_vector(0, levels[0], levels[2])
        right = self._handle_bridge_vector(1, levels[1], levels[2])
        if not len(left) or not len(right):
            return 0.0 + 0.0j
        contracted = np.einsum(
            "b,be,e->",
            left,
            self.inverse_gram(2, levels[2]),
            right,
            optimize=True,
        )
        fermions = tuple(level % 2 for level in levels)
        lift = math.prod(
            sign**fermion for sign, fermion in zip(lift_values, fermions)
        )
        return complex(glasses_orientation_sign(levels) * lift * contracted)

    def truncated_block(
        self,
        *,
        q_values: Sequence[complex],
        sector: int,
        max_total_twice_level: int,
        lifts: Sequence[int] = (1, 1, 1),
    ) -> complex:
        """Sum the direct PBW series through one total twice-level."""

        if len(q_values) != 3:
            raise ValueError("glasses sewing requires three q values")
        q_tuple = tuple(complex(value) for value in q_values)
        lift_tuple = tuple(int(value) for value in lifts)
        return sum(
            (
                self.coefficient(levels, int(sector), lift_tuple)
                * math.prod(
                    q ** (level / 2.0) for q, level in zip(q_tuple, levels)
                )
                for levels in level_tuples(int(max_total_twice_level))
            ),
            0.0 + 0.0j,
        )


@dataclass(frozen=True)
class ComparisonRow:
    sector: int
    max_total_twice_level: int
    direct_real: float
    direct_imag: float
    recursive_real: float
    recursive_imag: float
    absolute_difference: float


def run_check(
    *,
    c: complex = 37.25,
    weights: Sequence[complex] = (0.73, 0.91, 1.17),
    q_values: Sequence[complex] = (
        0.0040 + 0.0003j,
        0.0035 - 0.0002j,
        0.0042 + 0.00025j,
    ),
    cutoffs: Sequence[int] = (4, 6, 8),
    lifts: Sequence[int] = (1, 1, 1),
) -> dict:
    """Compare the direct truncated series with the functional recursion."""

    weight_tuple = tuple(complex(value) for value in weights)
    q_tuple = tuple(complex(value) for value in q_values)
    lift_tuple = tuple(int(value) for value in lifts)
    oracle = DirectGlassesOracle(c=c, weights=weight_tuple)
    rows = []
    for sector in (0, 1):
        previous_difference = math.inf
        for cutoff in tuple(int(value) for value in cutoffs):
            direct = oracle.truncated_block(
                q_values=q_tuple,
                sector=sector,
                max_total_twice_level=cutoff,
                lifts=lift_tuple,
            )
            recursion = NSGenus2CRecursion(
                channel="glasses",
                q_values=q_tuple,
                global_method="resummed",
                global_tolerance=1.0e-16,
                global_max_total_occupation=22,
                vacuum_word_length=12,
                vacuum_max_mode=70,
            )
            recursive = recursion.collision_aware_block_mp(
                weights=weight_tuple,
                sector=sector,
                recursion_order=cutoff,
                lifts=lift_tuple,
                central_charge=c,
                working_precision=70,
            )
            difference = float(abs(direct - recursive))
            if difference >= previous_difference:
                raise AssertionError(
                    "direct glasses truncation did not approach the functional "
                    f"recursion in sector {sector}: {difference} >= "
                    f"{previous_difference}"
                )
            previous_difference = difference
            rows.append(
                ComparisonRow(
                    sector=sector,
                    max_total_twice_level=cutoff,
                    direct_real=float(direct.real),
                    direct_imag=float(direct.imag),
                    recursive_real=float(recursive.real),
                    recursive_imag=float(recursive.imag),
                    absolute_difference=difference,
                )
            )

    global_errors = []
    for levels in level_tuples(2):
        sector = levels[2] % 2
        direct = oracle.coefficient(levels, sector, lift_tuple)
        expected = global_glasses_coefficient(
            weights=weight_tuple,
            twice_levels=levels,
            sector=sector,
            lifts=lift_tuple,
        )
        global_errors.append(abs(direct - expected))
    return {
        "status": "pass",
        "scope": "independent finite-c PBW/Gram/Ward glasses sewing",
        "central_charge": {"real": complex(c).real, "imag": complex(c).imag},
        "weights": [[value.real, value.imag] for value in weight_tuple],
        "q_values": [[value.real, value.imag] for value in q_tuple],
        "lifts": list(lift_tuple),
        "rows": [asdict(row) for row in rows],
        "global_coefficient_max_absolute_error_through_twice_level_2": float(
            max(global_errors, default=0.0)
        ),
        "integrated_convergence_certified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_check()
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
