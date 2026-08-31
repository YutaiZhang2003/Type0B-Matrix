#!/usr/bin/env python3
r"""Low-level PBW audit of the all-NS glasses-channel c-recursion.

The direct side constructs super-Virasoro PBW bases, Gram matrices, and both
descendant three-forms.  The recursion side uses an independently extracted
large-c vacuum quotient, the analytic global osp(1|2) network, and local Kac
residues.  No finite-c PBW coefficient is used as a recursion input.

Edge order is ``(left handle, right handle, separating bridge)``.  Twice-level
vectors are used, so ``max_total_twice_level=8`` means total physical level 4.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from genus_2.glasses_partition import glasses_null_transport
from ns_genus12_finite_c_check import level_tuples
from ns_genus2_glasses_finite_c_check import (
    DirectGlassesOracle,
    global_glasses_coefficient,
)
from ns_genus_c_recursion_checks import (
    ns_c_pole,
    ns_fusion_polynomial,
    ns_inverse_null_slope,
)


Level = tuple[int, int, int]


@lru_cache(maxsize=None)
def extract_glasses_vacuum_seed(
    max_total_twice_level: int,
    c_samples: tuple[float, ...] = (
        2000.0,
        4000.0,
        8000.0,
        16000.0,
        32000.0,
    ),
) -> dict[Level, complex]:
    """Extract the glasses vacuum quotient without finite-c Kac residues."""

    cutoff = int(max_total_twice_level)
    if cutoff < 0:
        raise ValueError("the level cutoff must be nonnegative")
    levels = tuple(level_tuples(cutoff))
    samples = np.asarray(c_samples, dtype=float)
    if len(samples) < 2:
        raise ValueError("at least two large-c samples are required")

    values: list[list[complex]] = []
    for central_charge in samples:
        oracle = DirectGlassesOracle(
            c=central_charge,
            weights=(0.0, 0.0, 0.0),
            vacuum=True,
        )
        values.append(
            [
                oracle.coefficient(level, 0, (1, 1, 1))
                for level in levels
            ]
        )
    richardson = np.column_stack(
        [samples ** (-power) for power in range(len(samples))]
    )
    limits = np.linalg.solve(richardson, np.asarray(values))[0]

    result: dict[Level, complex] = {}
    for level, value in zip(levels, limits):
        candidate = complex(value)
        nearest = round(candidate.real)
        if abs(candidate.imag) < 1.0e-9 and abs(candidate.real - nearest) < 1.0e-7:
            candidate = complex(nearest)
        result[level] = candidate
    return result


def regular_glasses_coefficient(
    *,
    weights: Sequence[complex],
    twice_levels: Sequence[int],
    sector: int,
    lifts: Sequence[int] = (1, 1, 1),
    vacuum_seed: dict[Level, complex] | None = None,
) -> complex:
    """Large-c coefficient from vacuum/global graded factorization."""

    levels = tuple(int(value) for value in twice_levels)
    lift_tuple = tuple(int(value) for value in lifts)
    if len(levels) != 3 or len(lift_tuple) != 3:
        raise ValueError("glasses data must have three entries")
    if sector not in (0, 1) or levels[2] % 2 != sector:
        return 0.0 + 0.0j
    if any(lift not in (-1, 1) for lift in lift_tuple):
        raise ValueError("lifts must be +/-1")
    seed = (
        extract_glasses_vacuum_seed(sum(levels))
        if vacuum_seed is None
        else vacuum_seed
    )
    lifted_seed = {
        level: coefficient
        * math.prod(lift ** (entry % 2) for lift, entry in zip(lift_tuple, level))
        for level, coefficient in seed.items()
    }
    # Unlike the theta graph, glasses self-loop factorization supplies the
    # Koszul factor internally at each trinion.  Applying the polarization of
    # Q_gl=e_B(e_L+e_R) once more would double-count it.  Direct PBW fixes the
    # regular seed to this ordinary convolution, beginning at (0,3,1).
    total = 0.0 + 0.0j
    for vacuum_level, vacuum_coefficient in lifted_seed.items():
        remainder = tuple(
            levels[edge] - vacuum_level[edge] for edge in range(3)
        )
        if min(remainder) < 0:
            continue
        total += vacuum_coefficient * global_glasses_coefficient(
            weights=weights,
            twice_levels=remainder,
            sector=sector,
            lifts=lift_tuple,
        )
    return total


def glasses_residue_prefactor(
    *,
    r: int,
    s: int,
    edge: int,
    weights: Sequence[complex],
    sector: int,
) -> tuple[complex, complex, int]:
    r"""Return ``(c_pole, J A P_1 P_2, child_sector)``.

    A handle is a self-loop.  Its two incidences must be factorized in order:
    the second polynomial sees both the shifted handle weight and, for an odd
    null, the toggled intermediate three-form label.  The final label toggles
    twice and is therefore unchanged.  The bridge meets distinct vertices;
    both polynomials retain the original label and the child label toggles
    once for an odd null.
    """

    weight_tuple = tuple(complex(value) for value in weights)
    if len(weight_tuple) != 3 or edge not in (0, 1, 2):
        raise ValueError("invalid glasses edge or weight tuple")
    if sector not in (0, 1):
        raise ValueError("sector must be zero or one")
    rs = int(r) * int(s)
    delta = rs % 2
    pole = ns_c_pole(r, s, weight_tuple[edge])

    if edge in (0, 1):
        handle = weight_tuple[edge]
        bridge = weight_tuple[2]
        first = ns_fusion_polynomial(
            r=r,
            s=s,
            a=sector,
            first_weight=bridge,
            second_weight=handle,
            b=pole.b,
        )
        second = ns_fusion_polynomial(
            r=r,
            s=s,
            a=sector ^ delta,
            first_weight=bridge,
            second_weight=handle + rs / 2.0,
            b=pole.b,
        )
        child_sector = sector
    else:
        first = ns_fusion_polynomial(
            r=r,
            s=s,
            a=sector,
            first_weight=weight_tuple[0],
            second_weight=weight_tuple[0],
            b=pole.b,
        )
        second = ns_fusion_polynomial(
            r=r,
            s=s,
            a=sector,
            first_weight=weight_tuple[1],
            second_weight=weight_tuple[1],
            b=pole.b,
        )
        child_sector = sector ^ delta

    residue = (
        pole.jacobian
        * ns_inverse_null_slope(r, s, pole.b)
        * first
        * second
    )
    return complex(pole.c), complex(residue), child_sector


def recursion_glasses_coefficient(
    *,
    c: complex,
    weights: Sequence[complex],
    twice_levels: Sequence[int],
    sector: int,
    lifts: Sequence[int] = (1, 1, 1),
    vacuum_seed: dict[Level, complex] | None = None,
) -> complex:
    """Evaluate the derived glasses c-recursion at one coefficient."""

    initial_weights = tuple(complex(value) for value in weights)
    initial_levels = tuple(int(value) for value in twice_levels)
    lift_tuple = tuple(int(value) for value in lifts)
    if vacuum_seed is None:
        vacuum_seed = extract_glasses_vacuum_seed(sum(initial_levels))

    @lru_cache(maxsize=None)
    def recurse(
        current_c: complex,
        current_weights: tuple[complex, complex, complex],
        levels: Level,
        current_sector: int,
    ) -> complex:
        if min(levels) < 0 or levels[2] % 2 != current_sector:
            return 0.0 + 0.0j
        total = regular_glasses_coefficient(
            weights=current_weights,
            twice_levels=levels,
            sector=current_sector,
            lifts=lift_tuple,
            vacuum_seed=vacuum_seed,
        )
        for edge, edge_level in enumerate(levels):
            for r in range(2, edge_level + 1):
                for s in range(1, edge_level // r + 1):
                    rs = r * s
                    if rs > edge_level or (r + s) % 2:
                        continue
                    pole_c, residue, child_sector = glasses_residue_prefactor(
                        r=r,
                        s=s,
                        edge=edge,
                        weights=current_weights,
                        sector=current_sector,
                    )
                    denominator = current_c - pole_c
                    if abs(denominator) < 1.0e-13:
                        raise ZeroDivisionError(
                            "the PBW audit hit a coincident c-pole; detune weights"
                        )
                    shifted_levels = list(levels)
                    shifted_levels[edge] -= rs
                    shifted_weights = list(current_weights)
                    shifted_weights[edge] += rs / 2.0
                    delta = rs % 2

                    transport = glasses_null_transport(
                        sector=current_sector,
                        lifts=lift_tuple,
                        edge=edge,
                        rs=rs,
                    )
                    if transport.child_sector != child_sector:
                        raise AssertionError(
                            "local residue and centralized parity transport disagree"
                        )
                    # Evaluate the lift change on the homogeneous child
                    # coefficient.  For an odd handle null this is (-1)^a;
                    # for an odd bridge null it is one.
                    sewing_transport = math.prod(
                        (child_lift * parent_lift) ** (level % 2)
                        for child_lift, parent_lift, level in zip(
                            transport.child_lifts,
                            lift_tuple,
                            shifted_levels,
                        )
                    )
                    null_lift = lift_tuple[edge] ** delta
                    total += (
                        null_lift
                        * sewing_transport
                        * residue
                        / denominator
                        * recurse(
                            pole_c,
                            tuple(shifted_weights),
                            tuple(shifted_levels),
                            child_sector,
                        )
                    )
        return total

    return recurse(
        complex(c), initial_weights, initial_levels, int(sector)
    )


def run_level_four_audit(
    *,
    c: complex = 37.25,
    weights: Sequence[complex] = (0.73, 0.91, 1.17),
    max_total_twice_level: int = 8,
    lifts_to_test: Sequence[Sequence[int]] = (
        (1, 1, 1),
        (-1, 1, 1),
        (1, -1, 1),
        (1, 1, -1),
        (-1, -1, 1),
        (-1, 1, -1),
        (1, -1, -1),
        (-1, -1, -1),
    ),
    tolerance: float = 2.0e-7,
) -> dict[str, object]:
    """Compare every glasses coefficient through total physical level 4."""

    cutoff = int(max_total_twice_level)
    weight_tuple = tuple(complex(value) for value in weights)
    vacuum_seed = extract_glasses_vacuum_seed(cutoff)
    rows: list[dict[str, object]] = []
    maximum_error = 0.0
    maximum_relative_error = 0.0

    for lifts in lifts_to_test:
        lift_tuple = tuple(int(value) for value in lifts)
        oracle = DirectGlassesOracle(c=c, weights=weight_tuple)
        lift_maximum = 0.0
        for levels in level_tuples(cutoff):
            sector = levels[2] % 2
            direct = oracle.coefficient(levels, sector, lift_tuple)
            recursive = recursion_glasses_coefficient(
                c=c,
                weights=weight_tuple,
                twice_levels=levels,
                sector=sector,
                lifts=lift_tuple,
                vacuum_seed=vacuum_seed,
            )
            error = float(abs(direct - recursive))
            relative = error / max(1.0, abs(direct), abs(recursive))
            maximum_error = max(maximum_error, error)
            maximum_relative_error = max(maximum_relative_error, relative)
            lift_maximum = max(lift_maximum, error)
            if error > tolerance * max(1.0, abs(direct), abs(recursive)):
                raise AssertionError(
                    "direct PBW/glasses c-recursion mismatch: "
                    f"levels={levels}, sector={sector}, lifts={lift_tuple}, "
                    f"direct={direct!r}, recursive={recursive!r}, error={error:.3e}"
                )
        rows.append(
            {
                "lifts": list(lift_tuple),
                "coefficient_count": math.comb(cutoff + 3, 3),
                "maximum_absolute_error": lift_maximum,
            }
        )

    vacuum_rounding_error = max(
        min(abs(value.real - round(value.real)), abs(value.real))
        + abs(value.imag)
        for value in vacuum_seed.values()
    )
    return {
        "status": "pass",
        "scope": "all-NS glasses direct PBW versus derived c-recursion",
        "central_charge": [complex(c).real, complex(c).imag],
        "weights": [[value.real, value.imag] for value in weight_tuple],
        "max_total_twice_level": cutoff,
        "max_total_physical_level": cutoff / 2.0,
        "sector_rule": "a = n_bridge mod 2",
        "handle_odd_null_transport": "sector fixed; flip bridge lift, giving (-1)^a for even primaries",
        "bridge_odd_null_transport": "toggle sector; no spectator-lift flip",
        "coefficient_comparisons": len(rows) * math.comb(cutoff + 3, 3),
        "maximum_absolute_error": maximum_error,
        "maximum_relative_error": maximum_relative_error,
        "vacuum_seed_nearest_integer_diagnostic": vacuum_rounding_error,
        "lift_rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-total-twice-level", type=int, default=8)
    args = parser.parse_args()
    result = run_level_four_audit(
        max_total_twice_level=args.max_total_twice_level
    )
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
