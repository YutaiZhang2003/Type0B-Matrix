#!/usr/bin/env python3
"""Compare exact genus-two NS PBW sewing with c-recursion for all ``p_i``.

The comparison uses the Human Note theta convention at fixed twice-levels
``ell=(A,C,E)``:

    a = A+C+E mod 2,
    parity_abs = a+p_1+p_2+p_3 mod 2,
    orientation = (-1)^Q(p+ell),
    Q(x) = x_1*x_2 + x_1*x_3 + x_2*x_3.

The local Ward tensor and Gram contraction are evaluated directly.  Any
parity-reversal convention sign of a local trinion appears twice in the sewn
block and cancels.  The c-recursion keeps ``p_i`` fixed; an odd null toggles
the child label ``a`` and transports ``Q(p+ell)`` by flipping the two adjacent
lifts.  Fusion polynomials are labelled directly by the same relative ``a``.
The absolute parity is recorded only for holomorphic/antiholomorphic matching.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from itertools import product

import sympy as sp

from ns_genus2_symbolic_low_order import (
    C,
    H0,
    H1,
    HINF,
    ExactDirectThetaOracle,
    ExactThetaRecursion,
    exact_regular_theta_coefficient,
    level_tuples,
    theta_orientation_sign,
)
from ns_human_convention import (
    absolute_three_form_parity,
    ns_null_factorization_sign,
)


PrimaryParities = tuple[int, int, int]
PARITY_TRIPLES: tuple[PrimaryParities, ...] = tuple(
    product((0, 1), repeat=3)
)
KAC_CHANNELS = ((3, 1, 3), (2, 2, 4), (5, 1, 5))


def relative_block_label(levels: tuple[int, int, int]) -> int:
    """Return the note's block/fusion label ``a=A+C+E`` modulo two."""

    return sum(levels) % 2


def absolute_parity(
    relative_label: int, primary_parities: PrimaryParities
) -> int:
    """Return ``a+p_1+p_2+p_3`` for left/right parity matching only."""

    return absolute_three_form_parity(relative_label, primary_parities)


def fusion_polynomial_label(relative_label: int) -> int:
    """The current note uses the block label ``a`` directly on ``P_rs^a``."""

    if relative_label not in (0, 1):
        raise ValueError("relative_label must be zero or one")
    return relative_label


def odd_null_transport_sign(
    *,
    levels: tuple[int, int, int],
    primary_parities: PrimaryParities,
    edge: int,
) -> int:
    r"""Return the two-adjacent-edge sign for an odd null on ``edge``."""

    if edge not in (0, 1, 2):
        raise ValueError("edge must be zero, one, or two")
    exponent = sum(
        (levels[other] + primary_parities[other]) % 2
        for other in range(3)
        if other != edge
    )
    return -1 if exponent % 2 else 1


@dataclass(frozen=True)
class ArbitraryPrimaryParityCheckSummary:
    maximum_total_twice_level: int
    parity_assignments: int
    coefficient_count: int
    direct_recursion_zero_count: int
    regular_seed_covariance_count: int
    direct_orientation_covariance_count: int
    relative_label_projector_count: int
    absolute_parity_count: int
    fusion_label_identity_count: int
    null_orientation_transport_count: int
    fusion_crossing_square_count: int
    checked_kac_channels: tuple[str, ...]


def run_checks(
    maximum_total_twice_level: int = 6,
) -> ArbitraryPrimaryParityCheckSummary:
    """Run exact symbolic direct-PBW/c-recursion comparisons for all ``p_i``."""

    if maximum_total_twice_level < 0 or maximum_total_twice_level > 6:
        raise ValueError("this exact low-order oracle supports total twice-level 0..6")
    weights = (H0, H1, HINF)
    direct = ExactDirectThetaOracle(c=C, weights=weights)
    recursive = ExactThetaRecursion()

    direct_recursion_count = 0
    regular_count = 0
    covariance_count = 0
    projector_count = 0
    absolute_count = 0
    label_count = 0
    transport_count = 0
    crossing_count = 0
    levels_to_check = tuple(level_tuples(maximum_total_twice_level))

    for levels in levels_to_check:
        zero_primary_direct = direct.coefficient(levels)
        relative_label_at_level = sum(levels) % 2
        zero_primary_regular = exact_regular_theta_coefficient(
            weights=weights,
            levels=levels,
            sectors=(relative_label_at_level, relative_label_at_level),
        )
        zero_orientation = theta_orientation_sign(levels)

        for primary_parities in PARITY_TRIPLES:
            relative_label = relative_block_label(levels)
            sectors = (relative_label, relative_label)
            direct_value = direct.coefficient(levels, primary_parities)
            recursive_value = recursive.coefficient(
                c=C,
                weights=weights,
                levels=levels,
                sectors=sectors,
                primary_parities=primary_parities,
            )
            difference = sp.cancel(sp.together(direct_value - recursive_value))
            if difference != 0:
                raise AssertionError(
                    "arbitrary-primary direct/c-recursion mismatch: "
                    f"levels={levels}, p={primary_parities}, "
                    f"difference={sp.factor(difference)}"
                )
            direct_recursion_count += 1

            orientation_ratio = sp.Rational(
                theta_orientation_sign(levels, primary_parities),
                zero_orientation,
            )
            if sp.cancel(
                direct_value - orientation_ratio * zero_primary_direct
            ) != 0:
                raise AssertionError(
                    f"direct orientation covariance failed at {levels}, {primary_parities}"
                )
            covariance_count += 1

            regular_value = exact_regular_theta_coefficient(
                weights=weights,
                levels=levels,
                sectors=sectors,
                primary_parities=primary_parities,
            )
            if sp.cancel(
                regular_value - orientation_ratio * zero_primary_regular
            ) != 0:
                raise AssertionError(
                    f"regular seed covariance failed at {levels}, {primary_parities}"
                )
            regular_count += 1

            rejected = recursive.coefficient(
                c=C,
                weights=weights,
                levels=levels,
                sectors=(relative_label ^ 1, relative_label ^ 1),
                primary_parities=primary_parities,
            )
            if rejected != 0:
                raise AssertionError(
                    f"relative label projector failed at {levels}, {primary_parities}"
                )
            projector_count += 1

            if absolute_parity(relative_label, primary_parities) != (
                sum(levels) + sum(primary_parities)
            ) % 2:
                raise AssertionError(
                    f"absolute parity failed at {levels}, {primary_parities}"
                )
            absolute_count += 1

            fusion_label = fusion_polynomial_label(relative_label)
            if fusion_label != relative_label_at_level:
                raise AssertionError(
                    f"fusion label failed at {levels}, {primary_parities}"
                )
            label_count += 1

            for edge, edge_level in enumerate(levels):
                for _r, _s, rs in KAC_CHANNELS:
                    if rs > edge_level:
                        continue
                    shifted_levels = list(levels)
                    shifted_levels[edge] -= rs
                    exact_transport = sp.Rational(
                        theta_orientation_sign(levels, primary_parities),
                        theta_orientation_sign(
                            shifted_levels, primary_parities
                        ),
                    )
                    expected_transport = (
                        odd_null_transport_sign(
                            levels=levels,
                            primary_parities=primary_parities,
                            edge=edge,
                        )
                        if rs % 2
                        else 1
                    )
                    if exact_transport != expected_transport:
                        raise AssertionError(
                            "null orientation transport failed: "
                            f"levels={levels}, p={primary_parities}, "
                            f"edge={edge}, rs={rs}"
                        )
                    transport_count += 1

                    child_descendants = tuple(
                        shifted_levels[index] % 2 for index in range(3)
                    )
                    endpoint_sign = ns_null_factorization_sign(
                        slot=edge,
                        null_parity=rs % 2,
                        descendant_parities=child_descendants,
                        primary_parities=primary_parities,
                    )
                    if endpoint_sign * endpoint_sign != 1:
                        raise AssertionError("squared local rho sign survived")
                    crossing_count += 1

    coefficient_count = len(levels_to_check) * len(PARITY_TRIPLES)
    return ArbitraryPrimaryParityCheckSummary(
        maximum_total_twice_level=maximum_total_twice_level,
        parity_assignments=len(PARITY_TRIPLES),
        coefficient_count=coefficient_count,
        direct_recursion_zero_count=direct_recursion_count,
        regular_seed_covariance_count=regular_count,
        direct_orientation_covariance_count=covariance_count,
        relative_label_projector_count=projector_count,
        absolute_parity_count=absolute_count,
        fusion_label_identity_count=label_count,
        null_orientation_transport_count=transport_count,
        fusion_crossing_square_count=crossing_count,
        checked_kac_channels=("(3,1)", "(2,2)", "(5,1)"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-twice-level", type=int, default=6)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    summary = run_checks(args.max_twice_level)
    if args.json:
        print(json.dumps(asdict(summary), indent=2))
        return
    print("arbitrary-primary NS theta c-recursion check: PASS")
    print(f"  primary parity assignments: {summary.parity_assignments}")
    print(
        "  exact direct-PBW/c-recursion identities: "
        f"{summary.direct_recursion_zero_count}/{summary.coefficient_count}"
    )
    print(
        "  regular-seed parity identities: "
        f"{summary.regular_seed_covariance_count}"
    )
    print(
        "  null orientation transports: "
        f"{summary.null_orientation_transport_count}"
    )
    print(
        "  squared fusion-crossing signs: "
        f"{summary.fusion_crossing_square_count}"
    )
    print(f"  Kac channels: {', '.join(summary.checked_kac_channels)}")


if __name__ == "__main__":
    main()
