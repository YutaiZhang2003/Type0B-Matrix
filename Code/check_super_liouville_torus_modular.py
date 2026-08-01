#!/usr/bin/env python3
"""Run the Type-0B ordinary-NS torus one-point modular S test."""

from __future__ import annotations

import argparse

from super_liouville_torus_one_point import (
    run_type0b_ns_modular_s_check,
    run_type0b_ns_modular_s_convergence,
)


def _format(value: complex) -> str:
    if abs(value.imag) < 1.0e-14:
        return f"{value.real:.15g}"
    return f"{value.real:.15g}{value.imag:+.15g}i"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare q and q_tilde frames for the NS torus one-point function."
    )
    parser.add_argument("--tau", type=complex, default=0.2 + 0.9j)
    parser.add_argument("--external-momentum", type=float, default=0.33)
    parser.add_argument("--max-twice-level", type=int, default=12)
    parser.add_argument("--p-max", type=float, default=4.5)
    parser.add_argument("--quadrature-order", type=int, default=48)
    parser.add_argument("--structure-precision", type=int, default=35)
    parser.add_argument("--finite-part-samples", type=int, default=24)
    parser.add_argument(
        "--show-convergence",
        action="store_true",
        help="also print the shared-quadrature levels 6,8,10,12 ledger",
    )
    args = parser.parse_args()

    result = run_type0b_ns_modular_s_check(
        tau=args.tau,
        external_momentum=args.external_momentum,
        max_twice_level=args.max_twice_level,
        p_max=args.p_max,
        quadrature_order=args.quadrature_order,
        structure_precision=args.structure_precision,
        finite_part_samples=args.finite_part_samples,
    )
    print("Type-0B NS torus one-point modular S check")
    print(f"  tau={_format(result.tau)}")
    print(f"  S tau={_format(result.s_tau)}")
    print(f"  q={_format(result.q)}")
    print(f"  q_tilde={_format(result.q_tilde)}")
    print(
        "  NS lift signs="
        f"({result.lift_sign:+d}, {result.lift_sign_tilde:+d})"
    )
    print(f"  external weight={_format(result.external_weight)}")
    print(
        f"  maximum twice-level={result.max_twice_level} "
        f"(through q^{result.max_twice_level / 2:g})"
    )
    print(f"  G(q)={_format(result.value_q)}")
    print(f"  G(q_tilde)={_format(result.value_q_tilde)}")
    print(f"  numeric ratio={_format(result.numeric_ratio)}")
    print(f"  |tau|^(2d)={result.expected_ratio:.15g}")
    print(f"  relative error={abs(result.relative_error):.6e}")

    if args.show_convergence:
        print("")
        print("  shared-quadrature convergence")
        for row in run_type0b_ns_modular_s_convergence(
            tau=args.tau,
            external_momentum=args.external_momentum,
            p_max=args.p_max,
            quadrature_order=args.quadrature_order,
            structure_precision=args.structure_precision,
            finite_part_samples=args.finite_part_samples,
        ):
            print(
                f"    twice-level {row.max_twice_level:2d}: "
                f"G(q)={_format(row.value_q)}, "
                f"relative error={abs(row.relative_error):.6e}"
            )


if __name__ == "__main__":
    main()
