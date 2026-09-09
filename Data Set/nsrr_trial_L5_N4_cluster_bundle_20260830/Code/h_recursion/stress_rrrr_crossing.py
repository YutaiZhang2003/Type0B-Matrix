"""Stress the physical Type-0B RRRR crossing equation.

The script compares

    G_4321(z, zbar) = G_4123(1-z, 1-zbar)

after the BRY structure constants, the dP/pi spectral measure, and all four
HJS chiral-sign branches have been assembled.  It supports both the symmetric
central-charge regulator and the direct coefficient-wise finite part at b=1.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, Sequence

from ramond_sphere_correlators import (
    SelfDualRRRRSphereCorrelator,
    SymmetricRRRRSphereCorrelator,
    relative_crossing_error,
)


@dataclass(frozen=True)
class CrossingResult:
    order: int
    z: complex
    left: complex
    right: complex

    @property
    def absolute_residual(self) -> float:
        return abs(self.left - self.right)

    @property
    def relative_residual(self) -> float:
        return relative_crossing_error(self.left, self.right)


def _parse_csv(text: str, caster) -> tuple:
    return tuple(caster(item.strip()) for item in text.split(",") if item.strip())


def _parse_complex(text: str) -> complex:
    return complex(text.replace("i", "j"))


def crossing_scan(
    *,
    momenta: Sequence[float],
    orders: Iterable[int],
    z_values: Sequence[complex],
    mode: str,
    structure_precision: int,
    p_max: float,
    quadrature_order: int,
    central_charge_shift: float,
    finite_part_radius: float,
    finite_part_check_radius: float,
    finite_part_samples: int,
) -> list[CrossingResult]:
    """Return the full order-by-position RRRR crossing ledger."""

    if len(momenta) != 4:
        raise ValueError("momenta must contain p1,p2,p3,p4")
    results: list[CrossingResult] = []
    common = dict(
        p1=momenta[0],
        p2=momenta[1],
        p3=momenta[2],
        p4=momenta[3],
        structure_precision=structure_precision,
    )
    for order in orders:
        if mode == "symmetric":
            correlator = SymmetricRRRRSphereCorrelator(
                **common,
                block_order=order,
                central_charge_shift=central_charge_shift,
            )
        elif mode == "self-dual":
            correlator = SelfDualRRRRSphereCorrelator(
                **common,
                block_order=order,
                finite_part_radius=finite_part_radius,
                finite_part_check_radius=finite_part_check_radius,
                finite_part_samples=finite_part_samples,
            )
        else:
            raise ValueError("mode must be 'symmetric' or 'self-dual'")
        crossed = correlator.crossed()
        for z in z_values:
            left = correlator.evaluate(
                z, p_max=p_max, quadrature_order=quadrature_order
            )
            right = crossed.evaluate(
                1.0 - z, p_max=p_max, quadrature_order=quadrature_order
            )
            results.append(
                CrossingResult(
                    order=int(order),
                    z=complex(z),
                    left=complex(left),
                    right=complex(right),
                )
            )
    return results


def _format_complex(value: complex) -> str:
    return f"{value.real:.12g}{value.imag:+.12g}i"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("symmetric", "self-dual"), default="symmetric")
    parser.add_argument("--momenta", default="0.20,0.35,0.40,0.55")
    parser.add_argument("--orders", default="6,8,10,12")
    parser.add_argument("--z", default="0.05,0.20,0.37,0.50,0.80,0.95")
    parser.add_argument("--structure-precision", type=int, default=30)
    parser.add_argument("--p-max", type=float, default=5.0)
    parser.add_argument("--quadrature-order", type=int, default=24)
    parser.add_argument("--central-charge-shift", type=float, default=1.0e-4)
    parser.add_argument("--finite-part-radius", type=float, default=0.04)
    parser.add_argument("--finite-part-check-radius", type=float, default=0.05)
    parser.add_argument("--finite-part-samples", type=int, default=24)
    args = parser.parse_args()

    results = crossing_scan(
        momenta=_parse_csv(args.momenta, float),
        orders=_parse_csv(args.orders, int),
        z_values=_parse_csv(args.z, _parse_complex),
        mode=args.mode,
        structure_precision=args.structure_precision,
        p_max=args.p_max,
        quadrature_order=args.quadrature_order,
        central_charge_shift=args.central_charge_shift,
        finite_part_radius=args.finite_part_radius,
        finite_part_check_radius=args.finite_part_check_radius,
        finite_part_samples=args.finite_part_samples,
    )

    print("order\tz\tleft\tright\tabs_residual\trel_residual")
    for result in results:
        print(
            f"{result.order}\t{_format_complex(result.z)}\t"
            f"{_format_complex(result.left)}\t{_format_complex(result.right)}\t"
            f"{result.absolute_residual:.6e}\t"
            f"{result.relative_residual:.6e}"
        )
    for order in sorted({result.order for result in results}):
        maximum = max(
            result.relative_residual
            for result in results
            if result.order == order
        )
        print(f"max_relative_residual(order={order})={maximum:.6e}")


if __name__ == "__main__":
    main()
