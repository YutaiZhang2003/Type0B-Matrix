"""Compare direct Ramond descendant sewing with recursion through q^3."""

from __future__ import annotations

from ramond_descendant_blocks import BruteForceMixedRExchangeSphereBlock
from self_dual_superconformal_blocks import (
    SelfDualMixedRExchangeSphereFourPointBlock,
)


MOMENTA = dict(
    p1_ns=0.31,
    p2_r=0.41,
    p3_r=0.23,
    p4_ns=0.37,
    internal_momentum=0.70,
)


def comparison_ledger():
    rows = []
    for sign3 in (1, -1):
        for sign2 in (1, -1):
            direct = BruteForceMixedRExchangeSphereBlock(
                b=1.0,
                sign3=sign3,
                sign2=sign2,
                **MOMENTA,
            ).elliptic_coefficients(3)
            recursive = SelfDualMixedRExchangeSphereFourPointBlock(
                sign3=sign3,
                sign2=sign2,
                **MOMENTA,
            ).elliptic_coefficients(4)
            for power in range(4):
                rows.append(
                    (
                        sign3,
                        sign2,
                        power,
                        direct[power],
                        recursive[power],
                        abs(direct[power] - recursive[power]),
                    )
                )
    return tuple(rows)


def main() -> None:
    print("Direct long-R descendant sewing versus exact-b=1 recursion")
    print(
        "momenta: "
        "(P1_NS,P2_R,P3_R,P4_NS;P)="
        "(0.31,0.41,0.23,0.37;0.70)"
    )
    print("signs  power       brute force             recursion              abs diff")
    for sign3, sign2, power, direct, recursive, difference in comparison_ledger():
        print(
            f"({sign3:+d},{sign2:+d}) q^{power:<1d}  "
            f"{direct.real:+.15f}  "
            f"{recursive.real:+.15f}  "
            f"{difference:.3e}"
        )


if __name__ == "__main__":
    main()
