"""Evaluate the first BRY-convention four-R sphere-block benchmark."""

from __future__ import annotations

import math

from ramond_sphere_blocks import RamondExternalSphereFourPointBlock, b_from_c


def main() -> None:
    c = 13.5 + 1.0e-5
    b = b_from_c(c)
    z = 1.0 / 3.0 + 3.0j / 5.0
    momenta = {
        "p1": 1.0 / 2.0,
        "p2": 1.0 / 3.0,
        "p3": 1.0 / 4.0,
        "p4": 3.0 / 5.0,
        "internal_momentum": math.sqrt(6.0 / 5.0),
    }

    print("BRY/HJS four-R sphere-block benchmark")
    print(f"  c={c:.12g}, b={b.real:.15g}, z={z}")
    print("  beta_i=i P_i/sqrt(2); internal channel=NS; q order=8")
    print()

    for sign3 in (1, -1):
        for sign2 in (1, -1):
            block = RamondExternalSphereFourPointBlock.from_liouville_momenta(
                **momenta,
                b=b,
                sign3=sign3,
                sign2=sign2,
            )
            even = block.elliptic_block(z, 8, "even")
            odd = block.elliptic_block(z, 8, "odd")
            even_product = block.diagonal_block_product(z, 8, "even")
            odd_product = block.diagonal_block_product(z, 8, "odd")
            print(f"  (sign3,sign2)=({sign3:+d},{sign2:+d})")
            print(f"    F_even={even.real:+.15g}{even.imag:+.15g}i")
            print(f"    F_odd ={odd.real:+.15g}{odd.imag:+.15g}i")
            print(
                "    diagonal even+odd="
                f"{(even_product + odd_product).real:.15g}"
            )

    print()
    print(
        "The diagonal products are unweighted block diagnostics.  The full "
        "Type-0B four-R correlator additionally requires the BRY structure "
        "constants and the chiral-to-nonchiral Ramond sewing matrix."
    )


if __name__ == "__main__":
    main()
