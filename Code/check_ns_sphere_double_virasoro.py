#!/usr/bin/env python3
"""Low-order NS sphere check of the free-fermion/double-Virasoro identity.

The ordered block is

    <phi_4(infinity) phi_3(1) phi_2(z) phi_1(0)>.

All external auxiliary-fermion states are the NS vacuum.  The super-Virasoro
weights use the human-note convention

    h(P) = Q**2/8 - P**2/2,       Q = b + 1/b.

The human note normalizes the odd three-form by

    rho_1(phi_4, phi_3, G_-1/2 phi) = -1,
    rho_1(G_-1/2 phi, phi_2, phi_1) = +1.

Consequently the double-Virasoro term with branch label k=2n carries the
endpoint factor (-1)**k.  ``superconformal_blocks.py`` uses the production
BRY sphere convention in which the first of these two entries is +1.  We
therefore negate its odd coefficients before making the human-note check.

Only ordinary Virasoro levels zero, one, and two are needed: this checks the
even NS block through level two and the odd NS block through level 3/2.  At
even level two the k=+/-2 branching primaries enter for the first time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import mpmath
import numpy as np

from ns_genus12_finite_c_check import (
    NSDescendantThreeForm,
    NumericNSVermaModule,
)
from two_virasoro_fusion import blow_up_factor, branch_norm


def ns_weight(momentum, b):
    q = b + 1 / b
    return q * q / 8 - momentum * momentum / 2


def double_virasoro_parameters(momentum, label: int, b):
    """Return ``((c1,h1),(c2,h2))`` in the human-note momentum branch."""

    d1_squared = 2 - 2 * b * b
    b1_squared = 4 * b * b / d1_squared
    q1_squared = b1_squared + 2 + 1 / b1_squared
    h1 = q1_squared / 4 - (momentum + label * b) ** 2 / d1_squared

    d2_squared = 2 - 2 / (b * b)
    inverse_b2_squared = 4 / (b * b * d2_squared)
    q2_squared = inverse_b2_squared + 2 + 1 / inverse_b2_squared
    h2 = q2_squared / 4 - (momentum + label / b) ** 2 / d2_squared
    return (
        (1 + 6 * q1_squared, h1),
        (1 + 6 * q2_squared, h2),
    )


def virasoro_reduced_coefficients(
    *, c, h1, h2, h3, h4, internal_weight, order: int
):
    """Reduced ordinary sphere block coefficients through level two."""

    if order < 0 or order > 2:
        raise ValueError("this low-order oracle supports Virasoro levels 0..2")
    h = internal_weight
    left1 = h + h3 - h4
    right1 = h + h2 - h1
    result = [mpmath.mpc(1)]
    if order == 0:
        return tuple(result)

    result.append(left1 * right1 / (2 * h))
    if order == 1:
        return tuple(result)

    gram = mpmath.matrix(
        [
            [4 * h + c / 2, 6 * h],
            [6 * h, 4 * h * (2 * h + 1)],
        ]
    )
    left = mpmath.matrix(
        [h + 2 * h3 - h4, left1 * (left1 + 1)]
    )
    right = mpmath.matrix(
        [h + 2 * h2 - h1, right1 * (right1 + 1)]
    )
    result.append((left.T * (gram**-1) * right)[0])
    return tuple(result)


def convolve(left: Iterable[complex], right: Iterable[complex], order: int):
    left = tuple(left)
    right = tuple(right)
    return tuple(
        sum(
            left[index] * right[level - index]
            for index in range(level + 1)
            if index < len(left) and level - index < len(right)
        )
        for level in range(order + 1)
    )


def branch_endpoint_coefficient(
    *, b, p1, p2, p3, p4, internal_momentum, label: int, human_sign: bool
):
    """Return the product of the two branching vertices over the branch norm."""

    q = b + 1 / b
    left = blow_up_factor(
        q / 2 + p3,
        0,
        p4,
        0,
        internal_momentum,
        label,
        b,
        precision=80,
    )
    right = blow_up_factor(
        q / 2 + p2,
        0,
        internal_momentum,
        label,
        p1,
        0,
        b,
        precision=80,
    )
    norm = branch_norm(internal_momentum, label, b, precision=80)
    endpoint_sign = -1 if human_sign and label % 2 else 1
    return endpoint_sign * left * right / norm


def double_virasoro_coefficients(
    *, b, momenta, internal_momentum, max_twice_level: int, human_sign: bool
):
    """Return reduced coefficients indexed by twice the NS level."""

    p1, p2, p3, p4 = momenta
    external_parameters = [
        double_virasoro_parameters(momentum, 0, b) for momentum in momenta
    ]
    output = {level: mpmath.mpc(0) for level in range(max_twice_level + 1)}
    maximum_label = int(mpmath.sqrt(max_twice_level / 2)) + 1
    for label in range(-maximum_label, maximum_label + 1):
        base_twice_level = label * label
        if base_twice_level > max_twice_level:
            continue
        remaining_level = (max_twice_level - base_twice_level) // 2
        branch_parameters = double_virasoro_parameters(
            internal_momentum, label, b
        )
        factor_series = []
        for factor in (0, 1):
            c_factor, h_factor = branch_parameters[factor]
            factor_series.append(
                virasoro_reduced_coefficients(
                    c=c_factor,
                    h1=external_parameters[0][factor][1],
                    h2=external_parameters[1][factor][1],
                    h3=external_parameters[2][factor][1],
                    h4=external_parameters[3][factor][1],
                    internal_weight=h_factor,
                    order=remaining_level,
                )
            )
        descendants = convolve(*factor_series, order=remaining_level)
        branch = branch_endpoint_coefficient(
            b=b,
            p1=p1,
            p2=p2,
            p3=p3,
            p4=p4,
            internal_momentum=internal_momentum,
            label=label,
            human_sign=human_sign,
        )
        for level, value in enumerate(descendants):
            output[base_twice_level + 2 * level] += branch * value
    return output


@dataclass(frozen=True)
class CheckRow:
    twice_level: int
    direct_human: complex
    double_human: complex
    absolute_error: float
    double_without_endpoint_sign: complex


def run_check() -> tuple[CheckRow, ...]:
    with mpmath.workdps(80):
        b = mpmath.mpf("1.37")
        momenta = tuple(
            map(
                mpmath.mpf,
                ("0.23", "-0.31", "0.41", "-0.19"),
            )
        )
        internal_momentum = mpmath.mpf("0.47")
        q = b + 1 / b
        weights = tuple(ns_weight(momentum, b) for momentum in momenta)
        internal_weight = ns_weight(internal_momentum, b)
        central_charge = mpmath.mpf("1.5") + 3 * q * q
        module = NumericNSVermaModule(
            c=complex(central_charge), weight=complex(internal_weight)
        )
        left_form = NSDescendantThreeForm(
            c=complex(central_charge),
            bra_weight=complex(weights[3]),
            middle_weight=complex(weights[2]),
            ket_weight=complex(internal_weight),
        )
        right_form = NSDescendantThreeForm(
            c=complex(central_charge),
            bra_weight=complex(internal_weight),
            middle_weight=complex(weights[1]),
            ket_weight=complex(weights[0]),
        )
        corrected = double_virasoro_coefficients(
            b=b,
            momenta=momenta,
            internal_momentum=internal_momentum,
            max_twice_level=4,
            human_sign=True,
        )
        uncorrected = double_virasoro_coefficients(
            b=b,
            momenta=momenta,
            internal_momentum=internal_momentum,
            max_twice_level=4,
            human_sign=False,
        )
        rows = []
        for twice_level in range(5):
            basis = module.basis(twice_level)
            inverse_gram = module.numeric_inverse_gram(twice_level)
            left_vector = np.asarray(
                [left_form.value((), (), state) for state in basis],
                dtype=np.complex128,
            )
            right_vector = np.asarray(
                [right_form.value(state, (), ()) for state in basis],
                dtype=np.complex128,
            )
            direct = mpmath.mpc(
                np.einsum("a,ab,b->", left_vector, inverse_gram, right_vector)
            )
            rows.append(
                CheckRow(
                    twice_level=twice_level,
                    direct_human=complex(direct),
                    double_human=complex(corrected[twice_level]),
                    absolute_error=float(abs(direct - corrected[twice_level])),
                    double_without_endpoint_sign=complex(
                        uncorrected[twice_level]
                    ),
                )
            )
        return tuple(rows)


def main() -> None:
    rows = run_check()
    print("NS sphere direct PBW/c-recursion versus double Virasoro")
    print("human-note odd endpoint sign included")
    print("2L | direct human | double Virasoro | abs error | without sign")
    for row in rows:
        print(
            f"{row.twice_level:2d} | "
            f"{row.direct_human.real:+.15e} | "
            f"{row.double_human.real:+.15e} | "
            f"{row.absolute_error:.3e} | "
            f"{row.double_without_endpoint_sign.real:+.15e}"
        )
    maximum = max(row.absolute_error for row in rows)
    if not np.isfinite(maximum) or maximum > 5.0e-11:
        raise AssertionError(f"low-order double-Virasoro mismatch: {maximum}")
    print(f"maximum absolute error: {maximum:.3e}")


if __name__ == "__main__":
    main()
