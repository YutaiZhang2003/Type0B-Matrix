"""All-NS two-Virasoro branching (fusion) coefficients.

This module implements the free-field prescription used in
Machine Notes/c-Recursion/two_virasoro_branching_coefficients.tex.
The formulas are the factorized blow-up products of arXiv:1111.2803,
translated to the ordered trinion convention of the human note.

The public trinion ordering is

    leg 1 = ket,  leg 2 = inserted field,  leg 3 = bra.

For integer branch labels k_i = 2 n_i, the unnormalized numerator is

    l(Q/2 + P2, k2 | P3, k3, P1, k1),

and the canonical coefficient entering the conformal-block decomposition is

    B_a^2 = l^2 / (N_{k3}(P3) N_{k2}(P2) N_{k1}(P1)).

The unsquared coefficient depends on choices of square roots of the three
Shapovalov norms.  ns_fusion_coefficient uses principal square roots and
documents that convention explicitly.
"""

from __future__ import annotations

import argparse
import operator
from dataclasses import dataclass
from typing import Union

import mpmath


Scalar = Union[int, float, complex, str, mpmath.mpf, mpmath.mpc]


def _integer_label(value: int, name: str) -> int:
    """Return an exact integer branch label, rejecting floats and booleans."""

    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise TypeError(
            f"{name} must be an integer branch label k=2n; got {value!r}"
        ) from exc


def _working_precision(precision: int) -> int:
    precision = _integer_label(precision, "precision")
    if precision < 2:
        raise ValueError("precision must be at least 2 decimal digits")
    return precision


def _mp(value: Scalar) -> mpmath.mpc:
    if isinstance(value, str):
        text = "".join(value.split()).lower().replace("i", "j")
        if "j" not in text:
            return mpmath.mpc(mpmath.mpf(text))
        if text.count("j") != 1 or not text.endswith("j"):
            raise ValueError(
                f"invalid complex number {value!r}; use a+bj or a+bi"
            )

        body = text[:-1]
        split_at = None
        for index in range(1, len(body)):
            if body[index] in "+-" and body[index - 1] not in "eE":
                split_at = index
        if split_at is None:
            real_text, imaginary_text = "0", body
        else:
            real_text = body[:split_at]
            imaginary_text = body[split_at:]
        if imaginary_text in ("", "+", "-"):
            imaginary_text += "1"
        return mpmath.mpc(
            mpmath.mpf(real_text),
            mpmath.mpf(imaginary_text),
        )
    return mpmath.mpc(value)


def _nonzero_b(b: Scalar) -> mpmath.mpc:
    b_mp = _mp(b)
    if b_mp == 0:
        raise ZeroDivisionError("the free-field parameter b must be nonzero")
    return b_mp


def _sign_from_parity(integer: int) -> int:
    return -1 if integer % 2 else 1


def _s_even_mp(
    x: mpmath.mpc,
    r: int,
    b: mpmath.mpc,
    q: mpmath.mpc,
) -> mpmath.mpc:
    """Internal even product with all arguments already normalized."""

    if r < 0:
        return _sign_from_parity(r) * _s_even_mp(q - x, -r, b, q)

    value = mpmath.power(2, -mpmath.mpf(r * r) / 2)
    for i in range(1, 2 * r):
        for j in range(1, 2 * r - i + 1):
            if (i + j) % 2 == 0:
                value *= x + (i - 1) * b + (j - 1) / b
    return value


def _s_odd_mp(
    x: mpmath.mpc,
    r: int,
    b: mpmath.mpc,
    q: mpmath.mpc,
) -> mpmath.mpc:
    """Internal odd product with all arguments already normalized."""

    if r < 0:
        return _s_odd_mp(q - x, -r, b, q)

    value = mpmath.power(2, -mpmath.mpf(r * (r + 1)) / 2)
    for i in range(1, 2 * r + 1):
        for j in range(1, 2 * r - i + 2):
            if (i + j) % 2 == 1:
                value *= x + (i - 1) * b + (j - 1) / b
    return value


def s_even(
    x: Scalar,
    r: int,
    b: Scalar,
    *,
    precision: int = 50,
) -> mpmath.mpc:
    r"""Evaluate the factor \(s_{\mathrm{even}}(x,r)\).

    For nonnegative integer r,

    .. math::

       s_{\rm even}(x,r)=2^{-r^2/2}
       \prod_{\substack{i,j\geq1,\ i+j\leq2r\\i+j\ {\rm even}}}
       [x+(i-1)b+(j-1)b^{-1}].

    Negative r is continued by
    s_even(x,r)=(-1)^r s_even(Q-x,-r).
    """

    precision = _working_precision(precision)
    r = _integer_label(r, "r")
    with mpmath.workdps(precision):
        b_mp = _nonzero_b(b)
        return +_s_even_mp(_mp(x), r, b_mp, b_mp + 1 / b_mp)


def s_odd(
    x: Scalar,
    r: int,
    b: Scalar,
    *,
    precision: int = 50,
) -> mpmath.mpc:
    r"""Evaluate the factor \(s_{\mathrm{odd}}(x,r)\).

    For nonnegative integer r,

    .. math::

       s_{\rm odd}(x,r)=2^{-r(r+1)/2}
       \prod_{\substack{i,j\geq1,\ i+j\leq2r+1\\i+j\ {\rm odd}}}
       [x+(i-1)b+(j-1)b^{-1}].

    Negative r is continued by s_odd(x,r)=s_odd(Q-x,-r).
    """

    precision = _working_precision(precision)
    r = _integer_label(r, "r")
    with mpmath.workdps(precision):
        b_mp = _nonzero_b(b)
        return +_s_odd_mp(_mp(x), r, b_mp, b_mp + 1 / b_mp)


def _integer_part_of_half_integer(numerator: int) -> int:
    """Return Int(numerator/2), where Int truncates toward zero."""

    magnitude = abs(numerator) // 2
    return -magnitude if numerator < 0 else magnitude


def _blow_up_factor_mp(
    alpha: mpmath.mpc,
    m: int,
    p_prime: mpmath.mpc,
    k_prime: int,
    p: mpmath.mpc,
    k: int,
    b: mpmath.mpc,
) -> mpmath.mpc:
    q = b + 1 / b
    even_channel = (m + k + k_prime) % 2 == 0
    value = mpmath.mpc(1)

    for sigma in (-1, 1):
        for tau in (-1, 1):
            x_sigma_tau = alpha + sigma * p_prime + tau * p
            twice_r = m + sigma * k_prime + tau * k
            if even_channel:
                if twice_r % 2:
                    raise ArithmeticError(
                        "internal parity error in the even blow-up product"
                    )
                value *= _s_even_mp(
                    x_sigma_tau,
                    twice_r // 2,
                    b,
                    q,
                )
            else:
                if twice_r % 2 == 0:
                    raise ArithmeticError(
                        "internal parity error in the odd blow-up product"
                    )
                value *= _s_odd_mp(
                    x_sigma_tau,
                    _integer_part_of_half_integer(twice_r),
                    b,
                    q,
                )
    return value


def blow_up_factor(
    alpha: Scalar,
    m: int,
    p_prime: Scalar,
    k_prime: int,
    p: Scalar,
    k: int,
    b: Scalar,
    *,
    precision: int = 50,
) -> mpmath.mpc:
    r"""Evaluate \(l(\alpha,m\mid P',k',P,k)\).

    m, k_prime, and k are the integer Fermi-sea labels of the inserted, bra,
    and ket branching vectors.  The even/odd product is selected by
    (m + k + k_prime) mod 2.  In the odd case the half-integer indices are
    mapped with Int(x)=sgn(x) floor(abs(x)).
    """

    precision = _working_precision(precision)
    m = _integer_label(m, "m")
    k_prime = _integer_label(k_prime, "k_prime")
    k = _integer_label(k, "k")
    with mpmath.workdps(precision):
        b_mp = _nonzero_b(b)
        return +_blow_up_factor_mp(
            _mp(alpha),
            m,
            _mp(p_prime),
            k_prime,
            _mp(p),
            k,
            b_mp,
        )


def _branch_norm_mp(
    p: mpmath.mpc,
    k: int,
    b: mpmath.mpc,
) -> mpmath.mpc:
    """Identity specialization of the blow-up product."""

    q = b + 1 / b
    return _s_even_mp(2 * p, k, b, q) * _s_even_mp(-2 * p, -k, b, q)


def branch_norm(
    p: Scalar,
    k: int,
    b: Scalar,
    *,
    precision: int = 50,
) -> mpmath.mpc:
    r"""Return the Shapovalov norm \(N_k(P)=\langle P,k\mid P,k\rangle\).

    It is computed from the same prescription by the identity specialization

    .. math::

       N_k(P)=s_{\rm even}(2P,k)s_{\rm even}(-2P,-k).
    """

    precision = _working_precision(precision)
    k = _integer_label(k, "k")
    with mpmath.workdps(precision):
        b_mp = _nonzero_b(b)
        return +_branch_norm_mp(_mp(p), k, b_mp)


@dataclass(frozen=True)
class NSFusionData:
    """The numerator, norms, parity, and normalized all-NS coefficient."""

    parity: int
    numerator: mpmath.mpc
    ket_norm: mpmath.mpc
    inserted_norm: mpmath.mpc
    bra_norm: mpmath.mpc
    coefficient_squared: mpmath.mpc
    principal_coefficient: mpmath.mpc


def ns_fusion_data(
    *,
    b: Scalar,
    p1: Scalar,
    p2: Scalar,
    p3: Scalar,
    k1: int,
    k2: int,
    k3: int,
    precision: int = 50,
) -> NSFusionData:
    r"""Compute the all-NS branching coefficient in human-note ordering.

    The ordered legs are (1,2,3) = (ket, inserted, bra) and k_i=2n_i must be
    integers.  The returned parity is a=(k1+k2+k3) mod 2.

    coefficient_squared is the canonical quantity \(B_a^2\) entering the
    conformal-block decomposition.  principal_coefficient uses the principal
    square root of each norm separately; changing any of those three roots
    changes the sign or phase of the unsquared coefficient.
    """

    precision = _working_precision(precision)
    k1 = _integer_label(k1, "k1")
    k2 = _integer_label(k2, "k2")
    k3 = _integer_label(k3, "k3")

    with mpmath.workdps(precision):
        b_mp = _nonzero_b(b)
        q = b_mp + 1 / b_mp
        p1_mp, p2_mp, p3_mp = _mp(p1), _mp(p2), _mp(p3)
        numerator = _blow_up_factor_mp(
            q / 2 + p2_mp,
            k2,
            p3_mp,
            k3,
            p1_mp,
            k1,
            b_mp,
        )
        ket_norm = _branch_norm_mp(p1_mp, k1, b_mp)
        inserted_norm = _branch_norm_mp(p2_mp, k2, b_mp)
        bra_norm = _branch_norm_mp(p3_mp, k3, b_mp)
        denominator = bra_norm * inserted_norm * ket_norm
        if denominator == 0:
            raise ZeroDivisionError(
                "the normalized fusion coefficient is singular because "
                "at least one branching-vector norm vanishes"
            )

        coefficient_squared = numerator**2 / denominator
        principal_coefficient = numerator / (
            mpmath.sqrt(bra_norm)
            * mpmath.sqrt(inserted_norm)
            * mpmath.sqrt(ket_norm)
        )
        return NSFusionData(
            parity=(k1 + k2 + k3) % 2,
            numerator=+numerator,
            ket_norm=+ket_norm,
            inserted_norm=+inserted_norm,
            bra_norm=+bra_norm,
            coefficient_squared=+coefficient_squared,
            principal_coefficient=+principal_coefficient,
        )


def ns_fusion_coefficient_squared(
    *,
    b: Scalar,
    p1: Scalar,
    p2: Scalar,
    p3: Scalar,
    k1: int,
    k2: int,
    k3: int,
    precision: int = 50,
) -> mpmath.mpc:
    """Return the canonical squared coefficient B_a^2."""

    return ns_fusion_data(
        b=b,
        p1=p1,
        p2=p2,
        p3=p3,
        k1=k1,
        k2=k2,
        k3=k3,
        precision=precision,
    ).coefficient_squared


def ns_fusion_coefficient(
    *,
    b: Scalar,
    p1: Scalar,
    p2: Scalar,
    p3: Scalar,
    k1: int,
    k2: int,
    k3: int,
    precision: int = 50,
) -> mpmath.mpc:
    """Return B_a using principal square roots of all three norms."""

    return ns_fusion_data(
        b=b,
        p1=p1,
        p2=p2,
        p3=p3,
        k1=k1,
        k2=k2,
        k3=k3,
        precision=precision,
    ).principal_coefficient


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the all-NS two-Virasoro branching coefficient in "
            "(ket, inserted, bra) ordering."
        )
    )
    parser.add_argument("--b", required=True, help="free-field parameter b")
    parser.add_argument("--p1", required=True, help="ket momentum P1")
    parser.add_argument("--p2", required=True, help="inserted momentum P2")
    parser.add_argument("--p3", required=True, help="bra momentum P3")
    parser.add_argument("--k1", required=True, type=int, help="ket label k1=2n1")
    parser.add_argument(
        "--k2",
        required=True,
        type=int,
        help="inserted label k2=2n2",
    )
    parser.add_argument("--k3", required=True, type=int, help="bra label k3=2n3")
    parser.add_argument("--precision", type=int, default=50)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    data = ns_fusion_data(
        b=args.b,
        p1=args.p1,
        p2=args.p2,
        p3=args.p3,
        k1=args.k1,
        k2=args.k2,
        k3=args.k3,
        precision=args.precision,
    )
    digits = args.precision
    print(f"parity a = {data.parity}")
    print(f"numerator = {mpmath.nstr(data.numerator, digits)}")
    print(f"N_k1(P1) = {mpmath.nstr(data.ket_norm, digits)}")
    print(f"N_k2(P2) = {mpmath.nstr(data.inserted_norm, digits)}")
    print(f"N_k3(P3) = {mpmath.nstr(data.bra_norm, digits)}")
    print(f"B_a^2 = {mpmath.nstr(data.coefficient_squared, digits)}")
    print(
        "B_a (principal norm roots) = "
        f"{mpmath.nstr(data.principal_coefficient, digits)}"
    )


__all__ = [
    "NSFusionData",
    "blow_up_factor",
    "branch_norm",
    "ns_fusion_coefficient",
    "ns_fusion_coefficient_squared",
    "ns_fusion_data",
    "s_even",
    "s_odd",
]


if __name__ == "__main__":
    main()
