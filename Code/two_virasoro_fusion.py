"""All-NS two-Virasoro branching (fusion) coefficients.

The public trinion order is exactly the order in the human note:

    slot 1 = BPZ-conjugate state at infinity,
    slot 2 = inserted state at one,
    slot 3 = ket state at zero.

ns_fusion_data computes the human-note coefficient directly for all triples
with k_i = 2 n_i in {-1, 0, 1}.  It uses branching vectors obtained from the
human Virasoro highest-weight equations and evaluates their norms and
trilinear forms with the ungraded auxiliary-fermion x SCA pairing and the
fixed-parity conventions, including arbitrary intrinsic primary parity.
Higher labels are rejected until their branching vectors have an equally
direct implementation.

blow_up_factor and the general-label part of branch_norm implement the
factorized ratio from arXiv:1111.2803 for comparison.  That ratio is not
identified with the human-note three-point numerator.
"""

from __future__ import annotations

import argparse
import operator
from dataclasses import dataclass
from typing import Sequence, Union

import mpmath

from ns_human_convention import (
    normalize_parity_triple,
)


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
    r"""Return the paper-product candidate for \(N_k(P)\).

    It is obtained by the identity specialization of the literature ratio,

    .. math::

       N_k(P)=s_{\rm even}(2P,k)s_{\rm even}(-2P,-k).

    For k in {-1,0,1}, this agrees with the direct human Gram calculation.
    General k is retained for comparison and is not used by ns_fusion_data.
    """

    precision = _working_precision(precision)
    k = _integer_label(k, "k")
    with mpmath.workdps(precision):
        b_mp = _nonzero_b(b)
        return +_branch_norm_mp(_mp(p), k, b_mp)


@dataclass(frozen=True)
class NSFusionData:
    """Human-note numerator, norms, parity, and normalized coefficient."""

    parity: int
    numerator: mpmath.mpc
    slot1_norm: mpmath.mpc
    slot2_norm: mpmath.mpc
    slot3_norm: mpmath.mpc
    coefficient_squared: mpmath.mpc
    principal_coefficient: mpmath.mpc


def _human_direct_norm_mp(
    p: mpmath.mpc,
    k: int,
    q: mpmath.mpc,
) -> mpmath.mpc:
    """Direct human Shapovalov norm for k in {-1, 0, 1}."""

    if k == 0:
        return mpmath.mpc(1)
    h = (q**2 / 4 - p**2) / 2
    gamma = q / 2 + k * p
    return 2 * h - gamma**2


def _human_direct_numerator_mp(
    *,
    q: mpmath.mpc,
    p1: mpmath.mpc,
    p2: mpmath.mpc,
    p3: mpmath.mpc,
    k1: int,
    k2: int,
    k3: int,
    primary_parities: tuple[int, int, int] = (0, 0, 0),
) -> mpmath.mpc:
    r"""Return \(\widehat\rho_a(v_{1,n_1},v_{2,n_2},v_{3,n_3})\).

    This is the direct expansion with the ungraded tensor-product pairing for
    the implemented level-one-half cases.  Because the three-point form is a
    matrix element, its crossing is ``(-1)^((p_2+|x_2|)|u_3|)`` in the same
    pairing convention.  In particular a lone odd state in slot 3 has
    numerator -1 for even primaries, consistently with its BPZ matrix element.
    """

    momenta = (p1, p2, p3)
    labels = (k1, k2, k3)
    weights = tuple((q**2 / 4 - p**2) / 2 for p in momenta)
    gammas = tuple(
        q / 2 + k * p if k else mpmath.mpc(0)
        for p, k in zip(momenta, labels)
    )
    primary1, primary2, _primary3 = primary_parities
    parity_phase = _sign_from_parity(
        primary1 * (k1 % 2) + primary2 * (k3 % 2)
    )
    if primary1:
        gammas = (-gammas[0], gammas[1], gammas[2])
    active = sum(k != 0 for k in labels)

    if active == 0:
        return mpmath.mpc(parity_phase)
    if active == 1:
        return mpmath.mpc(parity_phase * (-1 if k3 else 1))

    h1, h2, h3 = weights
    gamma1, gamma2, gamma3 = gammas
    if active == 2:
        if k3 == 0:
            return parity_phase * (h1 + h2 - h3 - gamma1 * gamma2)
        if k2 == 0:
            return parity_phase * (h1 - h2 + h3 - gamma1 * gamma3)
        return parity_phase * (h1 - h2 - h3 + gamma2 * gamma3)

    return parity_phase * (
        -(h1 + h2 + h3 - mpmath.mpf("0.5"))
        + gamma1 * gamma2
        + gamma1 * gamma3
        + gamma2 * gamma3
    )


def ns_fusion_data(
    *,
    b: Scalar,
    p1: Scalar,
    p2: Scalar,
    p3: Scalar,
    k1: int,
    k2: int,
    k3: int,
    primary_parities: Sequence[int] = (0, 0, 0),
    precision: int = 50,
) -> NSFusionData:
    r"""Compute the directly verified human-note branching coefficient.

    The slots are (1,2,3) = (BPZ/infinity, insertion/one, ket/zero).
    The supported labels are all in {-1,0,1}.  The returned parity is
    a=(k1+k2+k3) mod 2.  Labels with absolute value above one are rejected:
    their branching vectors are not implemented in this numerical routine.

    For intrinsic primary parities ``(p_1,p_2,p_3)``, the direct component
    expansion is used with the matrix-element crossing inherited from the
    ungraded tensor BPZ convention.  At this first branching level it equals
    the even-primary expression with ``gamma_1 -> (-1)^p_1 gamma_1`` and the
    overall factor ``(-1)^(p_1 k_1+p_2 k_3)``.  Therefore the squared
    coefficient is generally *not* parity independent when ``p_1=1``.

    ``principal_coefficient`` uses the principal square root of each norm
    separately; changing any of those three roots changes the sign or phase
    of the unsquared coefficient.
    """

    precision = _working_precision(precision)
    k1 = _integer_label(k1, "k1")
    k2 = _integer_label(k2, "k2")
    k3 = _integer_label(k3, "k3")
    labels = (k1, k2, k3)
    primaries = normalize_parity_triple(
        primary_parities, name="primary_parities"
    )
    if not all(abs(k) <= 1 for k in labels):
        raise NotImplementedError(
            "the human-convention calculation currently supports all "
            "k_i in {-1,0,1}; higher branching vectors are not implemented"
        )

    with mpmath.workdps(precision):
        b_mp = _nonzero_b(b)
        q = b_mp + 1 / b_mp
        p1_mp, p2_mp, p3_mp = _mp(p1), _mp(p2), _mp(p3)
        numerator = _human_direct_numerator_mp(
            q=q,
            p1=p1_mp,
            p2=p2_mp,
            p3=p3_mp,
            k1=k1,
            k2=k2,
            k3=k3,
            primary_parities=primaries,
        )
        slot1_norm = _human_direct_norm_mp(p1_mp, k1, q)
        slot2_norm = _human_direct_norm_mp(p2_mp, k2, q)
        slot3_norm = _human_direct_norm_mp(p3_mp, k3, q)
        denominator = slot1_norm * slot2_norm * slot3_norm
        if denominator == 0:
            raise ZeroDivisionError(
                "the normalized fusion coefficient is singular because "
                "at least one branching-vector norm vanishes"
            )

        coefficient_squared = numerator**2 / denominator
        principal_coefficient = numerator / (
            mpmath.sqrt(slot1_norm)
            * mpmath.sqrt(slot2_norm)
            * mpmath.sqrt(slot3_norm)
        )
        return NSFusionData(
            parity=(k1 + k2 + k3) % 2,
            numerator=+numerator,
            slot1_norm=+slot1_norm,
            slot2_norm=+slot2_norm,
            slot3_norm=+slot3_norm,
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
    primary_parities: Sequence[int] = (0, 0, 0),
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
        primary_parities=primary_parities,
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
    primary_parities: Sequence[int] = (0, 0, 0),
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
        primary_parities=primary_parities,
        precision=precision,
    ).principal_coefficient


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the directly verified all-NS two-Virasoro branching "
            "coefficient in human-note slot order (infinity, one, zero)."
        )
    )
    parser.add_argument("--b", required=True, help="free-field parameter b")
    parser.add_argument(
        "--p1", required=True, help="slot-1 BPZ/infinity momentum P1"
    )
    parser.add_argument("--p2", required=True, help="slot-2 momentum P2 at one")
    parser.add_argument("--p3", required=True, help="slot-3 ket momentum P3")
    parser.add_argument(
        "--k1", required=True, type=int, help="slot-1 label k1=2n1"
    )
    parser.add_argument(
        "--k2",
        required=True,
        type=int,
        help="slot-2 label k2=2n2",
    )
    parser.add_argument(
        "--k3", required=True, type=int, help="slot-3 label k3=2n3"
    )
    parser.add_argument(
        "--primary-parities",
        nargs=3,
        type=int,
        default=(0, 0, 0),
        metavar=("EPS1", "EPS2", "EPS3"),
        help="intrinsic primary parity bits in slots (infinity,one,zero)",
    )
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
        primary_parities=args.primary_parities,
        precision=args.precision,
    )
    digits = args.precision
    print(f"parity a = {data.parity}")
    print(f"human rho-hat numerator = {mpmath.nstr(data.numerator, digits)}")
    print(f"N_k1(P1) = {mpmath.nstr(data.slot1_norm, digits)}")
    print(f"N_k2(P2) = {mpmath.nstr(data.slot2_norm, digits)}")
    print(f"N_k3(P3) = {mpmath.nstr(data.slot3_norm, digits)}")
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
