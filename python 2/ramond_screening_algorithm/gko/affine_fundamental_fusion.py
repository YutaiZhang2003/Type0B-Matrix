#!/usr/bin/env python3
"""Fundamental affine sl(2) fusion matrix and the Ramond-arrow audit.

The 2 by 2 matrix below is the connection matrix of the KZ system with
one affine spin-1/2 insertion.  Its normalization is the highest-weight
normalization used in Corollary 4.22 of arXiv:2404.14350 and in the
``(2,1)`` appendix of hep-th/9712256.

This file also tests the corrected two-GKO Ramond lattice

    r = n + delta/4,       s = delta/4 - n.

The test is intentionally independent of the Ward evaluator.  It proves
symbolically at (0,3/4,3/4) that the delta=-1 path, after division by its
ground value at (0,1/4,1/4), is identical to the delta=+1 path.  Thus the
two affine-Weyl arrows give a rank-one normalized GKO answer and cannot by
themselves produce the crossed Ramond master.
"""

from __future__ import annotations

from pathlib import Path
import sys

import mpmath as mp
import sympy as sp


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from two_step_probe import (  # noqa: E402
    affine_kappa,
    affine_weight,
    signed_gko_ratio,
)
from audit_gko_relevance import hard_channel_obstruction  # noqa: E402


def fusion_parameters(mu, nu, lam, kappa):
    """Return the hypergeometric parameters (A,B,C).

    The affine four-point block is

      <v_mu | V_nu(1) X_1(z) | v_lam>.

    The two s-channel affine weights are ``lam +/- 1`` and the two
    t-channel weights are ``nu +/- 1``.
    """

    A = (lam - mu + nu + 1) / (2 * kappa)
    B = (lam + mu + nu + 3) / (2 * kappa)
    C = (lam + 1) / kappa
    return A, B, C


def affine_fusion_matrix(mu, nu, lam, kappa):
    """Exact normalized fundamental affine fusion matrix.

    Rows are the s-channel arrows ``lam+1, lam-1`` and columns are the
    t-channel arrows ``nu+1, nu-1``.  On the real interval 0<z<1 this is
    a pure reassociation matrix and carries no extra braid phase.
    """

    A, B, C = fusion_parameters(mu, nu, lam, kappa)
    gamma = sp.gamma
    return sp.Matrix(
        [
            [
                gamma(C) * gamma(C - A - B + 1)
                / (gamma(C - A) * gamma(C - B + 1)),
                gamma(C) * gamma(A + B - C)
                / (gamma(A) * gamma(B)),
            ],
            [
                gamma(1 - C) * gamma(C - A - B + 1)
                / (gamma(1 - A) * gamma(1 - B)),
                -gamma(1 - C) * gamma(A + B - C)
                / (gamma(A - C + 1) * gamma(B - C)),
            ],
        ]
    )


def half_braid_phases(lam, kappa):
    """Principal-branch half-monodromy phases of the two s blocks."""

    return (
        sp.exp(sp.pi * sp.I * lam / (2 * kappa)),
        sp.exp(sp.pi * sp.I * (1 - (lam + 2) / (2 * kappa))),
    )


def csl_character_exponents(x1, x2, x3, kappa, delta2, delta3):
    r"""Appendix-B character defining the absolute affine 3-point factor.

    In BFT Appendix B, equations (B.3)--(B.4), put

      q1=e^(2 tau), q2=e^(-2 kappa tau),
      (lambda,mu,nu)=(-1+delta3*kappa/2+x3,
                       -1+x1,
                       -1+delta2*kappa/2+x2).

    The returned tuple contains the seven exponents and their signs in

      C^sl = E[(sum sign*exp(tau*exponent)) /
               ((1-exp(2*tau))*(1-exp(-2*kappa*tau)))].

    This representation is exact and keeps the regularization convention
    of the paper explicit.
    """

    d2 = sp.Integer(delta2)
    d3 = sp.Integer(delta3)
    K = kappa
    exponents = (
        1 - 2 * K + (d2 + d3) * K / 2 + x1 + x2 + x3,
        2 + (d3 - 2) * K + 2 * x3,
        2 - 2 * K + 2 * x1,
        1 + (d3 - d2) * K / 2 + x3 - x1 - x2,
        1 - (d2 + d3) * K / 2 + x1 - x2 - x3,
        1 + (d2 - d3) * K / 2 - x1 + x2 - x3,
        2 - d3 * K - 2 * x3,
    )
    signs = (1, -1, -1, 1, 1, 1, -1)
    return tuple(zip(signs, map(sp.factor, exponents)))


def corrected_path_labels(branch, sector, delta=1):
    branch = sp.Rational(branch)
    if sector == "NS":
        return branch, -branch
    delta = sp.Integer(delta)
    return branch + delta / 4, delta / 4 - branch


def check_physical_momentum_dictionary():
    """Prove that the corrected labels give both target Virasoro momenta."""

    kappa, x, n, delta = sp.symbols("kappa x n delta", nonzero=True)
    lam_plus_one = delta * kappa / 2 + x
    r = n + delta / 4
    s = delta / 4 - n
    first_coordinate = sp.cancel(r - lam_plus_one / (2 * kappa))
    first_expected = sp.cancel(n - x / (2 * kappa))
    second_coordinate = sp.cancel(
        s - (lam_plus_one + 2 * r) / (2 * (kappa + 1))
    )
    second_expected = sp.cancel(
        -(kappa + 2) * n / (kappa + 1) - x / (2 * (kappa + 1))
    )
    if sp.cancel(first_coordinate - first_expected) != 0:
        raise AssertionError("first Virasoro momentum mismatch")
    if sp.cancel(second_coordinate - second_expected) != 0:
        raise AssertionError("second Virasoro momentum mismatch")


def check_arrow_selection_rule():
    """Verify the equal/mixed arrow rule on the complete stored level grid."""

    ns_levels = (sp.Integer(0), sp.Rational(1, 2), sp.Integer(1))
    r_levels = (sp.Rational(1, 4), sp.Rational(3, 4), sp.Rational(5, 4))
    for n1 in ns_levels:
        for n2 in r_levels:
            for n3 in r_levels:
                m2 = int(2 * n2 - sp.Rational(1, 2))
                m3 = int(2 * n3 - sp.Rational(1, 2))
                equal = (int(2 * n1) + m2 + m3) % 2 == 0
                allowed = []
                for delta2 in (1, -1):
                    for delta3 in (1, -1):
                        r1, s1 = corrected_path_labels(n1, "NS")
                        r2, s2 = corrected_path_labels(n2, "R", delta2)
                        r3, s3 = corrected_path_labels(n3, "R", delta3)
                        if (r1 + r2 + r3).q == 1 and (s1 + s2 + s3).q == 1:
                            allowed.append((delta2, delta3))
                expected = (
                    {(1, 1), (-1, -1)}
                    if equal
                    else {(1, -1), (-1, 1)}
                )
                if set(allowed) != expected:
                    raise AssertionError(((n1, n2, n3), allowed, expected))


def corrected_two_step_amplitude(labels, sample, delta2=1, delta3=1):
    """Two signed GKO ratios on the corrected Ramond arrow lattice."""

    n1, n2, n3 = map(sp.Rational, labels)
    b, p1, p2, p3 = sample
    kappa = affine_kappa(b)
    weights = (
        affine_weight(b, p1, "NS"),
        affine_weight(b, p2, "R", delta2),
        affine_weight(b, p3, "R", delta3),
    )
    pairs = (
        corrected_path_labels(n1, "NS"),
        corrected_path_labels(n2, "R", delta2),
        corrected_path_labels(n3, "R", delta3),
    )
    r = tuple(pair[0] for pair in pairs)
    s = tuple(pair[1] for pair in pairs)
    if (sum(r).q != 1) or (sum(s).q != 1):
        return None
    first = signed_gko_ratio(*r, *weights, kappa)
    shifted = tuple(weight + 2 * ri for weight, ri in zip(weights, r))
    second = signed_gko_ratio(*s, *shifted, kappa + 1)
    return sp.factor(sp.cancel(first * second))


def check_connection_matrix():
    """Numerically check the matrix against the two Frobenius bases."""

    mp.mp.dps = 60
    A = mp.mpf("0.34259259259259259259259259259259259")
    B = mp.mpf("0.79814814814814814814814814814814815")
    C = mp.mpf("0.52222222222222222222222222222222222")
    z = mp.mpf("0.27")
    # These values obey A=rho/kappa and A+B-C=(nu+1)/kappa.
    kappa = mp.mpf("2.7")
    rho = kappa * A
    nu = kappa * (A + B - C) - 1
    n_plus = (nu - rho + 1) / (nu + 1)
    d_s = (B - C) / (C - 1)
    hyp = mp.hyp2f1
    s_plus = hyp(A, B, C, z)
    s_minus = d_s * z ** (1 - C) * hyp(
        A - C + 1, B - C + 1, 2 - C, z
    )
    t_plus = n_plus * hyp(A, B, A + B - C + 1, 1 - z)
    t_minus = (1 - z) ** (C - A - B) * hyp(
        C - A, C - B, C - A - B + 1, 1 - z
    )
    gamma = mp.gamma
    matrix = (
        (
            gamma(C) * gamma(C - A - B + 1)
            / (gamma(C - A) * gamma(C - B + 1)),
            gamma(C) * gamma(A + B - C) / (gamma(A) * gamma(B)),
        ),
        (
            gamma(1 - C) * gamma(C - A - B + 1)
            / (gamma(1 - A) * gamma(1 - B)),
            -gamma(1 - C) * gamma(A + B - C)
            / (gamma(A - C + 1) * gamma(B - C)),
        ),
    )
    residuals = (
        s_plus - matrix[0][0] * t_plus - matrix[0][1] * t_minus,
        s_minus - matrix[1][0] * t_plus - matrix[1][1] * t_minus,
    )
    if max(map(abs, residuals)) > mp.mpf("1e-50"):
        raise AssertionError(residuals)
    determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    if abs(determinant + 1) > mp.mpf("1e-50"):
        raise AssertionError(determinant)


def exact_hard_rank_one_audit():
    """Prove the normalized delta=+/- hard paths are identical."""

    b, p1, p2, p3 = sp.symbols("b P1 P2 P3", nonzero=True)
    sample = (b, p1, p2, p3)
    ground_plus = corrected_two_step_amplitude(
        (0, sp.Rational(1, 4), sp.Rational(1, 4)), sample, 1, 1
    )
    ground_minus = corrected_two_step_amplitude(
        (0, sp.Rational(1, 4), sp.Rational(1, 4)), sample, -1, -1
    )
    hard_plus = corrected_two_step_amplitude(
        (0, sp.Rational(3, 4), sp.Rational(3, 4)), sample, 1, 1
    )
    hard_minus = corrected_two_step_amplitude(
        (0, sp.Rational(3, 4), sp.Rational(3, 4)), sample, -1, -1
    )
    assert ground_plus == 1
    residual = sp.factor(sp.cancel(hard_minus / ground_minus - hard_plus))
    if residual != 0:
        raise AssertionError(residual)
    return sp.factor(ground_minus), sp.factor(hard_plus)


def exact_mixed_arrow_rank_one_audit():
    """Check the two mixed arrows at the odd-NS base and hard levels."""

    b, p1, p2, p3 = sp.symbols("b P1 P2 P3", nonzero=True)
    sample = (b, p1, p2, p3)
    base_labels = (
        sp.Rational(1, 2),
        sp.Rational(1, 4),
        sp.Rational(1, 4),
    )
    hard_labels = (
        sp.Rational(1, 2),
        sp.Rational(3, 4),
        sp.Rational(3, 4),
    )
    bases = []
    normalized = []
    for delta2, delta3 in ((1, -1), (-1, 1)):
        base = corrected_two_step_amplitude(
            base_labels, sample, delta2, delta3
        )
        hard = corrected_two_step_amplitude(
            hard_labels, sample, delta2, delta3
        )
        bases.append(sp.factor(base))
        normalized.append(sp.factor(sp.cancel(hard / base)))
    residual = sp.factor(sp.cancel(normalized[0] - normalized[1]))
    if residual != 0:
        raise AssertionError(residual)
    return tuple(bases), normalized[0]


def exact_crossed_obstruction():
    """Show that the recorded crossed quartic is not the rank-one path.

    This uses the closed hard polynomials only; it does not import or call
    the state/Ward evaluator.  A change of basis attached separately to the
    two Ramond legs could multiply the rank-one GKO answer only by functions
    of those legs.  The nonzero P1 derivative below rules that out directly.
    """

    variables, factorized, crossed, _, _ = hard_channel_obstruction()
    _, p1, _, _ = variables
    quotient_derivative = sp.factor(sp.diff(sp.cancel(crossed / factorized), p1))
    if quotient_derivative == 0:
        raise AssertionError("the crossed/factorized ratio lost its P1 dependence")
    gcd = sp.gcd(sp.Poly(factorized, *variables), sp.Poly(crossed, *variables))
    if gcd.total_degree() != 0:
        raise AssertionError(("unexpected common hard factor", gcd.as_expr()))
    return True


def main():
    check_physical_momentum_dictionary()
    check_arrow_selection_rule()
    check_connection_matrix()
    ground_minus, hard = exact_hard_rank_one_audit()
    mixed_bases, mixed_hard = exact_mixed_arrow_rank_one_audit()
    crossed_obstruction = exact_crossed_obstruction()
    print("physical r,s momentum dictionary: exact")
    print("equal/mixed arrow selection: exact on all 27 stored level triples")
    print("fundamental affine fusion connection: checked to 50 digits")
    print("det(F) = -1 in this highest-weight normalization")
    print("corrected delta=- ground path:")
    print(ground_minus)
    print("normalized delta=- hard path equals delta=+ hard path:")
    print(hard)
    print("normalized two-arrow rank at the hard test: 1")
    print("mixed-arrow base paths (+,-), (-,+):")
    print(mixed_bases)
    print("normalized mixed-arrow hard paths agree:")
    print(mixed_hard)
    print(
        "crossed/factorized hard ratio has nonzero P1 derivative:",
        crossed_obstruction,
    )


if __name__ == "__main__":
    main()
