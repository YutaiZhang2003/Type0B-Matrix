#!/usr/bin/env python3
"""Certificate for the scalar ell-product ansatz for Ramond masters.

The first case in which both Ramond strings contain a nonzero mode is

    (n1,n2,n3; epsilon2,eta) = (0,3/4,3/4; 0,-1).

It is a useful gatekeeper for proposed closed formulas.  This script does
four independent things.

1.  It checks the four raw masters against the state-level Ward calculation
    at two exact rational specializations and rewrites them in terms of
    two quartics K and H.  ``--symbolic-ward`` requests the much slower fully
    symbolic Ward re-evaluation.
2.  It proves that H and the two local eta-combinations H +/- i K are
    irreducible over the appropriate exact coefficient fields.
3.  With ``--sparse-search`` it exhausts all one-, two-, and three-term sums
    of the 144 products

       prod_j ell(x_j,m_j),  |m_j| <= 4,

    having the correct total polynomial degree.  The sparse test is made at
    twelve exact rational specializations and evaluated at 40 digits; every
    putative hit is intended to be rechecked symbolically (there are no hits).
4.  It checks the complete constant basis-change question in this product
    class.  Only the K direction in the two-dimensional master-function span
    is an ell product.  Hence no invertible constant 4 by 4 transformation,
    including the local eta-Hadamard transform, turns all four masters into
    scalar NS-like products.

With ``--mixed-parity`` the script additionally tests 1024 products obtained
by choosing the even/odd screening lattice independently in all four
numerator factors and both Ramond leg factors, together with all reflected
Ramond label/momentum representatives.  No one- or two-term formula in that
larger class survives the same twelve specializations.

With ``--master-audit`` the script recomputes the 108 independent
``(epsilon_2,eta)`` masters at both exact samples and checks the complete
52-match/56-failure count.  ``--grid-audit`` is the slower version which
also recomputes the twelve phase-related restrictions at every level triple.

The earlier complete 27-level scan found the original single-product formula
for 208/432 restrictions, equivalently 52/108 master functions; it failed for
the remaining 224/432 restrictions, equivalently 56/108 masters.

This is a falsification certificate for a precise ansatz class, not a claim
that an arbitrary matrix-valued or state-sum formula cannot exist.
"""

from __future__ import annotations

import argparse
import itertools
import time

import numpy as np
import sympy as sp

import compute_grid as grid
import fit_signed_sectors as sectors


I = sp.I
SQRT2 = sp.sqrt(2)
EIGHTH = sp.Pow(2, sp.Rational(1, 8))
HARD_LABELS = (sp.Integer(0), sp.Rational(3, 4), sp.Rational(3, 4))
MASTER_ORDER = ((0, 1), (0, -1), (1, 1), (1, -1))
TEST_SAMPLES = sectors.FIT_SAMPLES[:12]
WARD_SAMPLES = sectors.FIT_SAMPLES[:2]


def hard_polynomials():
    """Return the exact Q-polynomials K and H for the gatekeeper case."""

    q_value, p1, p2, p3 = sp.symbols("Q P1 P2 P3")
    first = q_value / 2 + p1 + p2 + p3
    second = q_value / 2 - p1 + p2 + p3

    # ell(x,3)/2^(1/8)=(x+b)(x+b^{-1})=x^2+Q*x+1.
    product = sp.expand(
        (first**2 + q_value * first + 1)
        * (second**2 + q_value * second + 1)
    )
    hard = sp.expand(
        sp.Rational(57, 16) * q_value**4
        + sp.Rational(19, 2) * q_value**3 * (p2 + p3)
        + q_value**2
        * (
            sp.Rational(3, 2) * p1**2
            + sp.Rational(7, 2) * (p2**2 + p3**2)
            + 25 * p2 * p3
            + sp.Rational(7, 2)
        )
        + q_value
        * (
            2 * p1**2 * (p2 + p3)
            - 2 * (p2**3 + p3**3)
            + 10 * p2 * p3 * (p2 + p3)
            + 4 * (p2 + p3)
        )
        + p1**4
        - 2 * p1**2 * p2**2
        + 4 * p1**2 * p2 * p3
        - 2 * p1**2 * p3**2
        + 2 * p1**2
        + p2**4
        - 4 * p2**3 * p3
        + 6 * p2**2 * p3**2
        + 2 * p2**2
        - 4 * p2 * p3**3
        - 4 * p2 * p3
        + p3**4
        + 2 * p3**2
        + 1
    )
    return (q_value, p1, p2, p3), product, hard


def hard_crossed_identity():
    """Return the compact ell identity for the non-product quartic H.

    Put x_{st}=Q/2+P1+s P2+t P3 and abbreviate

      L = ell(x_{++},2) ell(x_{--},-2),
      E_j = ell(Q+2 P_j,2),  D_j = ell(Q+2 P_j,3).

    Because ell(x,2)=x, ell(x,-2)=x-Q, and
    ell(x,3)=2^(1/8)(x^2+Qx+1), this function returns the exact
    polynomial version of

      H = L^2 + 2 L E_2 E_3 + 2 L + 2^(-1/4) D_2 D_3.
    """

    variables, _, hard = hard_polynomials()
    q_value, p1, p2, p3 = variables
    x_plus_plus = q_value / 2 + p1 + p2 + p3
    x_minus_minus = q_value / 2 + p1 - p2 - p3
    crossed = x_plus_plus * (x_minus_minus - q_value)
    even_second = q_value + 2 * p2
    even_third = q_value + 2 * p3
    odd_second = even_second**2 + q_value * even_second + 1
    odd_third = even_third**2 + q_value * even_third + 1
    right_hand_side = sp.expand(
        crossed**2
        + 2 * crossed * even_second * even_third
        + 2 * crossed
        + odd_second * odd_third
    )
    return hard, right_hand_side


def hard_master_formulas(sample):
    """The four scaled raw masters in terms of K and H."""

    b_value, p1, p2, p3 = sample
    variables, product, hard = hard_polynomials()
    q_symbol, p1_symbol, p2_symbol, p3_symbol = variables
    substitutions = {
        q_symbol: b_value + 1 / b_value,
        p1_symbol: p1,
        p2_symbol: p2,
        p3_symbol: p3,
    }
    product = product.subs(substitutions)
    hard = hard.subs(substitutions)
    common = EIGHTH**2
    return (
        -(1 + I) * common * product,
        -(1 - I) * common * hard,
        SQRT2 * (1 - I) * common * product,
        SQRT2 * I * (1 - I) * common * hard,
    )


def direct_scaled_master(epsilon2, eta, sample):
    return sectors.scaled_raw(
        HARD_LABELS,
        (epsilon2, 0, 0, eta),
        sample,
    )


def exact_gatekeeper_certificate(symbolic_ward=False):
    """Prove the quartic statements and verify the direct Ward result."""

    if symbolic_ward:
        b_value, p1, p2, p3 = sp.symbols("b P1 P2 P3", nonzero=True)
        samples = ((b_value, p1, p2, p3),)
    else:
        samples = WARD_SAMPLES
    for sample in samples:
        expected = hard_master_formulas(sample)
        for (epsilon2, eta), expected_value in zip(MASTER_ORDER, expected):
            direct = direct_scaled_master(epsilon2, eta, sample)
            residual = sp.factor(sp.cancel(direct - expected_value))
            if residual != 0:
                raise AssertionError(
                    "The exact hard-master formula failed for "
                    f"sample={sample}, (epsilon2,eta)={(epsilon2, eta)}: "
                    f"{residual}"
                )

    hard, crossed_form = hard_crossed_identity()
    if sp.expand(hard - crossed_form) != 0:
        raise AssertionError("The compact crossed ell identity for H failed.")
    variables, product, hard = hard_polynomials()
    q_value, p1, p2, p3 = variables
    if not sp.Poly(16 * hard, q_value, p1, p2, p3).is_irreducible:
        raise AssertionError("The hard quartic H unexpectedly factorized.")
    for sign in (1, -1):
        local = sp.Poly(
            16 * (hard + sign * I * product),
            q_value,
            p1,
            p2,
            p3,
            extension=I,
        )
        if not local.is_irreducible:
            raise AssertionError(
                f"The local combination H {sign:+d} i K factorized."
            )
    ward_scope = "symbolically" if symbolic_ward else "at 2 exact samples"
    print(
        "exact hard-master formulas: residual=0 for all four masters "
        + ward_scope
    )
    print(
        "crossed identity: H=L^2+2*L*E2*E3+2*L+2^(-1/4)*D2*D3 "
        "holds exactly"
    )
    print("H irreducible over Q; H+iK and H-iK irreducible over Q(i)")


def numerical_array(rows):
    return np.array(
        [[complex(sp.N(value, 40)) for value in row] for row in rows],
        dtype=complex,
    )


def hard_values(samples):
    variables, _, hard = hard_polynomials()
    q_value, p1, p2, p3 = variables
    values = []
    for b_value, first, second, third in samples:
        values.append(
            hard.subs(
                {
                    q_value: b_value + 1 / b_value,
                    p1: first,
                    p2: second,
                    p3: third,
                }
            )
        )
    return np.array([complex(sp.N(value, 40)) for value in values])


def degree_four_product_matrix(samples):
    return numerical_array(
        [
            [
                sectors.four_argument_ell_product(pattern, sample)
                for pattern in sectors.DEGREE_FOUR_PATTERNS
            ]
            for sample in samples
        ]
    )


def sparse_three_product_certificate():
    """Exhaust all one-, two-, and three-column product fits."""

    matrix = degree_four_product_matrix(TEST_SAMPLES)
    target = hard_values(TEST_SAMPLES)
    tolerance = 1e-10 * max(1.0, float(np.max(np.abs(target))))

    single_hits = 0
    for column in range(matrix.shape[1]):
        vector = matrix[:, column]
        coefficient = np.vdot(vector, target) / np.vdot(vector, vector)
        if np.max(np.abs(coefficient * vector - target)) < tolerance:
            single_hits += 1

    pair_hits = 0
    for first, second in itertools.combinations(range(matrix.shape[1]), 2):
        pair = matrix[:, (first, second)]
        coefficients, _, rank, _ = np.linalg.lstsq(pair, target, rcond=1e-12)
        if rank == 2 and np.max(np.abs(pair @ coefficients - target)) < tolerance:
            pair_hits += 1

    triple_hits = 0
    column_count = matrix.shape[1]
    for first in range(column_count - 2):
        for second in range(first + 1, column_count - 1):
            pair = matrix[:, (first, second)]
            left_vectors, singular_values, _ = np.linalg.svd(
                pair, full_matrices=False
            )
            numerical_rank = int(
                np.count_nonzero(
                    singular_values
                    > 1e-12 * max(1.0, float(singular_values[0]))
                )
            )
            orthogonal = left_vectors[:, :numerical_rank]
            target_remainder = target - orthogonal @ (
                orthogonal.conj().T @ target
            )
            column_remainders = matrix - orthogonal @ (
                orthogonal.conj().T @ matrix
            )
            numerators = column_remainders.conj().T @ target_remainder
            denominators = np.sum(np.abs(column_remainders) ** 2, axis=0)
            coefficients = np.divide(
                numerators,
                denominators,
                out=np.zeros_like(numerators),
                where=denominators > 1e-24,
            )
            residuals = np.max(
                np.abs(
                    column_remainders * coefficients
                    - target_remainder[:, None]
                ),
                axis=0,
            )
            triple_hits += int(
                np.count_nonzero(
                    residuals[second + 1 :] < tolerance
                )
            )

    if (single_hits, pair_hits, triple_hits) != (0, 0, 0):
        raise AssertionError(
            "A sparse degree-four ell sum was found: "
            f"{(single_hits, pair_hits, triple_hits)}"
        )
    print(
        "144-product basis: no one-, two-, or three-term representation "
        "at 12 exact samples"
    )


def constant_basis_change_certificate():
    """Find all product columns in the constant master-function span."""

    master_matrix = numerical_array(
        [hard_master_formulas(sample) for sample in TEST_SAMPLES]
    )
    product_matrix = degree_four_product_matrix(TEST_SAMPLES)
    if np.linalg.matrix_rank(master_matrix, tol=1e-10) != 2:
        raise AssertionError("The hard master-function span should have rank two.")

    hits = []
    for column, pattern in enumerate(sectors.DEGREE_FOUR_PATTERNS):
        coefficients, _, _, _ = np.linalg.lstsq(
            master_matrix, product_matrix[:, column], rcond=1e-12
        )
        residual = np.max(
            np.abs(master_matrix @ coefficients - product_matrix[:, column])
        )
        scale = max(1.0, float(np.max(np.abs(product_matrix[:, column]))))
        if residual < 1e-10 * scale:
            hits.append(pattern)
    expected = [(3, 0, 0, -3)]
    if hits != expected:
        raise AssertionError(
            f"Unexpected ell-product directions in the master span: {hits}"
        )
    print(
        "constant master span: only pattern (3,0,0,-3), the K direction, "
        "is a scalar ell product"
    )
    print(
        "therefore no invertible constant 4x4 basis change makes every "
        "master an NS-like scalar product"
    )


def mixed_parity_candidate(
    sample,
    numerator_sectors,
    leg_sectors,
    label_signs,
    momentum_signs,
):
    b_value, p1, p2, p3 = sample
    q_value = b_value + 1 / b_value
    n1, n2, n3 = HARD_LABELS
    n2 *= label_signs[0]
    n3 *= label_signs[1]
    p2 *= momentum_signs[0]
    p3 *= momentum_signs[1]
    arguments = (
        q_value / 2 + p1 + p2 + p3,
        q_value / 2 - p1 + p2 + p3,
        q_value / 2 + p1 - p2 + p3,
        q_value / 2 + p1 + p2 - p3,
    )
    indices = (
        2 * (n1 + n2 + n3),
        2 * (-n1 + n2 + n3),
        2 * (n1 - n2 + n3),
        2 * (n1 + n2 - n3),
    )
    numerator = sp.prod(
        sectors.one_screening_factor(
            argument,
            sp.Rational(index) / 2,
            b_value,
            parity,
        )
        for argument, index, parity in zip(
            arguments, indices, numerator_sectors
        )
    )
    denominator = sp.prod(
        sectors.one_screening_factor(
            q_value + 2 * momentum,
            2 * label,
            b_value,
            parity,
        )
        for momentum, label, parity in zip(
            (p2, p3), (n2, n3), leg_sectors
        )
    )
    return sp.cancel(numerator / denominator)


def mixed_parity_certificate():
    """Test mixed ell(x,m)/ell(x,m+1) products and reflected legs."""

    parity_choices = ("odd", "even")
    names = tuple(
        itertools.product(
            itertools.product(parity_choices, repeat=4),
            itertools.product(parity_choices, repeat=2),
            sectors.SHEETS,
            sectors.SHEETS,
        )
    )
    rows = []
    target = []
    for sample in TEST_SAMPLES:
        rows.append(
            [mixed_parity_candidate(sample, *name) for name in names]
        )
        b_value, p1, p2, p3 = sample
        q_value = b_value + 1 / b_value
        denominator = sp.prod(
            grid.boundary.ell(q_value + 2 * momentum, 3, b_value)
            for momentum in (p2, p3)
        )
        target.append(hard_master_formulas(sample)[1] / denominator)
    matrix = numerical_array(rows)
    target = np.array([complex(sp.N(value, 40)) for value in target])
    tolerance = 1e-10 * max(1.0, float(np.max(np.abs(target))))

    single_hits = 0
    for column in range(matrix.shape[1]):
        vector = matrix[:, column]
        coefficient = np.vdot(vector, target) / np.vdot(vector, vector)
        single_hits += int(
            np.max(np.abs(coefficient * vector - target)) < tolerance
        )
    pair_hits = 0
    for first, second in itertools.combinations(range(matrix.shape[1]), 2):
        pair = matrix[:, (first, second)]
        coefficients, _, rank, _ = np.linalg.lstsq(pair, target, rcond=1e-12)
        pair_hits += int(
            rank == 2
            and np.max(np.abs(pair @ coefficients - target)) < tolerance
        )
    if (single_hits, pair_hits) != (0, 0):
        raise AssertionError(
            "A mixed-parity sparse formula was found: "
            f"{(single_hits, pair_hits)}"
        )
    print(
        "1024 mixed-parity/reflected products: no one- or two-term "
        "representation at 12 exact samples"
    )


def master_grid_audit():
    """Audit only the 108 independent (epsilon_2,eta) master amplitudes.

    The other twelve restrictions at a fixed level triple are universal
    phase/parity copies of these four masters.  The older full-grid audit
    recomputes all sixteen restrictions.  Here epsilon_3=f=0 is imposed from
    the outset, so only 27*4 amplitudes are evaluated at each exact sample.
    A product counts as a match precisely when at least one of the four
    (P_2,P_3) momentum sheets gives the same exact kappa^2 at both samples.
    """

    grid.kernel_self_checks()
    began = time.perf_counter()
    matched = 0
    failed = 0
    triple_count = 0
    master_choices = tuple(itertools.product((0, 1), (1, -1)))
    for labels in itertools.product(
        grid.GRID_NS_LEVELS, grid.GRID_R_LEVELS, grid.GRID_R_LEVELS
    ):
        certificates = []
        for sample in grid.SAMPLES:
            by_master = {}
            for epsilon2, eta in master_choices:
                by_master[(epsilon2, eta)] = grid.direct_certificate(
                    *labels,
                    epsilon2,
                    0,
                    0,
                    eta,
                    sample,
                )
            certificates.append(by_master)

        triple_matches = 0
        for master in master_choices:
            first = certificates[0][master]["candidates"]
            second = certificates[1][master]["candidates"]
            matching_sheets = tuple(
                sheets
                for sheets in first
                if sp.factor(sp.cancel(first[sheets] - second[sheets])) == 0
            )
            if matching_sheets:
                matched += 1
                triple_matches += 1
            else:
                failed += 1
        triple_count += 1
        elapsed = time.perf_counter() - began
        print(
            f"master audit {triple_count:02d}/27: levels={labels}, "
            f"matches={triple_matches}/4, cumulative={matched}/{4*triple_count}, "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )

    if (matched, failed) != (52, 56):
        raise AssertionError(
            "The exact master-grid count changed: "
            f"matched={matched}, failed={failed}."
        )
    elapsed = time.perf_counter() - began
    print(
        "master-grid audit: 108 exact masters; "
        f"single-product matches={matched}; failures={failed}; "
        f"elapsed={elapsed:.1f}s"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sparse-search",
        action="store_true",
        help="exhaust the 144-product one-, two-, and three-term ansatz",
    )
    parser.add_argument(
        "--mixed-parity",
        action="store_true",
        help="also run the slower 1024-product mixed-parity search",
    )
    parser.add_argument(
        "--grid-audit",
        action="store_true",
        help="rerun the slower complete 432-restriction grid audit",
    )
    parser.add_argument(
        "--master-audit",
        action="store_true",
        help="audit the 108 independent masters without phase-related copies",
    )
    parser.add_argument(
        "--symbolic-ward",
        action="store_true",
        help="rerun the slow fully symbolic state-level Ward calculation",
    )
    arguments = parser.parse_args()
    exact_gatekeeper_certificate(arguments.symbolic_ward)
    if arguments.sparse_search:
        sparse_three_product_certificate()
        constant_basis_change_certificate()
    if arguments.mixed_parity:
        mixed_parity_certificate()
    if arguments.grid_audit:
        grid.full_grid_report()
    if arguments.master_audit:
        master_grid_audit()
    print(
        "full-grid context: original single-product ansatz matched "
        "52/108 masters (208/432 restrictions) and failed for "
        "56/108 masters (224/432 restrictions)"
    )


if __name__ == "__main__":
    main()
