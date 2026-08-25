#!/usr/bin/env python3
"""Finite ground-path formula for the fixed-chiral Ramond masters.

The four-factor NS blow-up product does not survive as one product for a
fixed Ramond chiral structure.  What *does* survive is the one-leg ell
denominator.  After that denominator is cleared, the answer is a finite
sum over the auxiliary/physical paths in the three ordered chi strings.

This file implements that formula independently of
``compute_grid.enlarged_raw_three_point``.  It also contains an exact
certificate for the first irreducible crossed numerator,
``(n1,n2,n3)=(0,3/4,3/4)``.

For a positive branch label n put

    Delta_n(P) = 2^(-1/8 [4n odd]) ell(Q+2P,4n).

The factor removes the conventional 2^(1/8) in an odd ell product.  If
``C_{n,p}`` is a component coefficient of the raw chi-string branch, put
``Cbar_{n,p}=Delta_n C_{n,p}``.  The closed path formula is

    R = 1/(Delta_1 Delta_2 Delta_3)
        sum_{p1,p2,p3} (-1)^Koszul Cbar_1 Cbar_2 Cbar_3
        Pfaffian_aux(p1,p2,p3) Ward_phys^eta(p1,p2,p3).

``fermion_value_virasoro`` evaluates the displayed finite Pfaffian (in an
equivalent Virasoro basis) and ``PhysicalNRREvaluator`` evaluates the
finite triangular Ward polynomial.  Thus the formula has no fitted
continuous coefficient.  The only sums are the explicitly constructed
chi-string paths.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
import sys

import sympy as sp


THIS_DIR = Path(__file__).resolve().parent
GRID_DIR = THIS_DIR.parent
if str(GRID_DIR) not in sys.path:
    sys.path.insert(0, str(GRID_DIR))

import compute_grid as grid  # noqa: E402


I = sp.I
SQRT2 = sp.sqrt(2)


def stripped_ell(x, index, b_value):
    """The notes' ell with its fixed odd 2^(1/8) removed."""

    index = int(index)
    value = grid.boundary.ell(x, index, b_value)
    if index % 2:
        value *= sp.Pow(2, -sp.Rational(1, 8))
    return sp.factor(value)


def leg_clearing_factor(branch_label, b_value, momentum):
    """Return Delta_n(P), the common denominator of a raw branch."""

    branch_label = sp.Rational(branch_label)
    index = int(4 * branch_label)
    q_value = sp.cancel(b_value + 1 / b_value)
    return stripped_ell(q_value + 2 * momentum, index, b_value)


def _path_data(labels, epsilon2, sample):
    n1, n2, n3 = map(sp.Rational, labels)
    b_value, p1, p2, p3 = sample
    q_value = sp.cancel(b_value + 1 / b_value)
    deltas = (
        leg_clearing_factor(n1, b_value, p1),
        leg_clearing_factor(n2, b_value, p2),
        leg_clearing_factor(n3, b_value, p3),
    )
    components = (
        grid.ns_components(n1, q_value, p1),
        grid.ramond_components(n2, int(epsilon2), q_value, p2),
        grid.ramond_components(n3, 0, q_value, p3),
    )
    return q_value, deltas, components


def cleared_ground_path_sum(labels, epsilon2, eta, sample):
    """Return (cleared numerator, denominator, auxiliary parity).

    Every summand is one choice of a component in each of the three
    ordered chi strings.  Multiplication of each component coefficient by
    its leg's Delta makes the common ell denominator manifest before the
    paths are summed.
    """

    n1, n2, n3 = map(sp.Rational, labels)
    epsilon2 = int(epsilon2)
    eta = int(eta)
    b_value, p1, p2, p3 = sample
    q_value, deltas, components = _path_data(labels, epsilon2, sample)

    central_charge = sp.Rational(3, 2) + 3 * q_value**2
    h1 = (q_value**2 / 4 - p1**2) / 2
    h2 = sp.Rational(1, 16) + q_value**2 / 8 - p2**2 / 2
    h3 = sp.Rational(1, 16) + q_value**2 / 8 - p3**2 / 2
    physical = grid.PhysicalNRREvaluator(
        0,
        eta,
        h1,
        h2,
        h3,
        p2 / SQRT2,
        p3 / SQRT2,
        central_charge,
    )
    auxiliary_parity = (int(2 * n1) + epsilon2) % 2

    first, second, third = components
    numerator = sp.Integer(0)
    for auxiliary1, word1, coefficient1 in first:
        physical_parity1 = grid.state_parity(word1)
        auxiliary_parity1 = len(auxiliary1) % 2
        coefficient1 = sp.cancel(deltas[0] * coefficient1)
        for (
            auxiliary2,
            auxiliary_ground2,
            word2,
            physical_ground2,
            coefficient2,
        ) in second:
            physical_parity2 = grid.state_parity(word2, physical_ground2)
            auxiliary_parity2 = (
                len(auxiliary2) + auxiliary_ground2
            ) % 2
            coefficient2 = sp.cancel(deltas[1] * coefficient2)
            for (
                auxiliary3,
                auxiliary_ground3,
                word3,
                physical_ground3,
                coefficient3,
            ) in third:
                auxiliary_parity3 = (
                    len(auxiliary3) + auxiliary_ground3
                ) % 2
                auxiliary_value = grid.fermion_value_virasoro(
                    auxiliary_parity,
                    auxiliary1,
                    auxiliary2,
                    auxiliary_ground2,
                    auxiliary3,
                    auxiliary_ground3,
                )
                if auxiliary_value == 0:
                    continue
                physical_value = physical.value(
                    word1,
                    word2,
                    physical_ground2,
                    word3,
                    physical_ground3,
                )
                if physical_value == 0:
                    continue
                koszul = (
                    physical_parity1
                    * (auxiliary_parity2 + auxiliary_parity3)
                    + physical_parity2 * auxiliary_parity3
                )
                coefficient3_cleared = sp.cancel(deltas[2] * coefficient3)
                numerator += (
                    (-1) ** koszul
                    * coefficient1
                    * coefficient2
                    * coefficient3_cleared
                    * auxiliary_value
                    * physical_value
                )

    denominator = sp.prod(deltas)
    return (
        sp.factor(sp.cancel(numerator)),
        sp.factor(denominator),
        auxiliary_parity,
    )


def master_from_paths(labels, epsilon2, eta, sample):
    """Evaluate R_{epsilon2}^{(eta)} from the finite path formula."""

    numerator, denominator, auxiliary_parity = cleared_ground_path_sum(
        labels, epsilon2, eta, sample
    )
    return auxiliary_parity, sp.factor(sp.cancel(numerator / denominator))


def chiral_channel_pair(labels, epsilon2, sample):
    """Return the two coefficients multiplying 1 and eta."""

    _, plus = master_from_paths(labels, epsilon2, 1, sample)
    _, minus = master_from_paths(labels, epsilon2, -1, sample)
    return (
        sp.factor(sp.cancel((plus + minus) / 2)),
        sp.factor(sp.cancel((plus - minus) / 2)),
    )


def hard_crossed_certificate():
    """Certify the irreducible (0,3/4,3/4; epsilon2=0) numerator."""

    q, p1, p2, p3 = sp.symbols("Q P_1 P_2 P_3")
    x = p2 + p3
    y = p2 - p3
    d2 = 4 * p2**2 + 6 * q * p2 + 2 * q**2 + 1
    d3 = 4 * p3**2 + 6 * q * p3 + 2 * q**2 + 1
    a = p1**2 - y**2 + sp.Rational(3, 4) * q**2 + q * x + 1
    c = y**2 - (q + x) ** 2 - 1
    hard = sp.expand(16 * (a**2 - c**2 + d2 * d3))

    expected = sp.expand(
        16 * y**4
        - 72 * q**2 * y**2
        - 64 * q * x * y**2
        - 32 * p1**2 * y**2
        + 32 * y**2
        + 57 * q**4
        + 152 * q**3 * x
        + 128 * q**2 * x**2
        + 24 * q**2 * p1**2
        + 56 * q**2
        + 32 * q * x**3
        + 32 * q * x * p1**2
        + 64 * q * x
        + 16 * p1**4
        + 32 * p1**2
        + 16
    )
    if sp.expand(hard - expected) != 0:
        raise AssertionError("The crossed hard-numerator identity failed.")

    coefficient, factors = sp.Poly(
        hard,
        p1,
        domain=sp.QQ.frac_field(q, p2, p3),
    ).factor_list()
    if not (
        coefficient == 1
        and len(factors) == 1
        and factors[0][0].degree() == 4
        and factors[0][1] == 1
    ):
        raise AssertionError("The hard quartic unexpectedly factorized.")
    return hard


def hard_ell_master(epsilon2, eta, sample):
    """Closed ell formula for the first genuinely crossed master.

    This is the smallest case in which the general path polynomial is not
    one scalar NS-like product.  The two ground channels nevertheless close
    on a two by two bilinear.  In the notation of the notes,

        L   = ell(x_{++},2) ell(x_{--},-2),
        E_j = ell(Q+2P_j,2),
        d_j = 2^(-1/8) ell(Q+2P_j,3),

    and the non-product numerator is

        (1,L) [[d2*d3, 1+E2*E3], [1+E2*E3, 1]] (1,L)^T.

    The parity-copy relation then supplies epsilon2=1.  No Ward evaluator
    is used in this function.
    """

    epsilon2 = int(epsilon2)
    eta = int(eta)
    b_value, p1, p2, p3 = sample
    q_value = sp.cancel(b_value + 1 / b_value)
    x_plus_plus = q_value / 2 + p1 + p2 + p3
    x_minus_minus = q_value / 2 + p1 - p2 - p3
    d2 = stripped_ell(q_value + 2 * p2, 3, b_value)
    d3 = stripped_ell(q_value + 2 * p3, 3, b_value)
    e2 = grid.boundary.ell(q_value + 2 * p2, 2, b_value)
    e3 = grid.boundary.ell(q_value + 2 * p3, 2, b_value)
    crossed = (
        grid.boundary.ell(x_plus_plus, 2, b_value)
        * grid.boundary.ell(x_minus_minus, -2, b_value)
    )

    if eta == 1:
        numerator = (
            stripped_ell(x_plus_plus, 3, b_value)
            * stripped_ell(x_minus_minus, -3, b_value)
        )
        answer = -(1 + I) * numerator / (d2 * d3)
    elif eta == -1:
        ground_vector = sp.Matrix([[1, crossed]])
        ground_kernel = sp.Matrix(
            [[d2 * d3, 1 + e2 * e3], [1 + e2 * e3, 1]]
        )
        numerator = (ground_vector * ground_kernel * ground_vector.T)[0]
        answer = -(1 - I) * numerator / (d2 * d3)
    else:
        raise ValueError("eta must be +1 or -1")

    if epsilon2:
        answer *= I * SQRT2 * eta
    return sp.factor(sp.cancel(answer))


def audit_hard_ell_formula():
    """Check the crossed two-channel formula for all four hard masters."""

    labels = (sp.Integer(0), sp.Rational(3, 4), sp.Rational(3, 4))
    checked = 0
    for sample in grid.SAMPLES:
        for epsilon2, eta in itertools.product((0, 1), (1, -1)):
            candidate = hard_ell_master(epsilon2, eta, sample)
            _, direct = grid.enlarged_raw_three_point(
                *labels, epsilon2, 0, 0, eta, *sample
            )
            residual = sp.factor(sp.cancel(candidate - direct))
            if residual != 0:
                raise AssertionError(
                    "Hard two-channel ell formula failed at "
                    f"sample={sample}, epsilon2={epsilon2}, eta={eta}: "
                    f"{residual}"
                )
            checked += 1
    print(f"hard two-channel ell formula: {checked} exact residuals are zero")


def audit(samples):
    """Check every one of the 108 known masters at the requested samples."""

    labels_grid = itertools.product(
        grid.GRID_NS_LEVELS,
        grid.GRID_R_LEVELS,
        grid.GRID_R_LEVELS,
    )
    checked = 0
    for labels in labels_grid:
        for sample in samples:
            for epsilon2, eta in itertools.product((0, 1), (1, -1)):
                auxiliary_parity, candidate = master_from_paths(
                    labels, epsilon2, eta, sample
                )
                direct_parity, direct = grid.enlarged_raw_three_point(
                    *labels,
                    epsilon2,
                    0,
                    0,
                    eta,
                    *sample,
                )
                residual = sp.factor(sp.cancel(candidate - direct))
                if auxiliary_parity != direct_parity or residual != 0:
                    raise AssertionError(
                        "Path formula failed at "
                        f"labels={labels}, epsilon2={epsilon2}, eta={eta}, "
                        f"sample={sample}: parity=({auxiliary_parity},"
                        f"{direct_parity}), residual={residual}"
                    )
                checked += 1
        print(f"path formula: labels={labels} residual=0", flush=True)
    print(f"path formula: {checked} exact master evaluations residual=0")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit",
        action="store_true",
        help="check all 108 masters at both exact samples (216 evaluations)",
    )
    arguments = parser.parse_args()

    hard_crossed_certificate()
    print("hard crossed numerator: identity and irreducibility residual=0")
    audit_hard_ell_formula()
    if arguments.audit:
        audit(grid.SAMPLES)
    else:
        labels = (sp.Integer(0), sp.Rational(3, 4), sp.Rational(3, 4))
        for epsilon2, eta in itertools.product((0, 1), (1, -1)):
            _, value = master_from_paths(
                labels, epsilon2, eta, grid.SAMPLES[0]
            )
            print(
                f"labels={labels} epsilon2={epsilon2} eta={eta} "
                f"R={sp.N(value, 12)}"
            )


if __name__ == "__main__":
    main()
