#!/usr/bin/env python3
"""Exact audits for the diagonal (no-J) branch quadratic.

There are two logically different checks.

``--corpus`` reconstructs the diagonal branch contraction from all sixteen
direct restrictions at every level triple.  It then rewrites the same answer
in terms of the *normalized* branching bilinears

    B bar(B) = A iota(A)/(N_1 N_2 N_3).

No square roots, fitted constants, or proportionality signs occur in this
identity.  The form parity ``f`` and the branch-copy labels ``epsilon_j``
are kept as separate summation indices.

``--scalar-shortcut`` is a falsification test for a stronger claim.  It
removes only the known branch-leg denominators and compares the result with
one four-factor ``ell`` numerator on every reflected momentum sheet.  A
match must have the same quotient at two exact rational samples.  Failure of
this test means that the fixed-eta branch quadratic is not one scalar
double-Liouville product; it says nothing against the published *full local*
double-Liouville correspondence, which also contains its local structure
sum and the published one-leg factors.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import sympy as sp


HERE = Path(__file__).resolve().parent
GRID_DIR = HERE.parent
RAMOND_DIR = GRID_DIR.parent / "ramond_branching_coefficient_check"
for directory in (GRID_DIR, RAMOND_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import compute_grid as grid  # noqa: E402
import compute_ramond_kappa as boundary  # noqa: E402
import bell_tomography as bell  # noqa: E402


I = sp.I
SQRT2 = sp.sqrt(2)
T = (1 + I) / SQRT2


def local_leg_vector(branch_label, sheet=1):
    """Exact diagonal-field coefficients in the raw W basis.

    The order is epsilon=0,1.  ``sheet`` is the sign of the local label
    k=2 n.  This is Eq. (local-diagonal-vector) in the accompanying notes.
    """

    sheet = int(sheet)
    if sheet not in (1, -1):
        raise ValueError("sheet must be +1 or -1")
    mode_count = bell.ramond_mode_count(branch_label)
    parity = mode_count % 2
    vector = [None, None]
    vector[parity] = -sheet * T
    vector[1 - parity] = 2 * sheet * (-1) ** mode_count / T
    return tuple(map(sp.simplify, vector))


def restriction_table(labels, sample, eta):
    """Return A_f[epsilon_2,epsilon_3] and the three raw BPZ norms."""

    n1, n2, n3 = map(sp.Rational, labels)
    b_value, p1, p2, p3 = sample
    amplitudes = {}
    norms = {}
    for form_parity in (0, 1):
        for epsilon2 in (0, 1):
            for epsilon3 in (0, 1):
                _, amplitude = grid.enlarged_raw_three_point(
                    n1,
                    n2,
                    n3,
                    epsilon2,
                    epsilon3,
                    form_parity,
                    eta,
                    b_value,
                    p1,
                    p2,
                    p3,
                )
                key = (form_parity, epsilon2, epsilon3)
                amplitudes[key] = amplitude
                norms[key] = grid.raw_norms(
                    n1,
                    n2,
                    n3,
                    epsilon2,
                    epsilon3,
                    b_value,
                    p1,
                    p2,
                    p3,
                )
    return amplitudes, norms


def direct_branch_quadratic(labels, sample, eta, sheets=(1, 1)):
    """Contract all eight fixed-eta restrictions with no omitted scalar."""

    amplitudes, norms = restriction_table(labels, sample, eta)
    c2 = local_leg_vector(labels[1], sheets[0])
    c3 = local_leg_vector(labels[2], sheets[1])
    direct = 0
    normalized = 0
    checked = 0
    for (form_parity, epsilon2, epsilon3), amplitude in amplitudes.items():
        coefficient = c2[epsilon2] * c3[epsilon3]
        barred = bell.spin_frame_bar(amplitude)
        norm_product = sp.prod(norms[(form_parity, epsilon2, epsilon3)])

        # This is the ratio-safe definition of B bar(B).  Multiplication by
        # the three norms converts the normalized bilinear back to the raw
        # local-field expansion, with no choice of norm square root.
        normalized_bilinear = sp.cancel(amplitude * barred / norm_product)
        direct += coefficient * amplitude * barred
        normalized += coefficient * norm_product * normalized_bilinear
        checked += 1
    direct = sp.factor(sp.cancel(direct))
    normalized = sp.factor(sp.cancel(normalized))
    if sp.factor(sp.cancel(direct - normalized)) != 0:
        raise AssertionError((labels, sample, eta, direct, normalized))
    return direct, checked


def master_branch_quadratic(labels, sample, eta):
    """Reconstruct the same contraction from the two f=0 masters."""

    n1, n2, n3 = map(sp.Rational, labels)
    b_value, p1, p2, p3 = sample
    masters = [
        grid.enlarged_raw_three_point(
            n1, n2, n3, epsilon2, 0, 0, eta,
            b_value, p1, p2, p3,
        )[1]
        for epsilon2 in (0, 1)
    ]
    barred = [bell.spin_frame_bar(value) for value in masters]
    return bell.forward_local_data(
        masters[0], masters[1], barred[0], barred[1], n2, n3, eta
    )[0]


def corpus_audit():
    """Audit 432 direct restrictions at each of the two exact samples."""

    restrictions = 0
    quadratics = 0
    for sample_index, sample in enumerate(grid.SAMPLES, start=1):
        sample_restrictions = 0
        sample_quadratics = 0
        for labels in itertools.product(
            grid.GRID_NS_LEVELS, grid.GRID_R_LEVELS, grid.GRID_R_LEVELS
        ):
            for eta in (1, -1):
                direct, checked = direct_branch_quadratic(
                    labels, sample, eta
                )
                reconstructed = sp.factor(
                    sp.cancel(master_branch_quadratic(labels, sample, eta))
                )
                if sp.factor(sp.cancel(direct - reconstructed)) != 0:
                    raise AssertionError(
                        (sample_index, labels, eta, direct, reconstructed)
                    )
                sample_restrictions += checked
                sample_quadratics += 1
        restrictions += sample_restrictions
        quadratics += sample_quadratics
        print(
            f"sample {sample_index}: {sample_restrictions}/432 direct "
            f"restrictions and {sample_quadratics}/54 fixed-eta "
            "quadratics passed",
            flush=True,
        )
    print(
        f"two-sample total: {restrictions}/864 restrictions and "
        f"{quadratics}/108 quadratics passed"
    )


def leg_clearance(label, momentum, b_value):
    """The denominator Delta_n(P) used by the raw ordered branch state."""

    label = sp.Rational(label)
    q_value = b_value + 1 / b_value
    value = boundary.ell(q_value + 2 * momentum, int(4 * label), b_value)
    if int(4 * label) % 2:
        value *= sp.Pow(2, -sp.Rational(1, 8))
    return sp.factor(value)


def diagonal_value(labels, sample, eta):
    """Return D_sf and its denominator-cleared value."""

    n1, n2, n3 = map(sp.Rational, labels)
    b_value, p1, p2, p3 = sample
    masters = [
        grid.enlarged_raw_three_point(
            n1, n2, n3, epsilon2, 0, 0, eta,
            b_value, p1, p2, p3,
        )[1]
        for epsilon2 in (0, 1)
    ]
    barred = [bell.spin_frame_bar(value) for value in masters]
    diagonal = bell.forward_local_data(
        masters[0], masters[1], barred[0], barred[1], n2, n3, eta
    )[0]
    clearance = sp.prod(
        leg_clearance(label, momentum, b_value) ** 2
        for label, momentum in zip(labels, (p1, p2, p3))
    )
    return sp.factor(diagonal), sp.factor(sp.cancel(diagonal * clearance))


def numerator(labels, sample, sheets):
    """Four-factor numerator with reflected momenta on selected legs."""

    b_value, p1, p2, p3 = sample
    momenta = tuple(s * p for s, p in zip(sheets, (p1, p2, p3)))
    return boundary.numerator_product(*labels, *momenta, b_value)


def fixed_label_audit(labels):
    rows = {}
    for eta in (1, -1):
        cleared = [diagonal_value(labels, sample, eta)[1] for sample in grid.SAMPLES]
        matches = []
        for sheets in itertools.product((1, -1), repeat=3):
            products = [numerator(labels, sample, sheets) for sample in grid.SAMPLES]
            if any(value == 0 for value in products):
                continue
            quotients = [
                sp.factor(sp.cancel(value / product**2))
                for value, product in zip(cleared, products)
            ]
            if sp.factor(sp.cancel(quotients[0] - quotients[1])) == 0:
                matches.append((sheets, quotients[0]))
        rows[eta] = tuple(matches)
    return rows


def scalar_shortcut_audit():
    passed = 0
    total = 0
    for labels in itertools.product(
        grid.GRID_NS_LEVELS, grid.GRID_R_LEVELS, grid.GRID_R_LEVELS
    ):
        result = fixed_label_audit(labels)
        for matches in result.values():
            passed += bool(matches)
            total += 1
        print(f"labels={labels} matches={result}", flush=True)
    print(f"no-J diagonal single-product ratios: {passed}/{total}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        action="store_true",
        help="audit all 432 direct restrictions at both exact samples",
    )
    parser.add_argument(
        "--scalar-shortcut",
        action="store_true",
        help="test the stronger one-ell-product-square shortcut",
    )
    arguments = parser.parse_args()
    if not arguments.corpus and not arguments.scalar_shortcut:
        arguments.corpus = True
    if arguments.corpus:
        corpus_audit()
    if arguments.scalar_shortcut:
        scalar_shortcut_audit()


if __name__ == "__main__":
    main()
