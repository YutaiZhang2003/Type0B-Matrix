#!/usr/bin/env python3
"""Rank of the parity-copy sum in the GSO-even Ramond sewing.

At fixed branch labels the two ordinary Virasoro blocks do not depend on
the Ramond copy labels.  All copy dependence is therefore in the finite
sum of the two branching products.  This script proves that varying the
two Ramond lift signs gives two independent diagonal bilinears in the
masters R_0 and R_1.  A single diagonal local-Phi quadratic form cannot
encode all lift-resolved sewn blocks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import sympy as sp


HERE = Path(__file__).resolve().parent
GRID_DIR = HERE.parent
if str(GRID_DIR) not in sys.path:
    sys.path.insert(0, str(GRID_DIR))

import compute_grid as grid  # noqa: E402
import bell_tomography as bell  # noqa: E402


def sewing_row(n1, n3, f, g, lift2, lift3, eta_left=1, eta_right=1):
    """Coefficient row on (R0*barR0,R1*barR1).

    The common first-edge lift and all nonzero norm factors are omitted;
    neither changes the rank.  ``eta_left`` and ``eta_right`` label the two
    chiral Ramond forms at the two oriented vertices.
    """

    a = int(2 * sp.Rational(n1)) % 2
    m3 = bell.ramond_mode_count(n3)
    r3_squared = sp.Integer(2) ** ((-1) ** (m3 + 1))
    total_parity = (int(f) + int(g)) % 2
    h = (total_parity - a) % 2
    common = (int(eta_left) * int(eta_right)) ** int(f)
    row = []
    for epsilon2 in (0, 1):
        epsilon3 = (h - epsilon2) % 2
        koszul = (-1) ** (
            a * epsilon2 + a * epsilon3 + epsilon2 * epsilon3
        )
        coefficient = (
            common
            * koszul
            * int(lift2) ** epsilon2
            * int(lift3) ** epsilon3
            * r3_squared**epsilon3
        )
        row.append(sp.simplify(coefficient))
    return tuple(row)


def closed_row(n1, n3, f, g, lift2, lift3, eta_left=1, eta_right=1):
    """The same row after doing the two-term parity sum algebraically."""

    a = int(2 * sp.Rational(n1)) % 2
    m3 = bell.ramond_mode_count(n3)
    r3_squared = sp.Integer(2) ** ((-1) ** (m3 + 1))
    h = (int(f) + int(g) - a) % 2
    common = (int(eta_left) * int(eta_right)) ** int(f)
    if h == 0:
        return common, -common * int(lift2) * int(lift3) * r3_squared
    sign = (-1) ** a * common
    return sign * int(lift3) * r3_squared, sign * int(lift2)


def exact_rank_audit():
    """Check both parity cases and every lift choice exactly."""

    checked = 0
    for n1 in grid.GRID_NS_LEVELS:
        for n3 in grid.GRID_R_LEVELS:
            for f in (0, 1):
                for g in (0, 1):
                    for eta_left in (1, -1):
                        for eta_right in (1, -1):
                            rows = []
                            for lift2 in (1, -1):
                                for lift3 in (1, -1):
                                    direct = sewing_row(
                                        n1,
                                        n3,
                                        f,
                                        g,
                                        lift2,
                                        lift3,
                                        eta_left,
                                        eta_right,
                                    )
                                    expected = closed_row(
                                        n1,
                                        n3,
                                        f,
                                        g,
                                        lift2,
                                        lift3,
                                        eta_left,
                                        eta_right,
                                    )
                                    if direct != expected:
                                        raise AssertionError(
                                            (
                                                n1,
                                                n3,
                                                f,
                                                g,
                                                eta_left,
                                                eta_right,
                                                direct,
                                                expected,
                                            )
                                        )
                                    rows.append(direct)
                            rank = sp.Matrix(rows).rank()
                            if rank != 2:
                                raise AssertionError(
                                    (
                                        n1,
                                        n3,
                                        f,
                                        g,
                                        eta_left,
                                        eta_right,
                                        rows,
                                        rank,
                                    )
                                )
                            checked += 1
    print(
        "lift-resolved parity-copy row rank: "
        f"2 for all {checked} low-level parity cases"
    )


def direct_sum(labels, sample, lift2, lift3, eta=1, f=0, g=0):
    """Evaluate the allowed epsilon sum from the state-level Ward code."""

    n1, n2, n3 = map(sp.Rational, labels)
    b_value, p1, p2, p3 = sample
    a = int(2 * n1) % 2
    h = (f + g - a) % 2
    total = 0
    masters = []
    for epsilon2 in (0, 1):
        master = grid.enlarged_raw_three_point(
            n1, n2, n3, epsilon2, 0, 0, eta,
            b_value, p1, p2, p3,
        )[1]
        masters.append(master)
        epsilon3 = (h - epsilon2) % 2
        amplitude = grid.enlarged_raw_three_point(
            n1, n2, n3, epsilon2, epsilon3, f, eta,
            b_value, p1, p2, p3,
        )[1]
        barred = bell.spin_frame_bar(amplitude)
        koszul = (-1) ** (
            a * epsilon2 + a * epsilon3 + epsilon2 * epsilon3
        )
        total += (
            koszul
            * lift2**epsilon2
            * lift3**epsilon3
            * amplitude
            * barred
        )
    row = sewing_row(n1, n3, f, g, lift2, lift3, eta, eta)
    expected = sum(
        row[epsilon] * masters[epsilon]
        * bell.spin_frame_bar(masters[epsilon])
        for epsilon in (0, 1)
    )
    residual = sp.factor(sp.cancel(total - expected))
    if residual != 0:
        raise AssertionError((labels, lift2, lift3, residual))
    return sp.factor(total), tuple(masters), row


def low_level_counterexamples():
    sample = (
        sp.Rational(3, 2),
        sp.Rational(1, 3),
        sp.Rational(2, 5),
        sp.Rational(3, 7),
    )

    ground = (sp.Integer(0), sp.Rational(1, 4), sp.Rational(1, 4))
    plus, masters, row_plus = direct_sum(ground, sample, 1, 1)
    minus, _, row_minus = direct_sum(ground, sample, 1, -1)
    if row_plus != (1, sp.Rational(-1, 2)):
        raise AssertionError(row_plus)
    if row_minus != (1, sp.Rational(1, 2)):
        raise AssertionError(row_minus)
    if (
        sp.simplify(plus - sp.Rational(3, 2)) != 0
        or sp.simplify(minus - sp.Rational(5, 2)) != 0
    ):
        raise AssertionError((masters, plus, minus))
    determinant = sp.Matrix([row_plus, row_minus]).det()
    if determinant != 1:
        raise AssertionError(determinant)
    print(
        "ground counterexample: rows=(1,-1/2),(1,1/2), "
        "det=1, values=3/2,5/2"
    )

    generic = (
        sp.Rational(1, 2),
        sp.Rational(3, 4),
        sp.Rational(3, 4),
    )
    first, _, first_row = direct_sum(generic, sample, 1, 1)
    second, _, second_row = direct_sum(generic, sample, 1, -1)
    determinant = sp.Matrix([first_row, second_row]).det()
    if determinant == 0 or sp.simplify(first - second) == 0:
        raise AssertionError((first_row, second_row, first, second))
    print(
        "generic counterexample (1/2,3/4,3/4): "
        f"rows={first_row},{second_row}, det={determinant}, "
        f"values~={sp.N(first, 8)},{sp.N(second, 8)}"
    )


def main():
    exact_rank_audit()
    low_level_counterexamples()


if __name__ == "__main__":
    main()
