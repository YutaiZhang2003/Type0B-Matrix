#!/usr/bin/env python3
"""Optional audit of the primary screening ratio on neutrality planes.

This file, and only this file in the directory, imports the old Ward
evaluator.  The production Pfaffian/Selberg modules do not.  Its purpose is
to distinguish a genuine screening oracle from a shifted-primary shortcut.
"""

from __future__ import annotations

from pathlib import Path
import sys

import sympy as sp


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
GRID = ROOT / "python 2" / "ramond_three_point_grid"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(GRID) not in sys.path:
    sys.path.insert(0, str(GRID))

import compute_grid as ward  # noqa: E402

from python.ramond_screening_algorithm.pfaffian.special_oracle import (  # noqa: E402
    primary_shift_ratio,
)


def _gamma_at_rational(argument):
    argument = sp.Rational(argument)
    floor = int(sp.floor(argument))
    remainder = argument - floor
    if remainder == 0:
        remainder = sp.Integer(1)
        floor -= 1
    coefficient = sp.Integer(1)
    if floor >= 0:
        for shift in range(floor):
            coefficient *= remainder + shift
    else:
        for shift in range(floor, 0):
            coefficient /= remainder + shift
    return coefficient, remainder


def rationalize_gamma(expression):
    """Cancel Gamma values whose rational arguments differ by integers."""

    bases = {}

    def replace(item):
        if item.func != sp.gamma or not item.args[0].is_Rational:
            return item
        coefficient, remainder = _gamma_at_rational(item.args[0])
        bases.setdefault(remainder, sp.Symbol(f"Gamma_{str(remainder).replace('/', '_') }"))
        return coefficient * bases[remainder]

    return sp.factor(sp.cancel(expression.replace(lambda item: item.func == sp.gamma, replace)))


SAMPLES = (
    (sp.Rational(3, 2), sp.Rational(2, 5), sp.Rational(3, 7)),
    (sp.Rational(5, 3), sp.Rational(3, 8), sp.Rational(5, 9)),
)


def restricted_raw(labels, eta, sample, epsilon2=0, epsilon3=0):
    n1, n2, n3 = map(sp.Rational, labels)
    b, p2, p3 = sample
    q = b + 1 / b
    screenings = int(2 * (n1 + n2 + n3))
    p1 = -screenings * b - q / 2 - p2 - p3
    ratio = rationalize_gamma(primary_shift_ratio(n1, n2, n3, b, p2, p3))
    if ratio.free_symbols:
        raise AssertionError((labels, sample, ratio))
    raw = ward.enlarged_raw_three_point(
        n1,
        n2,
        n3,
        epsilon2,
        epsilon3,
        0,
        eta,
        b,
        p1,
        p2,
        p3,
    )[1]
    return sp.factor(sp.cancel(raw / ratio))


def audit():
    # With M2,M3 <= 1 the maximal-screening value has no nontrivial
    # staircase minor.  The selected eta is the charge-preserving channel.
    short_cases = (
        (0, sp.Rational(1, 4), sp.Rational(1, 4)),
        (0, sp.Rational(1, 4), sp.Rational(3, 4)),
        (0, sp.Rational(3, 4), sp.Rational(1, 4)),
        (0, sp.Rational(3, 4), sp.Rational(3, 4)),
        (sp.Rational(1, 2), sp.Rational(1, 4), sp.Rational(1, 4)),
        (sp.Rational(1, 2), sp.Rational(1, 4), sp.Rational(3, 4)),
    )
    for labels in short_cases:
        n1, n2, n3 = map(sp.Rational, labels)
        M2 = int(2 * n2 - sp.Rational(1, 2))
        M3 = int(2 * n3 - sp.Rational(1, 2))
        eta = (-1) ** int(2 * n1 + M2 + M3)
        values = tuple(restricted_raw(labels, eta, sample) for sample in SAMPLES)
        if sp.factor(values[0] - values[1]) != 0:
            raise AssertionError((labels, eta, values))
        print(f"levels={labels}, eta={eta:+d}, raw/primary-ratio={values[0]}")

    # The first longer-staircase certificate.  Neither eta ratio is a
    # momentum-independent spin-frame constant.  Hence the missing factor
    # cannot be repaired by a phase or by changing the primary labels.
    labels = (0, sp.Rational(5, 4), sp.Rational(5, 4))
    for eta in (1, -1):
        values = tuple(restricted_raw(labels, eta, sample) for sample in SAMPLES)
        if sp.factor(values[0] - values[1]) == 0:
            raise AssertionError(("unexpected primary match", eta, values))
        print(f"long-staircase levels={labels}, eta={eta:+d}: unequal ratios {values}")


if __name__ == "__main__":
    audit()
