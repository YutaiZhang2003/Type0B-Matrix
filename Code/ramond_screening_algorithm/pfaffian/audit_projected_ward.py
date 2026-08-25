#!/usr/bin/env python3
"""Optional Ward audit of the resolved factorized screening projection.

Production modules do not import the old Ward evaluator.  This harness
checks the exact boundary of their scope: all ground components and the
hard factorized component agree, while the first longer natural staircase
is not either fixed raw Ramond chiral form up to a constant phase.
"""

from __future__ import annotations

from pathlib import Path
import sys

import sympy as sp


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
GRID = ROOT / "python" / "ramond_three_point_grid"
if str(GRID) not in sys.path:
    sys.path.insert(0, str(GRID))

import compute_grid as ward  # noqa: E402

from .audit_ward_hyperplanes import rationalize_gamma  # noqa: E402
from .boundary_zero_modes import projected_selberg_ratio  # noqa: E402
from .screening_pfaffian import natural_ramond_parity  # noqa: E402


SAMPLES = (
    (sp.Rational(3, 2), sp.Rational(2, 5), sp.Rational(3, 7)),
    (sp.Rational(5, 3), sp.Rational(3, 8), sp.Rational(5, 9)),
)


def _screen_and_raw(labels, form_parity, eta, sample):
    n1, n2, n3 = map(sp.Rational, labels)
    b, p2, p3 = sample
    q = b + 1 / b
    count = int(2 * (n1 + n2 + n3))
    p1 = -count * b - q / 2 - p2 - p3
    A = -b * (q / 2 + p3) - sp.Rational(1, 2)
    B = -b * (q / 2 + p2) - sp.Rational(1, 2)
    g = -b * q / 2
    screened = rationalize_gamma(
        projected_selberg_ratio(
            n1, n2, n3, form_parity, eta, A, B, g
        )
    )
    epsilon2 = natural_ramond_parity(n2)
    epsilon3 = natural_ramond_parity(n3)
    raw = ward.enlarged_raw_three_point(
        n1,
        n2,
        n3,
        epsilon2,
        epsilon3,
        form_parity,
        eta,
        b,
        p1,
        p2,
        p3,
    )[1]
    return screened, raw


def audit():
    hard = (sp.Integer(0), sp.Rational(3, 4), sp.Rational(3, 4))
    for form_parity in (0, 1):
        screened, raw = _screen_and_raw(hard, form_parity, 1, SAMPLES[0])
        if sp.factor(sp.cancel(screened - raw)) != 0:
            raise AssertionError((hard, form_parity, screened, raw))
    print("hard factorized projection: f=0,1 exact Ward checks passed")

    longer = (sp.Integer(0), sp.Rational(5, 4), sp.Rational(5, 4))
    quotients = []
    for sample in SAMPLES:
        screened, raw = _screen_and_raw(longer, 0, 1, sample)
        quotient = sp.factor(sp.cancel(raw / screened))
        if quotient == 1:
            raise AssertionError(("unexpected long-staircase equality", sample))
        quotients.append(quotient)
    if sp.factor(sp.cancel(quotients[0] - quotients[1])) == 0:
        raise AssertionError(("unexpected constant recoupling", quotients))
    print(
        "long natural staircase: raw eta=+ / screened projection is "
        f"momentum dependent: {tuple(quotients)}"
    )


if __name__ == "__main__":
    audit()
