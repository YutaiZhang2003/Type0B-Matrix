#!/usr/bin/env python3
"""Full stored-grid audit of the published two-holonomy product.

The production module :mod:`colored_staircase` is state-free.  This file is
the deliberately separate validation harness and is the only colored-core
file which imports the old Ward evaluator.

For every one of the 108 independent ``(epsilon_2,eta)`` masters and both
stored exact samples, we ask a sharply defined question: is the
denominator-cleared master a momentum-independent multiple of one
published colored-bifundamental entry?  Only the two path pairs permitted
by ``2*n1`` parity are accepted.  This does not fit a coefficient: the
quotient obtained at sample one must be exactly identical at sample two.

The test is useful even though it is expected to fail.  arXiv:1210.7454
computes a vertex in ``H + sl(2)_2 + NSR``.  A fixed ``F + NSR`` chiral
master additionally needs the WZW/Racah projection.  Enumerating the full
grid prevents a low-level coincidence from being mistaken for that
projection.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import itertools
from pathlib import Path
import sys

import sympy as sp


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
GRID_DIR = ROOT / "python 2" / "ramond_three_point_grid"
if str(GRID_DIR) not in sys.path:
    sys.path.insert(0, str(GRID_DIR))

import compute_grid as grid  # noqa: E402

from .colored_staircase import (  # noqa: E402
    allowed_path_pairs,
    ramond_holonomy_matrix,
)


MASTER_CHOICES = tuple(itertools.product((0, 1), (1, -1)))


@lru_cache(None)
def colored_matrix(n2, n3, sample):
    b, p1, p2, p3 = sample
    return ramond_holonomy_matrix(n2, n3, b, p1, p2, p3)


def leg_denominator(labels, sample):
    b, p1, p2, p3 = sample
    q = b + 1 / b
    return sp.prod(
        grid.boundary.ell(q + 2 * momentum, int(4 * label), b)
        for label, momentum in zip(labels, (p1, p2, p3))
    )


@lru_cache(None)
def direct_cleared(labels, epsilon2, eta, sample):
    _, raw = grid.enlarged_raw_three_point(
        *labels,
        int(epsilon2),
        0,
        0,
        int(eta),
        *sample,
    )
    return sp.factor(sp.cancel(raw * leg_denominator(labels, sample)))


def exact_quotient(value, direction):
    if direction == 0:
        return None
    return sp.factor(sp.cancel(value / direction))


def audit_master(labels, epsilon2, eta):
    allowed = allowed_path_pairs(labels[0])
    quotients = {}
    for pair in allowed:
        values = []
        for sample in grid.SAMPLES:
            matrix = colored_matrix(labels[1], labels[2], sample)
            target = direct_cleared(labels, epsilon2, eta, sample)
            values.append(exact_quotient(target, matrix[pair]))
        if None not in values and sp.factor(sp.cancel(values[0] - values[1])) == 0:
            quotients[pair] = values[0]
    return quotients


def full_audit() -> dict[str, int]:
    checked = 0
    matched = 0
    triples_with_match = 0
    by_eta = {1: 0, -1: 0}
    by_epsilon = {0: 0, 1: 0}
    for triple_number, labels in enumerate(
        itertools.product(
            grid.GRID_NS_LEVELS,
            grid.GRID_R_LEVELS,
            grid.GRID_R_LEVELS,
        ),
        start=1,
    ):
        labels = tuple(map(sp.Rational, labels))
        current = 0
        for epsilon2, eta in MASTER_CHOICES:
            hits = audit_master(labels, epsilon2, eta)
            checked += 1
            if hits:
                matched += 1
                current += 1
                by_eta[eta] += 1
                by_epsilon[epsilon2] += 1
        triples_with_match += bool(current)
        print(
            f"colored audit {triple_number:02d}/27: levels={labels}, "
            f"single-entry matches={current}/4",
            flush=True,
        )

    if checked != 108:
        raise AssertionError(f"expected 108 masters, checked {checked}")
    report = {
        "checked_masters": checked,
        "matched_masters": matched,
        "failed_masters": checked - matched,
        "triples_with_match": triples_with_match,
        "eta_plus_matches": by_eta[1],
        "eta_minus_matches": by_eta[-1],
        "epsilon_zero_matches": by_epsilon[0],
        "epsilon_one_matches": by_epsilon[1],
    }
    print("colored two-core full-grid report:", report)
    print(
        "interpretation: a match means one published holonomy entry works "
        "up to one momentum-independent convention factor; a failure is "
        "the missing trivalent/WZW-Racah projection, not a numerical error."
    )
    return report


def hard_direct_check() -> None:
    """Check the fully normalized hard spin-frame map against Ward data."""

    labels = (sp.Integer(0), sp.Rational(3, 4), sp.Rational(3, 4))
    for sample in grid.SAMPLES:
        b, p1, p2, p3 = sample
        q = b + 1 / b
        matrix = colored_matrix(labels[1], labels[2], sample)
        d2 = (q + 2 * p2) ** 2 + q * (q + 2 * p2) + 1
        d3 = (q + 2 * p3) ** 2 + q * (q + 2 * p3) + 1
        expected = -(1 + sp.I) * matrix[1, 1] / (d2 * d3)
        direct = grid.enlarged_raw_three_point(
            *labels, 0, 0, 0, 1, *sample
        )[1]
        if sp.factor(sp.cancel(expected - direct)) != 0:
            raise AssertionError((sample, sp.factor(expected - direct)))
    print("hard R_0^+ normalization: two exact Ward residuals zero")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hard-only",
        action="store_true",
        help="run only the quick hard normalization check",
    )
    arguments = parser.parse_args()
    hard_direct_check()
    if not arguments.hard_only:
        full_audit()


if __name__ == "__main__":
    main()
