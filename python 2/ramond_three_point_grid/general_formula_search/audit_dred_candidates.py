#!/usr/bin/env python3
"""Search and certify the exact local diagonal branching projection.

The published local NS--R--R field pairs equal Ramond parity copies, but it
also glues the even and odd chiral three-point forms with a convention phase.
This audit keeps that relative phase explicit.  For every stored branch
triple and both exact momentum samples it compares the normalized local
quadratic with every reflected four-ell product.  A match is accepted only
when the quotient is exactly momentum independent between the two samples.
"""

from __future__ import annotations

import itertools
import sys
from functools import lru_cache
from pathlib import Path

import sympy as sp


HERE = Path(__file__).resolve().parent
GRID_DIR = HERE.parent
CONSTRUCTIVE = GRID_DIR / "constructive_formula"
for directory in (GRID_DIR, CONSTRUCTIVE):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import compute_grid as grid  # noqa: E402
import bell_tomography as bell  # noqa: E402


I = sp.I
PHASES = (sp.Integer(1), -sp.Integer(1), I, -I)


@lru_cache(None)
def weighted_diagonal_raw(labels, eta, sample, odd_weight):
    """Return the exact raw diagonal local contraction D_sf(odd_weight)."""

    n1, n2, n3 = map(sp.Rational, labels)
    b_value, p1, p2, p3 = sample
    masters = tuple(
        grid.enlarged_raw_three_point(
            n1, n2, n3, epsilon2, 0, 0, eta,
            b_value, p1, p2, p3,
        )[1]
        for epsilon2 in (0, 1)
    )
    barred_masters = tuple(bell.spin_frame_bar(value) for value in masters)
    m2 = bell.ramond_mode_count(n2)
    m3 = bell.ramond_mode_count(n3)
    amplitudes = bell.amplitude_matrices(*masters, m2, m3, eta)
    barred = bell.amplitude_matrices(
        *barred_masters, m2, m3, eta, barred=True
    )
    left = bell.diagonal_pairing(m2)
    right = bell.diagonal_pairing(m3)
    even = bell.local_contraction(
        left, right, amplitudes, barred, form_parity=0
    )
    odd = bell.local_contraction(
        left, right, amplitudes, barred, form_parity=1
    )
    return sp.factor(sp.cancel(even + sp.sympify(odd_weight) * odd))


@lru_cache(None)
def normalized_local(labels, eta, sample, odd_weight):
    """Divide the raw local contraction by the three raw branch norms."""

    n1, n2, n3 = map(sp.Rational, labels)
    b_value, p1, p2, p3 = sample
    raw = weighted_diagonal_raw(labels, eta, sample, odd_weight)
    norms = grid.raw_norms(
        n1, n2, n3, 0, 0, b_value, p1, p2, p3
    )
    return sp.factor(sp.cancel(raw / sp.prod(norms)))


@lru_cache(None)
def ell_square(labels, sample, sheets):
    """Return the normalized four-ell square on one reflected branch sheet.

    A branch reflection acts on the label and momentum together:
    ``(n_j,P_j)->(-n_j,-P_j)``.  Reflecting only the momentum compares
    different objects once an NS branch is excited.
    """

    n1, n2, n3 = map(sp.Rational, labels)
    b_value, p1, p2, p3 = sample
    momenta = (p1, p2, p3)
    reflected = tuple(sheet * momentum for sheet, momentum in zip(sheets, momenta))
    reflected_labels = tuple(
        sheet * label for sheet, label in zip(sheets, (n1, n2, n3))
    )
    numerator = grid.boundary.numerator_product(
        *reflected_labels, *reflected, b_value
    )
    denominator = sp.prod(
        grid.boundary.leg_product(momentum, label, b_value)
        for momentum, label in zip(momenta, labels)
    )
    return sp.factor(sp.cancel(numerator**2 / denominator))


def matches_for(labels):
    """Return every (eta, odd phase, sheet) with a constant exact quotient."""

    answer = []
    for eta, odd_weight, sheets in itertools.product(
        (1, -1), PHASES, itertools.product((1, -1), repeat=3)
    ):
        quotients = []
        for sample in grid.SAMPLES:
            direct = normalized_local(labels, eta, sample, odd_weight)
            candidate = ell_square(labels, sample, sheets)
            if direct == 0 or candidate == 0:
                break
            quotients.append(sp.factor(sp.cancel(direct / candidate)))
        if len(quotients) == 2 and sp.factor(
            sp.cancel(quotients[0] - quotients[1])
        ) == 0:
            answer.append((eta, odd_weight, sheets, quotients[0]))
    return tuple(answer)


def main():
    total = 0
    failures = []
    for labels in itertools.product(
        grid.GRID_NS_LEVELS, grid.GRID_R_LEVELS, grid.GRID_R_LEVELS
    ):
        matches = matches_for(labels)
        if not matches:
            failures.append(labels)
        print(f"labels={labels} matches={matches}", flush=True)
        total += 1
    print(f"summary triples={total} matched={total-len(failures)} failures={failures}")


if __name__ == "__main__":
    main()
