#!/usr/bin/env python3
"""Audit the proposed scalar no-J row against the complete stored grid.

The literal claim under test is

    D_red^eta = lambda * (B_0^2 + i B_1^2)

with ``B_epsilon`` the two normalized ``f=0, epsilon_3=0`` branching
masters and

    lambda = i eta (-1)^(2 n_1 + M_2 + M_3 + 1).

If this row were the scalar diagonal double-Liouville shift factor, then at
fixed branch labels its quotient by

    C^2 = P(n_1,n_2,n_3)^2 / prod_j leg_j

would be independent of ``b,P_1,P_2,P_3``.  We test this exactly at the two
stored rational samples.  Besides the unreflected chamber, the audit checks
all eight simultaneous branch reflections ``(n_j,P_j)->(-n_j,-P_j)``.  A
reflection is required to act on the label and momentum together.

The calculation uses every one of the 108 independent master functions at
both samples.  The twelve remaining restrictions at each level triple are
the already certified universal phase copies, so every fixed-row result has
four identical copies among the 432 restrictions.
"""

from __future__ import annotations

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
SIGNS = tuple(itertools.product((1, -1), repeat=3))


def branching_square(labels, epsilon2, eta, sample):
    """Normalized B_epsilon^2 without evaluating any ell ansatz."""

    n1, n2, n3 = labels
    b_value, p1, p2, p3 = sample
    _, raw = grid.enlarged_raw_three_point(
        n1, n2, n3, epsilon2, 0, 0, eta,
        b_value, p1, p2, p3,
    )
    norms = grid.raw_norms(
        n1, n2, n3, epsilon2, 0,
        b_value, p1, p2, p3,
    )
    return sp.factor(sp.cancel(raw**2 / sp.prod(norms)))


def d_red(labels, eta, sample):
    """The literal normalized no-J row stated in the current discussion."""

    b0 = branching_square(labels, 0, eta, sample)
    b1 = branching_square(labels, 1, eta, sample)
    lam = bell.spin_frame_phase(*labels, eta)
    return sp.factor(sp.cancel(lam * (b0 + I * b1)))


def scalar_dl_shift_square(labels, sample, signs=(1, 1, 1)):
    """The NS-like ell shift square in one reflected branch chamber."""

    b_value, p1, p2, p3 = sample
    signed_labels = tuple(sign * label for sign, label in zip(signs, labels))
    signed_momenta = tuple(
        sign * momentum
        for sign, momentum in zip(signs, (p1, p2, p3))
    )
    numerator = boundary.numerator_product(
        *signed_labels, *signed_momenta, b_value
    )
    legs = sp.prod(
        boundary.leg_product(momentum, label, b_value)
        for label, momentum in zip(labels, (p1, p2, p3))
    )
    return sp.factor(sp.cancel(numerator**2 / legs))


def audit():
    total_rows = 0
    direct_chamber_passes = 0
    any_chamber_passes = 0
    per_eta = {1: [0, 0], -1: [0, 0]}
    first_failures = []

    for labels in itertools.product(
        grid.GRID_NS_LEVELS, grid.GRID_R_LEVELS, grid.GRID_R_LEVELS
    ):
        for eta in (1, -1):
            values = [d_red(labels, eta, sample) for sample in grid.SAMPLES]
            matches = []
            for signs in SIGNS:
                ratios = []
                for sample, value in zip(grid.SAMPLES, values):
                    candidate = scalar_dl_shift_square(labels, sample, signs)
                    if candidate == 0:
                        break
                    ratios.append(sp.factor(sp.cancel(value / candidate)))
                if len(ratios) == 2 and sp.factor(
                    sp.cancel(ratios[0] - ratios[1])
                ) == 0:
                    matches.append((signs, ratios[0]))

            direct = any(signs == (1, 1, 1) for signs, _ in matches)
            any_match = bool(matches)
            direct_chamber_passes += int(direct)
            any_chamber_passes += int(any_match)
            per_eta[eta][0] += int(direct)
            per_eta[eta][1] += int(any_match)
            total_rows += 1
            if not any_match and len(first_failures) < 8:
                first_failures.append((labels, eta))
        print(f"D_red audit labels={labels} complete", flush=True)

    print(
        "D_red exact audit: "
        f"direct chamber {direct_chamber_passes}/{total_rows}; "
        f"any jointly reflected chamber {any_chamber_passes}/{total_rows}"
    )
    print(f"per eta [direct,any]: {per_eta}")
    print(f"first no-chamber failures: {first_failures}")
    print(
        "state values used: 108 masters x 2 samples = 216; "
        "universal copies represented: 432 restrictions x 2 samples = 864"
    )
    print(
        "restriction-copy counts: "
        f"direct={4 * direct_chamber_passes}/216 rows, "
        f"any={4 * any_chamber_passes}/216 rows"
    )


if __name__ == "__main__":
    audit()
