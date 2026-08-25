#!/usr/bin/env python3
"""Exact audit of the actual diagonal no-J row in normalized branches.

Suchanek's local Ramond vertex weights the physical even and odd chiral
forms by ``1`` and ``-i``.  These labels are the physical-form parity
``f``; they are not the Ramond branch-copy label ``epsilon``.  For positive
diagonal Ramond representatives the complete reduced row is

  sum_{e2,e3,f} (-i)^f d_2[e2] d_3[e3]
      N_1 N_2[e2] N_3[e3]
      B_{f,g}(e2,e3) bar(B_{f,g}(e2,e3)),

  g = 2 n_1 + e2 + e3 - f  (mod 2).

Here ``d_j`` is the diagonal local-field coefficient and ``N_j`` is the
raw branch norm.  The norm factors are mandatory: the local fields are
expanded in raw branch states, while ``B`` is defined using unit-normalized
states.  This script proves the displayed conversion against a direct sum
of all raw restrictions, then tests whether the result is a single
four-factor ell square at fixed labels.
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


def delta(label, momentum, b_value):
    q_value = b_value + 1 / b_value
    index = int(4 * sp.Rational(label))
    value = boundary.ell(q_value + 2 * momentum, index, b_value)
    if index % 2:
        value *= sp.Pow(2, -sp.Rational(1, 8))
    return sp.factor(value)


def all_restrictions(labels, eta, sample):
    """Return raw values and normalized B*bar(B) for all eight entries.

    The two masters are evaluated from states.  The other six entries use
    the exact discrete-reduction identity already audited independently on
    the complete 432-restriction grid.  This avoids repeating the expensive
    30-by-30 PBW contraction eight times at the 5/4 corner.
    """

    n1, n2, n3 = map(sp.Rational, labels)
    values = {}
    products = {}
    norms = {}
    m2 = bell.ramond_mode_count(n2)
    m3 = bell.ramond_mode_count(n3)
    r3 = sp.Pow(2, sp.Rational((-1) ** (m3 + 1), 2))
    masters = {
        epsilon2: grid.enlarged_raw_three_point(
            n1, n2, n3, epsilon2, 0, 0, eta, *sample
        )[1]
        for epsilon2 in (0, 1)
    }
    for epsilon2, epsilon3, form_parity in itertools.product((0, 1), repeat=3):
        key = (epsilon2, epsilon3, form_parity)
        spin_phase = (
            eta * (-1) ** (m2 + 1 + epsilon2) * grid.EIGHTH_MINUS
        )
        raw = sp.factor(
            masters[epsilon2]
            * r3**epsilon3
            * (-1) ** (epsilon3 * form_parity)
            * spin_phase**form_parity
        )
        branch_norms = grid.raw_norms(
            n1, n2, n3, epsilon2, epsilon3, *sample
        )
        total_norm = sp.prod(branch_norms)
        values[key] = raw
        norms[key] = total_norm
        products[key] = sp.factor(
            sp.cancel(raw * bell.spin_frame_bar(raw) / total_norm)
        )
    return values, products, norms


def normalized_row(labels, eta, sample):
    """Evaluate the local row twice and return the exact residual."""

    n1, n2, n3 = map(sp.Rational, labels)
    values, products, norms = all_restrictions(labels, eta, sample)
    m2 = bell.ramond_mode_count(n2)
    m3 = bell.ramond_mode_count(n3)
    d2 = tuple(bell.diagonal_pairing(m2)[e, e] for e in (0, 1))
    d3 = tuple(bell.diagonal_pairing(m3)[e, e] for e in (0, 1))

    raw_sum = 0
    normalized_sum = 0
    for epsilon2, epsilon3, form_parity in itertools.product((0, 1), repeat=3):
        key = (epsilon2, epsilon3, form_parity)
        coefficient = (-I) ** form_parity * d2[epsilon2] * d3[epsilon3]
        raw_sum += coefficient * values[key] * bell.spin_frame_bar(values[key])
        normalized_sum += coefficient * norms[key] * products[key]

    raw_sum = sp.factor(sp.cancel(raw_sum))
    normalized_sum = sp.factor(sp.cancel(normalized_sum))
    residual = sp.factor(sp.cancel(raw_sum - normalized_sum))
    return raw_sum, residual


def scalar_square(labels, sample, signs):
    b_value, p1, p2, p3 = sample
    signed_labels = tuple(sign * label for sign, label in zip(signs, labels))
    signed_momenta = tuple(
        sign * momentum
        for sign, momentum in zip(signs, (p1, p2, p3))
    )
    return boundary.numerator_product(
        *signed_labels, *signed_momenta, b_value
    ) ** 2


def audit():
    row_checks = 0
    direct_passes = 0
    any_passes = 0
    failures = []
    for labels in itertools.product(
        grid.GRID_NS_LEVELS, grid.GRID_R_LEVELS, grid.GRID_R_LEVELS
    ):
        for eta in (1, -1):
            cleared_values = []
            for sample in grid.SAMPLES:
                value, residual = normalized_row(labels, eta, sample)
                if residual != 0:
                    raise AssertionError((labels, eta, sample, residual))
                b_value, p1, p2, p3 = sample
                clearance = sp.prod(
                    delta(label, momentum, b_value) ** 2
                    for label, momentum in zip(labels, (p1, p2, p3))
                )
                cleared_values.append(sp.factor(sp.cancel(value * clearance)))
                row_checks += 1

            matches = []
            for signs in SIGNS:
                ratios = [
                    sp.factor(
                        sp.cancel(
                            value / scalar_square(labels, sample, signs)
                        )
                    )
                    for value, sample in zip(cleared_values, grid.SAMPLES)
                ]
                if sp.factor(sp.cancel(ratios[0] - ratios[1])) == 0:
                    matches.append((signs, ratios[0]))
            direct_passes += int(
                any(signs == (1, 1, 1) for signs, _ in matches)
            )
            any_passes += int(bool(matches))
            if not matches and len(failures) < 8:
                failures.append((labels, eta))
        print(f"normalized no-J row labels={labels} complete", flush=True)

    print(
        f"normalized-row conversion: {row_checks}/108 exact residuals zero; "
        "all 864 stored restrictions used"
    )
    print(
        "single-ell-square audit: "
        f"direct chamber={direct_passes}/54; "
        f"any jointly reflected chamber={any_passes}/54"
    )
    print(f"first no-chamber failures: {failures}")


if __name__ == "__main__":
    audit()
