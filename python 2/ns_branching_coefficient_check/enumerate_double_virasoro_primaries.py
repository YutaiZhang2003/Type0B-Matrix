#!/usr/bin/env python3
"""Enumerate low NS double-Virasoro primary states in the f x SCA basis.

The branch label used here is ``n`` in the decomposition

    F_NS x V_NS(P) = direct_sum_(n in Z/2) V^(1)_n x V^(2)_n.

Equivalently, the Fermi-sea label in the literature is ``k=2*n``.  The
symbolic normalization agrees with SCblock through ``|n|=1``.  Negative
branches use the reflection convention ``v_-n(P)=v_n(-P)``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import sympy as sp


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import check_ns_branch_norms as ns  # noqa: E402


P, Q = ns.P, ns.Q


def positive_normalization(branch_label, momentum=P):
    """Return Omega_(2n)(P) for n=0,1/2,1 in Q,P notation."""

    branch_label = sp.Rational(branch_label)
    momentum = sp.sympify(momentum)
    if branch_label == 0:
        return sp.Integer(1)
    if branch_label == sp.Rational(1, 2):
        return Q / 2 + momentum
    if branch_label == 1:
        x = Q + 2 * momentum
        return sp.factor(x * (x + Q) * (x**2 + 2 * Q * x + 4) / 4)
    raise ValueError("Symbolic normalization is implemented through |n|=1.")


def primary_components(branch_label):
    """Return normalized ``(f modes2, L modes, G modes2, coefficient)`` terms."""

    branch_label = sp.Rational(branch_label)
    magnitude = abs(branch_label)
    if magnitude > 1 or 2 * magnitude not in sp.S.Integers:
        raise ValueError("This low-level enumerator supports n=0,+/-1/2,+/-1.")
    effective_momentum = P if branch_label >= 0 else -P
    scale = positive_normalization(magnitude, effective_momentum)
    _, sectors = ns.branch_in_abstract_basis(magnitude)
    answer = []
    for auxiliary_modes2, (_, ordered_basis, coefficients) in sectors.items():
        for (virasoro_modes, supercurrent_modes2), coefficient in zip(
            ordered_basis, coefficients
        ):
            coefficient = coefficient.subs(P, effective_momentum)
            coefficient = sp.factor(sp.cancel(scale * coefficient))
            if coefficient != 0:
                answer.append(
                    (
                        auxiliary_modes2,
                        virasoro_modes,
                        supercurrent_modes2,
                        coefficient,
                    )
                )
    return tuple(answer)


def _half_mode(mode2):
    return str(sp.Rational(mode2, 2))


def component_name(auxiliary_modes2, virasoro_modes, supercurrent_modes2):
    """Human-readable auxiliary-first tensor-product basis name."""

    auxiliary = " ".join(f"f_-{_half_mode(mode)}" for mode in auxiliary_modes2)
    physical = " ".join(f"L_-{mode}" for mode in virasoro_modes)
    if physical and supercurrent_modes2:
        physical += " "
    physical += " ".join(
        f"G_-{_half_mode(mode)}" for mode in supercurrent_modes2
    )
    auxiliary = auxiliary or "1_F"
    physical = physical or "|P>"
    return f"{auxiliary} x {physical}"


def state_rows(branch_label):
    """Return serializable rows for one normalized primary."""

    rows = []
    for auxiliary, virasoro, supercurrent, coefficient in primary_components(
        branch_label
    ):
        rows.append(
            {
                "auxiliary_modes2": auxiliary,
                "virasoro_modes": virasoro,
                "supercurrent_modes2": supercurrent,
                "coefficient": str(coefficient),
                "basis_state": component_name(auxiliary, virasoro, supercurrent),
            }
        )
    return rows


def parse_labels(value):
    return tuple(sp.Rational(item.strip()) for item in value.split(","))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels",
        default="-1,-1/2,0,1/2,1",
        help="comma-separated branch labels",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON")
    arguments = parser.parse_args()
    labels = parse_labels(arguments.labels)
    data = {str(label): state_rows(label) for label in labels}
    if arguments.json:
        print(json.dumps(data, indent=2))
        return
    for label in labels:
        print(f"v_({label})(P):")
        for row in data[str(label)]:
            print(f"  {row['coefficient']}  [{row['basis_state']}]")


if __name__ == "__main__":
    main()
