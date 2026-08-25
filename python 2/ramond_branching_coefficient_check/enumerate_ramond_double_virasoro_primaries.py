#!/usr/bin/env python3
"""Enumerate low Ramond double-Virasoro primaries in the native ground basis.

The symbolic default covers n=+/-1/4,+/-3/4 and both parity copies.  Labels
of larger magnitude, including n=+/-5/4, require exact ``--q`` and
``--momentum`` values so that the Fock-to-PBW transition is inverted only
after specialization.
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

import check_ramond_branching as branch  # noqa: E402


def primary_components(branch_label, parity, q_value=None, momentum=None):
    """Return native ``f x Ramond-PBW`` components of one highest state."""

    branch_label = sp.Rational(branch_label)
    parity = int(parity)
    if parity not in (0, 1):
        raise ValueError("The Ramond parity copy must be 0 or 1.")
    if (q_value is None) != (momentum is None):
        raise ValueError("q_value and momentum must be supplied together.")
    if abs(branch_label) > sp.Rational(3, 4) and q_value is None:
        raise ValueError(
            "Ramond labels beyond |n|=3/4 require exact Q and P values."
        )
    substitutions = None
    if q_value is not None:
        substitutions = {
            branch.Q: sp.sympify(q_value),
            branch.P: sp.sympify(momentum),
        }
    operators, sectors = branch.branch_in_abstract_basis(
        branch_label, parity, substitutions=substitutions
    )
    answer = []
    for (auxiliary_modes, auxiliary_ground), (
        level,
        ordered_basis,
        coefficients,
    ) in sectors.items():
        for (
            virasoro_modes,
            supercurrent_modes,
            physical_ground,
        ), coefficient in zip(ordered_basis, coefficients):
            coefficient = sp.factor(sp.cancel(coefficient))
            if coefficient != 0:
                answer.append(
                    (
                        auxiliary_modes,
                        auxiliary_ground,
                        virasoro_modes,
                        supercurrent_modes,
                        physical_ground,
                        coefficient,
                    )
                )
    return tuple(operators), tuple(answer)


def component_name(
    auxiliary_modes,
    auxiliary_ground,
    virasoro_modes,
    supercurrent_modes,
    physical_ground,
):
    """Human-readable name in the native free-field ground convention."""

    auxiliary = " ".join(f"f_-{mode}" for mode in auxiliary_modes)
    physical = " ".join(f"L_-{mode}" for mode in virasoro_modes)
    if physical and supercurrent_modes:
        physical += " "
    physical += " ".join(f"G_-{mode}" for mode in supercurrent_modes)
    auxiliary = (auxiliary + " " if auxiliary else "") + f"|{auxiliary_ground}>_F"
    physical = (physical + " " if physical else "") + f"|P,{physical_ground}>_R"
    return f"{auxiliary} x {physical}"


def state_rows(branch_label, parity, q_value=None, momentum=None):
    operators, components = primary_components(
        branch_label, parity, q_value=q_value, momentum=momentum
    )
    rows = []
    for component in components:
        auxiliary, auxiliary_ground, virasoro, supercurrent, ground, coefficient = (
            component
        )
        rows.append(
            {
                "auxiliary_modes": auxiliary,
                "auxiliary_ground": auxiliary_ground,
                "virasoro_modes": virasoro,
                "supercurrent_modes": supercurrent,
                "physical_ground": ground,
                "coefficient": str(coefficient),
                "basis_state": component_name(
                    auxiliary,
                    auxiliary_ground,
                    virasoro,
                    supercurrent,
                    ground,
                ),
            }
        )
    return {"chi_operators": operators, "components": rows}


def parse_labels(value):
    return tuple(sp.Rational(item.strip()) for item in value.split(","))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels",
        default="-3/4,-1/4,1/4,3/4",
        help="comma-separated Ramond branch labels",
    )
    parser.add_argument(
        "--parities", default="0,1", help="comma-separated parity copies"
    )
    parser.add_argument("--q", type=sp.Rational)
    parser.add_argument("--momentum", type=sp.Rational)
    parser.add_argument("--json", action="store_true")
    arguments = parser.parse_args()
    labels = parse_labels(arguments.labels)
    parities = tuple(int(value) for value in arguments.parities.split(","))
    data = {
        f"n={label},parity={parity}": state_rows(
            label,
            parity,
            q_value=arguments.q,
            momentum=arguments.momentum,
        )
        for label in labels
        for parity in parities
    }
    if arguments.json:
        print(json.dumps(data, indent=2))
        return
    for key, state in data.items():
        print(f"{key}, chi operators={state['chi_operators']}:")
        for row in state["components"]:
            print(f"  {row['coefficient']}  [{row['basis_state']}]")


if __name__ == "__main__":
    main()
