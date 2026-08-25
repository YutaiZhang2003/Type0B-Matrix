#!/usr/bin/env python3
"""Show exactly where the raw endpoint-Z identity stops being an SCA identity.

For the consecutive Ramond strings the *formal Fock-path coefficients* obey

    path(-n,P) = (-1)**physical_ground path(+n,-P).

This does not identify the corresponding abstract SCA states.  The two
paths must be converted with different free-field transition matrices.  At
``n=3/4`` their first discrepancy is in the ``L_-1`` components.  This is
the finite-level version of the warning in arXiv:1312.4520: a reflected
fermion is a momentum-dependent mixture of bosonic and fermionic
oscillators, not a two-dimensional ground operation.

The final check applies the endpoint-only proposal on its first signed
zero-screening plane.  It disagrees with the independent SCA three-form by
a nonzero exact rational number.  Ward data occur only in this audit; the
positive native screening evaluator imports none.
"""

from __future__ import annotations

from collections import defaultdict

import sympy as sp

from python.nsrr_chi_branching import nsrr_chi_formula as chi
from python.ramond_three_point_grid import compute_grid as grid

from ..pfaffian.native_hard_screening import (
    hard_neutrality_momentum,
    hard_screening_value,
)


I = sp.I
SQRT2 = sp.sqrt(2)
HARD = sp.Rational(3, 4)
FOCK_TO_SCBLOCK_MINUS = -(1 - I) / SQRT2


def _component_map(components):
    answer = defaultdict(lambda: sp.Integer(0))
    for auxiliary, auxiliary_ground, word, ground, coefficient in components:
        answer[(auxiliary, auxiliary_ground, word, ground)] += coefficient
    return {
        key: sp.cancel(value)
        for key, value in answer.items()
        if sp.cancel(value) != 0
    }


def endpoint_z_positive_components(parity, q, momentum):
    """Convert endpoint-Z paths of ``W_+3/4(-P)`` to the SCA basis."""

    grouped = defaultdict(lambda: defaultdict(lambda: sp.Integer(0)))
    for state, coefficient in chi.ramond_fock_paths(HARD, int(parity)):
        auxiliary, auxiliary_ground, physical, physical_ground = state
        grouped[(auxiliary, auxiliary_ground)][
            ((), physical, physical_ground)
        ] += (-1) ** int(physical_ground) * coefficient

    substitutions = {
        grid.ramond_branch.Q: q,
        grid.ramond_branch.P: -momentum,
    }
    answer = []
    for auxiliary_state, expression in grouped.items():
        sample_state = next(iter(expression))
        level = sum(sample_state[0]) + sum(sample_state[1])
        basis, transition = grid.ramond_branch.transition(level, -1)
        transition = transition.subs(substitutions, simultaneous=True)
        row = {state: index for index, state in enumerate(basis)}
        vector = sp.zeros(len(basis), 1)
        for state, coefficient in expression.items():
            vector[row[state]] += coefficient
        coefficients = transition.inv() * vector
        for index, (virasoro, supercurrent, ground) in enumerate(basis):
            coefficient = sp.cancel(coefficients[index])
            if ground == 1:
                coefficient *= FOCK_TO_SCBLOCK_MINUS
            coefficient = sp.cancel(coefficient)
            if coefficient:
                word = tuple(("L", -mode) for mode in virasoro) + tuple(
                    ("G", -mode) for mode in supercurrent
                )
                answer.append(
                    (
                        auxiliary_state[0],
                        auxiliary_state[1],
                        word,
                        ground,
                        coefficient,
                    )
                )
    return tuple(answer)


def abstract_state_residuals(parity, q, momentum):
    """Return ``W_-3/4(P) - Z_path W_+3/4(-P)`` componentwise."""

    negative = _component_map(
        chi.ramond_path_components(-HARD, int(parity), q, momentum)
    )
    endpoint = _component_map(
        endpoint_z_positive_components(int(parity), q, momentum)
    )
    keys = set(negative) | set(endpoint)
    return {
        key: sp.factor(sp.cancel(negative.get(key, 0) - endpoint.get(key, 0)))
        for key in keys
        if sp.factor(
            sp.cancel(negative.get(key, 0) - endpoint.get(key, 0))
        )
        != 0
    }


def audit_raw_paths():
    checked = 0
    for parity in (0, 1):
        negative = dict(chi.ramond_fock_paths(-HARD, parity))
        positive = dict(chi.ramond_fock_paths(HARD, parity))
        if set(negative) != set(positive):
            raise AssertionError((parity, "path support"))
        for state, coefficient in negative.items():
            residual = sp.simplify(
                coefficient - (-1) ** int(state[3]) * positive[state]
            )
            if residual != 0:
                raise AssertionError((parity, state, residual))
            checked += 1
    return checked


def audit_abstract_obstruction():
    q, momentum = sp.symbols("Q P")
    denominator = 4 * momentum**2 - 6 * momentum * q + 2 * q**2 + 1
    expected = {
        0: {
            ((), 1, (("L", -1),), 1):
                2 * SQRT2 * (-1 - I) / denominator,
            ((), 0, (("L", -1),), 0): 4 / denominator,
        },
        1: {
            ((), 0, (("L", -1),), 1): 4 * (1 + I) / denominator,
            ((), 1, (("L", -1),), 0): 4 * SQRT2 / denominator,
        },
    }
    for parity in (0, 1):
        calculated = abstract_state_residuals(parity, q, momentum)
        if set(calculated) != set(expected[parity]):
            raise AssertionError((parity, calculated))
        for key, value in expected[parity].items():
            if sp.factor(sp.cancel(calculated[key] - value)) != 0:
                raise AssertionError((parity, key, calculated[key], value))
    return expected


def audit_first_signed_value():
    b = sp.Rational(3, 2)
    p2 = sp.Rational(2, 5)
    p3 = sp.Rational(3, 7)
    p1 = hard_neutrality_momentum(b, p2, -p3, 0)
    expected_residuals = {
        0: (-1 + I) / 8,
        1: I * SQRT2 / 8,
    }
    for form_parity in (0, 1):
        endpoint_only = hard_screening_value(
            0,
            form_parity,
            1,
            b,
            p2,
            -p3,
            right_endpoint_z=True,
        )
        actual = grid.enlarged_raw_three_point(
            0,
            HARD,
            -HARD,
            0,
            0,
            form_parity,
            1,
            b,
            p1,
            p2,
            p3,
        )[1]
        residual = sp.factor(sp.cancel(endpoint_only - actual))
        if residual != expected_residuals[form_parity]:
            raise AssertionError((form_parity, residual))
    return p1, expected_residuals


def audit():
    checked = audit_raw_paths()
    residuals = audit_abstract_obstruction()
    p1, value_residuals = audit_first_signed_value()
    print(f"raw endpoint-Z path coefficients: {checked} exact checks passed")
    for parity in (0, 1):
        print(f"epsilon={parity}: abstract L_-1 residuals={residuals[parity]}")
    print(
        "first signed zero-screening plane: "
        f"P1={p1}, endpoint-only residuals={value_residuals}"
    )
    print("conclusion: reflected SCA operator required; endpoint Z is not a value callback")


if __name__ == "__main__":
    audit()
