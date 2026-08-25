#!/usr/bin/env python3
"""Constructive finite-path formula for every NS--R--R raw master.

The scalar NS blow-up factor stops being the correct chiral object when two
Ramond strings are excited.  The replacement is a finite two-colour path
sum.  This file implements that sum independently of
``compute_grid.enlarged_raw_three_point``:

* every branch operator is expanded by the sparse update

      X_r = A_r tensor 1 - i (-1)^F_A tensor Theta_r;

* the physical Fock endpoint is transported to the abstract SCA PBW basis;
* the auxiliary endpoints are contracted directly with the two-spin-field
  Pfaffian; and
* the physical endpoints are contracted with the NS--R--R Ward recursion.

If D_j=ell(Q+2P_j,4n_j), the transported endpoint columns may equivalently
be replaced by D_j*T_j^{-1}e.  Thus the result has the explicit form

      R_epsilon^eta = PathPolynomial_epsilon^eta/(D_1 D_2 D_3).

The path polynomial is finite at every fixed set of branch labels.  Unlike a
single ell product, it retains paths in which the two excited Ramond strings
contract with one another.  ``--audit`` compares this construction with the
original state calculation on all 108 independent masters.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from functools import lru_cache
import itertools
from pathlib import Path
import sys
import time

import sympy as sp


THIS_DIR = Path(__file__).resolve().parent
GRID_DIR = THIS_DIR.parent
if str(GRID_DIR) not in sys.path:
    sys.path.insert(0, str(GRID_DIR))

import compute_grid as grid  # noqa: E402


I = sp.I
SQRT2 = sp.sqrt(2)
EIGHTH_MINUS = (1 - I) / SQRT2
MASTER_CHOICES = ((0, 1), (0, -1), (1, 1), (1, -1))


def _add_term(expression, state, coefficient):
    coefficient = sp.cancel(coefficient)
    if coefficient == 0:
        return
    expression[state] = sp.cancel(expression.get(state, 0) + coefficient)
    if expression[state] == 0:
        del expression[state]


def _fermion_action(mode, modes, ground, zero_sign=1):
    """Apply one creator/annihilator/zero mode to a Fock endpoint."""

    mode = int(mode)
    if mode < 0:
        created = -mode
        if created in modes:
            return None, sp.Integer(0)
        crossings = sum(existing > created for existing in modes)
        final = tuple(sorted(modes + (created,), reverse=True))
        return (final, ground), sp.Integer((-1) ** crossings)
    if mode == 0:
        coefficient = (-1) ** len(modes) * sp.Rational(zero_sign, 1) / SQRT2
        return (modes, 1 - ground), coefficient
    if mode not in modes:
        return None, sp.Integer(0)
    position = modes.index(mode)
    return (modes[:position] + modes[position + 1 :], ground), sp.Integer(
        (-1) ** position
    )


def ramond_fock_paths(branch_label, parity):
    """Expand the ordered Ramond chi string by its binary path recurrence.

    A state key is ``(aux_modes,aux_ground,physical_modes,physical_ground)``.
    At every mode the update has exactly two possible colours: auxiliary or
    physical.  The second zero mode needed for the opposite parity copy uses
    the reflected physical realization and hence ``zero_sign=-1``.
    """

    branch_label = sp.Rational(branch_label)
    if branch_label <= 0:
        raise ValueError("This path formula uses the positive Ramond branch.")
    largest_mode = int(2 * branch_label - sp.Rational(1, 2))
    operators = [0] + [-mode for mode in range(1, largest_mode + 1)]
    raw_parity = len(operators) % 2
    appended_opposite_zero = raw_parity != int(parity)
    if appended_opposite_zero:
        operators.append(0)

    expression = {((), 0, (), 0): sp.Integer(1)}
    indexed_operators = tuple(enumerate(operators))
    for position, mode in reversed(indexed_operators):
        next_expression = {}
        for state, outer in expression.items():
            aux_modes, aux_ground, physical_modes, physical_ground = state

            aux_final, aux_coefficient = _fermion_action(
                mode, aux_modes, aux_ground
            )
            if aux_coefficient:
                _add_term(
                    next_expression,
                    (
                        aux_final[0],
                        aux_final[1],
                        physical_modes,
                        physical_ground,
                    ),
                    outer * aux_coefficient,
                )

            opposite = (
                appended_opposite_zero
                and mode == 0
                and position == len(operators) - 1
            )
            physical_final, physical_coefficient = _fermion_action(
                mode,
                physical_modes,
                physical_ground,
                zero_sign=-1 if opposite else 1,
            )
            if physical_coefficient:
                auxiliary_parity = (len(aux_modes) + aux_ground) % 2
                _add_term(
                    next_expression,
                    (
                        aux_modes,
                        aux_ground,
                        physical_final[0],
                        physical_final[1],
                    ),
                    outer
                    * (-I)
                    * (-1) ** auxiliary_parity
                    * physical_coefficient,
                )
        expression = next_expression
    return tuple(expression.items())


@lru_cache(None)
def ramond_path_components(branch_label, parity, q_value, momentum):
    """Transport every Ramond physical path endpoint through T_L(P)^(-1)."""

    by_auxiliary = defaultdict(lambda: defaultdict(lambda: sp.Integer(0)))
    for state, coefficient in ramond_fock_paths(branch_label, parity):
        aux_modes, aux_ground, physical_modes, physical_ground = state
        by_auxiliary[(aux_modes, aux_ground)][
            ((), physical_modes, physical_ground)
        ] += coefficient

    answer = []
    substitutions = {
        grid.ramond_branch.Q: q_value,
        grid.ramond_branch.P: momentum,
    }
    for auxiliary_state, physical_expression in by_auxiliary.items():
        one_state = next(iter(physical_expression))
        level = sum(one_state[0]) + sum(one_state[1])
        ordered_basis, transition = grid.ramond_branch.transition(level, -1)
        transition = transition.subs(substitutions, simultaneous=True)
        vector = sp.zeros(len(ordered_basis), 1)
        row = {state: index for index, state in enumerate(ordered_basis)}
        for state, coefficient in physical_expression.items():
            vector[row[state]] += coefficient
        coefficients = transition.inv() * vector
        for index, (virasoro_modes, supercurrent_modes, ground) in enumerate(
            ordered_basis
        ):
            coefficient = sp.cancel(coefficients[index])
            if ground == 1:
                coefficient *= -EIGHTH_MINUS
            coefficient = sp.cancel(coefficient)
            if coefficient == 0:
                continue
            word = tuple(("L", -mode) for mode in virasoro_modes) + tuple(
                ("G", -mode) for mode in supercurrent_modes
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


@lru_cache(None)
def ns_path_components(branch_label, q_value, momentum):
    """NS binary path sum, with its physical endpoints transported to PBW."""

    branch_label = sp.Rational(branch_label)
    if branch_label == 0:
        return (((), (), sp.Integer(1)),)
    all_modes2 = tuple(range(int(4 * branch_label - 1), 0, -2))
    answer = []
    for physical_count in range(len(all_modes2) + 1):
        for physical_modes2 in itertools.combinations(
            all_modes2, physical_count
        ):
            physical_modes2 = tuple(sorted(physical_modes2, reverse=True))
            physical_set = set(physical_modes2)
            auxiliary_modes2 = tuple(
                mode for mode in all_modes2 if mode not in physical_set
            )
            crossings = sum(
                physical > auxiliary
                for physical in physical_modes2
                for auxiliary in auxiliary_modes2
            )
            path_coefficient = (-I) ** len(physical_modes2) * (-1) ** crossings
            ordered_basis, coefficients = (
                grid.ns_branch.abstract_eta_coefficients(physical_modes2)
            )
            coefficients = coefficients.subs(
                {grid.ns_branch.Q: q_value, grid.ns_branch.P: momentum},
                simultaneous=True,
            )
            for index, (virasoro_modes, supercurrent_modes2) in enumerate(
                ordered_basis
            ):
                coefficient = sp.cancel(path_coefficient * coefficients[index])
                if coefficient == 0:
                    continue
                word = tuple(("L", -sp.Integer(mode)) for mode in virasoro_modes)
                word += tuple(
                    ("G", -sp.Rational(mode2, 2))
                    for mode2 in supercurrent_modes2
                )
                answer.append(
                    (
                        tuple(sp.Rational(mode2, 2) for mode2 in auxiliary_modes2),
                        word,
                        coefficient,
                    )
                )
    return tuple(answer)


def raw_path_master(labels, epsilon2, eta, sample):
    """Evaluate R_epsilon2^eta from the explicit finite path formula."""

    n1, n2, n3 = map(sp.Rational, labels)
    b_value, p1, p2, p3 = sample
    q_value = sp.cancel(b_value + 1 / b_value)
    central_charge = sp.Rational(3, 2) + 3 * q_value**2
    h1 = (q_value**2 / 4 - p1**2) / 2
    h2 = sp.Rational(1, 16) + q_value**2 / 8 - p2**2 / 2
    h3 = sp.Rational(1, 16) + q_value**2 / 8 - p3**2 / 2
    evaluator = grid.PhysicalNRREvaluator(
        0,
        eta,
        h1,
        h2,
        h3,
        p2 / SQRT2,
        p3 / SQRT2,
        central_charge,
    )
    first = ns_path_components(n1, q_value, p1)
    second = ramond_path_components(n2, epsilon2, q_value, p2)
    third = ramond_path_components(n3, 0, q_value, p3)
    auxiliary_form_parity = (int(2 * n1) + int(epsilon2)) % 2

    algebraic_answer = [sp.Integer(0)] * 4
    for auxiliary1, word1, coefficient1 in first:
        physical_parity1 = grid.state_parity(word1)
        for (
            auxiliary2,
            auxiliary_ground2,
            word2,
            physical_ground2,
            coefficient2,
        ) in second:
            physical_parity2 = grid.state_parity(word2, physical_ground2)
            auxiliary_parity2 = (
                len(auxiliary2) + auxiliary_ground2
            ) % 2
            for (
                auxiliary3,
                auxiliary_ground3,
                word3,
                physical_ground3,
                coefficient3,
            ) in third:
                auxiliary_parity3 = (
                    len(auxiliary3) + auxiliary_ground3
                ) % 2
                # Use the Ising-Virasoro reduction, which is the exact
                # all-level definition of the auxiliary three-point form.
                # ``fermion_value`` is a low-mode contour-kernel helper;
                # its present local-coordinate expansion is not valid once
                # a mode 2 occurs on the middle Ramond leg (first reached at
                # n_2=5/4).
                auxiliary = grid.fermion_value_virasoro(
                    auxiliary_form_parity,
                    auxiliary1,
                    auxiliary2,
                    auxiliary_ground2,
                    auxiliary3,
                    auxiliary_ground3,
                )
                if auxiliary == 0:
                    continue
                physical = evaluator.value(
                    word1,
                    word2,
                    physical_ground2,
                    word3,
                    physical_ground3,
                )
                if physical == 0:
                    continue
                tensor_sign = (-1) ** (
                    physical_parity1 * (auxiliary_parity2 + auxiliary_parity3)
                    + physical_parity2 * auxiliary_parity3
                )
                term = (
                    tensor_sign
                    * coefficient1
                    * coefficient2
                    * coefficient3
                    * auxiliary
                    * physical
                )
                for index, component in enumerate(
                    grid.quadratic_number_components(term)
                ):
                    algebraic_answer[index] += component
    answer = (
        algebraic_answer[0]
        + algebraic_answer[1] * SQRT2
        + I * algebraic_answer[2]
        + I * SQRT2 * algebraic_answer[3]
    )
    return sp.factor(sp.cancel(answer))


def leg_denominator(labels, sample):
    b_value, p1, p2, p3 = sample
    q_value = sp.cancel(b_value + 1 / b_value)
    return sp.prod(
        grid.boundary.ell(q_value + 2 * momentum, int(4 * label), b_value)
        for label, momentum in zip(labels, (p1, p2, p3))
    )


def cleared_path_master(labels, epsilon2, eta, sample):
    return sp.factor(
        sp.cancel(
            raw_path_master(labels, epsilon2, eta, sample)
            * leg_denominator(labels, sample)
        )
    )


def audit(sample_count=1):
    """Compare the independent path formula with all 108 direct masters."""

    began = time.perf_counter()
    checked = 0
    for sample_number, sample in enumerate(grid.SAMPLES[:sample_count], start=1):
        for triple_number, labels in enumerate(
            itertools.product(
                grid.GRID_NS_LEVELS,
                grid.GRID_R_LEVELS,
                grid.GRID_R_LEVELS,
            ),
            start=1,
        ):
            for epsilon2, eta in MASTER_CHOICES:
                calculated = raw_path_master(labels, epsilon2, eta, sample)
                direct = grid.enlarged_raw_three_point(
                    *labels, epsilon2, 0, 0, eta, *sample
                )[1]
                residual = sp.factor(sp.cancel(calculated - direct))
                if residual != 0:
                    raise AssertionError(
                        "Path formula mismatch at "
                        f"sample={sample_number}, labels={labels}, "
                        f"(epsilon2,eta)={(epsilon2, eta)}: {residual}"
                    )
                checked += 1
            print(
                f"path audit sample={sample_number}/{sample_count} "
                f"triple={triple_number:02d}/27 labels={labels} "
                f"residuals=0 elapsed={time.perf_counter()-began:.1f}s",
                flush=True,
            )
    expected = 108 * sample_count
    if checked != expected:
        raise AssertionError(f"Expected {expected} masters, checked {checked}.")
    print(
        f"path audit complete: {checked} exact master residuals are zero; "
        f"elapsed={time.perf_counter()-began:.1f}s"
    )


def small_self_check():
    """Fast exact checks including the first crossed Ramond case."""

    sample = grid.SAMPLES[0]
    cases = (
        (sp.Integer(0), sp.Rational(1, 4), sp.Rational(1, 4)),
        (sp.Integer(0), sp.Rational(3, 4), sp.Rational(3, 4)),
        (sp.Rational(1, 2), sp.Rational(3, 4), sp.Rational(3, 4)),
    )
    for labels in cases:
        for epsilon2, eta in MASTER_CHOICES:
            calculated = raw_path_master(labels, epsilon2, eta, sample)
            direct = grid.enlarged_raw_three_point(
                *labels, epsilon2, 0, 0, eta, *sample
            )[1]
            residual = sp.factor(sp.cancel(calculated - direct))
            if residual != 0:
                raise AssertionError((labels, epsilon2, eta, residual))
    print("small path check: 12 exact master residuals are zero")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit",
        action="store_true",
        help="compare all 108 masters at exact rational momentum data",
    )
    parser.add_argument(
        "--samples",
        type=int,
        choices=(1, 2),
        default=1,
        help="number of exact samples used by --audit (default: 1)",
    )
    arguments = parser.parse_args()
    if arguments.audit:
        audit(arguments.samples)
    else:
        small_self_check()


if __name__ == "__main__":
    main()
