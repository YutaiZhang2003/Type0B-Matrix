#!/usr/bin/env python3
"""Decompose physical Ramond L_1 images into lower branching primaries.

This file treats ``L_1`` as ``1_F x L_1^SCA``.  The positive modes of the
two embedded Virasoro algebras separately annihilate every branching
primary; the physical SCA mode is a different operator.

At the first nontrivial Ramond step, L_1 maps n=+/-3/4 to the opposite
n=-/+1/4 primary in the same parity copy.  The calculation is performed
directly in the native four-ground-state basis.  The optional n=5/4
calculation constructs the first genuine level-two Vir x Vir descendant
reduction in a common free-field basis.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import product
from pathlib import Path
import sys

import sympy as sp


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import check_ramond_branching as branch  # noqa: E402
import enumerate_ramond_double_virasoro_primaries as primaries  # noqa: E402


SQRT2 = sp.sqrt(2)
I = sp.I


def add_term(expression, state, coefficient):
    coefficient = sp.factor(sp.cancel(coefficient))
    if coefficient == 0:
        return
    expression[state] = sp.factor(
        sp.cancel(expression.get(state, 0) + coefficient)
    )
    if expression[state] == 0:
        del expression[state]


def _add_product_term(expression, state, coefficient):
    """Add one coefficient in the auxiliary x physical Fock basis."""

    add_term(expression, state, coefficient)


def _split_product_state(state):
    auxiliary_modes, auxiliary_ground, bosons, fermions, physical_ground = state
    return (
        (auxiliary_modes, auxiliary_ground),
        (bosons, fermions, physical_ground),
    )


def _join_product_state(auxiliary, physical):
    return auxiliary[0], auxiliary[1], physical[0], physical[1], physical[2]


def physical_l1_image(branch_label, parity):
    """Return the ground-basis image of ``1_F x L_1`` at |n|=3/4."""

    branch_label = sp.Rational(branch_label)
    if abs(branch_label) != sp.Rational(3, 4):
        raise ValueError("This first-step decomposition expects |n|=3/4.")
    _, components = primaries.primary_components(branch_label, parity)
    h_ramond = (
        sp.Rational(1, 16) + branch.Q**2 / 8 - branch.P**2 / 2
    )
    answer = defaultdict(lambda: sp.Integer(0))
    for (
        auxiliary_modes,
        auxiliary_ground,
        virasoro_modes,
        supercurrent_modes,
        physical_ground,
        outer,
    ) in components:
        if not virasoro_modes and not supercurrent_modes:
            continue
        if virasoro_modes == (1,) and not supercurrent_modes:
            add_term(
                answer,
                (auxiliary_modes, auxiliary_ground, physical_ground),
                outer * 2 * h_ramond,
            )
            continue
        if not virasoro_modes and supercurrent_modes == (1,):
            add_term(
                answer,
                (auxiliary_modes, auxiliary_ground, 1 - physical_ground),
                outer * sp.Rational(3, 2) * (-I * branch.P / SQRT2),
            )
            continue
        raise AssertionError(
            "The |n|=3/4 state contains an unexpected physical descendant."
        )
    return dict(answer)


def onset_vector(branch_label, parity):
    """Return one n=+/-1/4 primary as a native ground vector."""

    _, components = primaries.primary_components(branch_label, parity)
    answer = {}
    for auxiliary_modes, auxiliary_ground, virasoro, supercurrent, ground, c in (
        components
    ):
        if auxiliary_modes or virasoro or supercurrent:
            raise AssertionError("A Ramond onset state must be ground-level.")
        answer[((), auxiliary_ground, ground)] = c
    return answer


def decompose_l1_image(branch_label, parity):
    """Solve the image in the ordered onset basis n=(+1/4,-1/4)."""

    target = physical_l1_image(branch_label, parity)
    labels = (sp.Rational(1, 4), -sp.Rational(1, 4))
    columns = tuple(onset_vector(label, parity) for label in labels)
    states = tuple(sorted(set(target).union(*(set(column) for column in columns))))
    matrix = sp.Matrix(
        [[column.get(state, 0) for column in columns] for state in states]
    )
    vector = sp.Matrix([target.get(state, 0) for state in states])
    solution = tuple(
        sp.factor(sp.cancel(value))
        for value in sp.linsolve((matrix, vector)).args[0]
    )
    residual = matrix * sp.Matrix(solution) - vector
    if any(sp.cancel(value) != 0 for value in residual):
        raise AssertionError(residual)
    return dict(zip(labels, solution))


def closed_coefficient(branch_label, parity):
    """Closed coefficient multiplying W_(-sign(n)/4)^parity."""

    branch_label = sp.Rational(branch_label)
    if abs(branch_label) != sp.Rational(3, 4):
        raise ValueError("The first closed coefficient expects |n|=3/4.")
    copy_factor = 1 if int(parity) == 0 else 2
    return sp.Rational(copy_factor, 4)


def primary_in_fixed_fock(branch_label, parity, q_value, momentum):
    """Expand a branch primary in one common minus-realization Fock basis."""

    branch_label = sp.Rational(branch_label)
    substitutions = {branch.Q: q_value, branch.P: momentum}
    if abs(branch_label) > sp.Rational(3, 4):
        _, components = primaries.primary_components(
            branch_label,
            parity,
            q_value=q_value,
            momentum=momentum,
        )
    else:
        _, components = primaries.primary_components(branch_label, parity)
    answer = {}
    for (
        auxiliary_modes,
        auxiliary_ground,
        virasoro_modes,
        supercurrent_modes,
        physical_ground,
        outer,
    ) in components:
        descendant = (virasoro_modes, supercurrent_modes, physical_ground)
        physical_expression = branch.descendant_to_fock(
            descendant, realization=-1, momentum=momentum
        )
        for physical_state, inner in physical_expression.items():
            coefficient = (outer * inner).subs(substitutions, simultaneous=True)
            _add_product_term(
                answer,
                _join_product_state(
                    (auxiliary_modes, auxiliary_ground), physical_state
                ),
                coefficient,
            )
    return answer


def _evaluate_auxiliary_word(word, ground):
    """Evaluate an ordered auxiliary-fermion word on one Ramond ground."""

    expression = {((), int(ground)): sp.Integer(1)}
    for mode in reversed(tuple(word)):
        next_expression = {}
        for state, outer in expression.items():
            final, inner = branch.apply_auxiliary(int(mode), state)
            if inner:
                add_term(next_expression, final, outer * inner)
        expression = next_expression
    return expression


def auxiliary_lf_minus(expression, level):
    """Apply the auxiliary free-fermion Virasoro mode ``L^F_-level``.

    The implementation uses
    ``[L_m^F,f_r]=(-m/2-r)f_(m+r)`` and the exact Ramond-ground action.
    """

    level = int(level)
    if level <= 0:
        raise ValueError("A lowering-mode level must be positive.")
    answer = {}
    for full_state, outer in expression.items():
        auxiliary, physical = _split_product_state(full_state)
        modes, ground = auxiliary
        word = tuple(-mode for mode in modes)

        # Commutator with each creator in the canonical word.
        for position, creator in enumerate(modes):
            replaced = list(word)
            replaced[position] = -level - creator
            coefficient = sp.Rational(level, 2) + creator
            for final, inner in _evaluate_auxiliary_word(replaced, ground).items():
                _add_product_term(
                    answer,
                    _join_product_state(final, physical),
                    outer * coefficient * inner,
                )

        # Direct Ramond-ground action:
        #   L^F_-m|a> = (m/2) f_-m f_0|a>
        #     + sum_{j=1}^{floor((m-1)/2)} (m-2j)/2
        #         f_-(m-j) f_-j |a>.
        ground_image = {
            ((level,), 1 - ground): sp.Rational(level, 2) / SQRT2
        }
        for lower_mode in range(1, (level - 1) // 2 + 1):
            ground_image[((level - lower_mode, lower_mode), ground)] = (
                sp.Rational(level - 2 * lower_mode, 2)
            )
        for base, base_coefficient in ground_image.items():
            descendants = {base: base_coefficient}
            for mode in reversed(word):
                next_descendants = {}
                for state, coefficient in descendants.items():
                    final, inner = branch.apply_auxiliary(mode, state)
                    if inner:
                        add_term(
                            next_descendants, final, coefficient * inner
                        )
                descendants = next_descendants
            for final, inner in descendants.items():
                _add_product_term(
                    answer,
                    _join_product_state(final, physical),
                    outer * inner,
                )
    return answer


def physical_l_minus(expression, level, q_value, momentum):
    """Apply ``1_F x L_-level`` in the fixed physical realization."""

    answer = {}
    substitutions = {branch.Q: q_value, branch.P: momentum}
    for full_state, outer in expression.items():
        auxiliary, physical = _split_product_state(full_state)
        physical_image = branch.apply_L_to_state(
            -int(level), physical, realization=-1, momentum=momentum
        )
        for final, inner in physical_image.items():
            _add_product_term(
                answer,
                _join_product_state(auxiliary, final),
                outer * inner.subs(substitutions, simultaneous=True),
            )
    return answer


def physical_g_mode(mode, state, q_value, momentum):
    """Apply a physical free-field ``G_mode`` for any low integer mode."""

    mode = int(mode)
    if mode < 0:
        answer = branch.apply_G_to_state(
            mode, state, realization=-1, momentum=momentum
        )
        return {
            final: coefficient.subs(branch.Q, q_value)
            for final, coefficient in answer.items()
        }

    bosons, fermions, _ = state
    summation_modes = set(bosons)
    summation_modes.update(mode - fermion for fermion in fermions)
    summation_modes.add(mode)  # the physical zero-fermion term in the sum
    answer = {}
    for summation_mode in summation_modes:
        if summation_mode == 0:
            continue
        final, coefficient = branch.apply_two(
            lambda current, m=summation_mode: branch.apply_c(m, current),
            lambda current, r=mode - summation_mode: branch.apply_fermion(
                r, current, -1
            ),
            state,
        )
        if coefficient:
            add_term(answer, final, coefficient)

    final, coefficient = branch.apply_fermion(mode, state, -1)
    if coefficient:
        add_term(
            answer,
            final,
            I * (q_value * mode - momentum) * coefficient,
        )
    return answer


def mixed_u_minus(expression, level, q_value, momentum):
    """Apply ``U_-level=sum_r f_(-level-r) G_r``."""

    mode = -int(level)
    answer = {}
    for full_state, outer in expression.items():
        auxiliary, physical = _split_product_state(full_state)
        auxiliary_modes, auxiliary_ground = auxiliary
        maximum_auxiliary_mode = max(auxiliary_modes, default=0)
        physical_level = sum(physical[0]) + sum(physical[1])
        tensor_sign = (-1) ** (
            (len(auxiliary_modes) + auxiliary_ground) % 2
        )
        for supercurrent_mode in range(
            mode - maximum_auxiliary_mode, physical_level + 1
        ):
            auxiliary_final, auxiliary_coefficient = branch.apply_auxiliary(
                mode - supercurrent_mode, auxiliary
            )
            if not auxiliary_coefficient:
                continue
            physical_image = physical_g_mode(
                supercurrent_mode, physical, q_value, momentum
            )
            for physical_final, physical_coefficient in physical_image.items():
                _add_product_term(
                    answer,
                    _join_product_state(auxiliary_final, physical_final),
                    outer
                    * tensor_sign
                    * auxiliary_coefficient
                    * physical_coefficient,
                )
    return answer


def _linear_combination(terms):
    answer = {}
    for coefficient, expression in terms:
        for state, value in expression.items():
            _add_product_term(answer, state, coefficient * value)
    return answer


def double_virasoro_l_minus(
    expression, level, copy, b, momentum=branch.P
):
    """Apply one lowering mode of either embedded Virasoro algebra."""

    b = sp.sympify(b)
    q_value = b + 1 / b
    denominator = 1 / b - b
    physical = physical_l_minus(expression, level, q_value, momentum)
    auxiliary = auxiliary_lf_minus(expression, level)
    mixed = mixed_u_minus(expression, level, q_value, momentum)
    if int(copy) == 1:
        return _linear_combination(
            (
                ((1 / b) / denominator, physical),
                (-(1 / b + 2 * b) / denominator, auxiliary),
                (1 / denominator, mixed),
            )
        )
    if int(copy) == 2:
        return _linear_combination(
            (
                (-b / denominator, physical),
                ((b + 2 / b) / denominator, auxiliary),
                (-1 / denominator, mixed),
            )
        )
    raise ValueError("copy must be 1 or 2")


def double_virasoro_descendant(
    branch_label, parity, first, second, b, momentum=branch.P
):
    """Return ``L_-first^(1) L_-second^(2) W_label^parity``."""

    q_value = b + 1 / b
    expression = primary_in_fixed_fock(
        branch_label, parity, q_value=q_value, momentum=momentum
    )
    for level in reversed(tuple(second)):
        expression = double_virasoro_l_minus(
            expression, level, 2, b, momentum=momentum
        )
    for level in reversed(tuple(first)):
        expression = double_virasoro_l_minus(
            expression, level, 1, b, momentum=momentum
        )
    return expression


def physical_l1_positive_raw(branch_label, parity):
    """Apply physical L_1 directly to a positive-label finite chi string."""

    branch_label = sp.Rational(branch_label)
    if branch_label <= 0:
        raise ValueError("The raw fixed-realization calculation expects n>0.")
    realization, _, expression = branch.expand_chi_string(branch_label, parity)
    if realization != -1:
        raise AssertionError("A positive branch must use the minus realization.")
    answer = {}
    for full_state, outer in expression.items():
        auxiliary_modes, auxiliary_ground, bosons, fermions, ground = full_state
        if bosons:
            raise AssertionError("A finite chi string should contain no bosons.")
        word = tuple(-fermion for fermion in fermions)
        for position, fermion in enumerate(fermions):
            replaced = list(word)
            replaced[position] = 1 - fermion
            commutator = sp.Rational(2 * fermion - 1, 2)
            physical_expression = {((), (), ground): sp.Integer(1)}
            for mode in reversed(tuple(replaced)):
                next_expression = {}
                for state, coefficient in physical_expression.items():
                    final, inner = branch.apply_fermion(mode, state, -1)
                    if inner:
                        add_term(
                            next_expression, final, coefficient * inner
                        )
                physical_expression = next_expression
            for physical, inner in physical_expression.items():
                _add_product_term(
                    answer,
                    (
                        auxiliary_modes,
                        auxiliary_ground,
                        physical[0],
                        physical[1],
                        physical[2],
                    ),
                    outer * commutator * inner,
                )
    return answer


def auxiliary_l1_positive_raw(branch_label, parity):
    """Apply ``L_1^F`` directly to a positive-label finite chi string."""

    branch_label = sp.Rational(branch_label)
    if branch_label <= 0:
        raise ValueError("The raw fixed-realization calculation expects n>0.")
    _, _, expression = branch.expand_chi_string(branch_label, parity)
    answer = {}
    for full_state, outer in expression.items():
        auxiliary_modes, auxiliary_ground, bosons, fermions, ground = full_state
        word = tuple(-mode for mode in auxiliary_modes)
        for position, auxiliary_mode in enumerate(auxiliary_modes):
            replaced = list(word)
            replaced[position] = 1 - auxiliary_mode
            commutator = sp.Rational(2 * auxiliary_mode - 1, 2)
            for auxiliary, inner in _evaluate_auxiliary_word(
                replaced, auxiliary_ground
            ).items():
                _add_product_term(
                    answer,
                    (
                        auxiliary[0],
                        auxiliary[1],
                        bosons,
                        fermions,
                        ground,
                    ),
                    outer * commutator * inner,
                )
    return answer


def ramond_onset_level(branch_label):
    """Total enlarged-module level at which a Ramond branch starts."""

    branch_label = sp.Rational(branch_label)
    return int(2 * branch_label**2 - sp.Rational(1, 8))


def descendant_labels_at_level(level):
    """All lower Ramond branch labels whose onset does not exceed level."""

    labels = []
    k = 0
    while k * (k + 1) // 2 <= int(level):
        absolute_label = sp.Rational(2 * k + 1, 4)
        labels.extend((-absolute_label, absolute_label))
        k += 1
    return tuple(labels)


def descendant_basis(target_level):
    """Enumerate ``(n,A,B)`` with ``N_n+|A|+|B|=target_level``."""

    answer = []
    for label in descendant_labels_at_level(target_level):
        remaining = int(target_level) - ramond_onset_level(label)
        for first_level in range(remaining + 1):
            second_level = remaining - first_level
            for first, second in product(
                branch.partitions(first_level),
                branch.partitions(second_level),
            ):
                answer.append((label, first, second))
    return tuple(answer)


def expected_single_lower_basis(upper_label):
    """Descendants of ``W_(upper_label-1)`` at the physical-L1 level."""

    upper_label = sp.Rational(upper_label)
    lower_label = upper_label - 1
    remaining = (
        ramond_onset_level(upper_label)
        - 1
        - ramond_onset_level(lower_label)
    )
    answer = []
    for first_level in range(remaining + 1):
        second_level = remaining - first_level
        for first, second in product(
            branch.partitions(first_level),
            branch.partitions(second_level),
        ):
            answer.append((lower_label, first, second))
    return tuple(answer)


def decompose_positive_l1_single_lower_module(
    upper_label, parity, b, momentum=branch.P
):
    """Try the exact reduction ``L_1 W_n`` inside only the ``W_(n-1)`` module."""

    upper_label = sp.Rational(upper_label)
    if upper_label < sp.Rational(3, 4):
        raise ValueError("The first nonzero positive-label L_1 image is n=3/4.")
    target = physical_l1_positive_raw(upper_label, parity)
    descriptors = expected_single_lower_basis(upper_label)
    columns = tuple(
        double_virasoro_descendant(
            label,
            parity,
            first,
            second,
            b,
            momentum=momentum,
        )
        for label, first, second in descriptors
    )
    states = tuple(
        sorted(set(target).union(*(set(column) for column in columns)), key=str)
    )
    matrix = sp.Matrix(
        [[column.get(state, 0) for column in columns] for state in states]
    )
    vector = sp.Matrix([target.get(state, 0) for state in states])
    solution_set = sp.linsolve((matrix, vector))
    if solution_set is sp.EmptySet:
        raise AssertionError(
            f"L_1 W_({upper_label}) is not in the W_({upper_label - 1}) module."
        )
    solution = tuple(
        sp.factor(sp.cancel(value)) for value in solution_set.args[0]
    )
    residual = matrix * sp.Matrix(solution) - vector
    if any(sp.factor(sp.cancel(value)) != 0 for value in residual):
        raise AssertionError(
            f"The W_({upper_label - 1})-module residual is nonzero."
        )
    return dict(zip(descriptors, solution))


def decompose_positive_l1_at_five_quarters(parity, b):
    """Decompose physical ``L_1 W_(5/4)`` into level-two Vir x Vir states."""

    upper_label = sp.Rational(5, 4)
    # An exact solve in the complete 14-vector target basis shows that only
    # the five level-two descendants of the same-sheet onset W_(1/4) occur.
    # Solving this nonzero block directly keeps the generic (b,P) run quick.
    return decompose_positive_l1_single_lower_module(
        upper_label, parity, b
    )


def closed_five_quarter_coefficients(b):
    """Closed coefficients for the five level-two descendants of W_(1/4)."""

    b = sp.sympify(b)
    p = branch.P
    x = 2 * p * b
    return {
        (sp.Rational(1, 4), (), (2,)): -(
            x + b**2 + 6
        ) / ((x + b**2 + 4) * (x + 2 * b**2 + 1)),
        (sp.Rational(1, 4), (), (1, 1)): 2 * (
            2 * x + 3 * b**2 + 7
        )
        / (
            (x + b**2 + 2)
            * (x + b**2 + 4)
            * (x + 2 * b**2 + 1)
        ),
        (sp.Rational(1, 4), (1,), (1,)): 8 * b**2
        / ((x + b**2 + 2) * (x + 2 * b**2 + 1)),
        (sp.Rational(1, 4), (2,), ()): -b**2 * (
            x + 6 * b**2 + 1
        ) / ((x + b**2 + 2) * (x + 4 * b**2 + 1)),
        (sp.Rational(1, 4), (1, 1), ()): 2
        * b**4
        * (2 * x + 7 * b**2 + 3)
        / (
            (x + b**2 + 2)
            * (x + 2 * b**2 + 1)
            * (x + 4 * b**2 + 1)
        ),
    }


def descendant_name(label, parity, first, second):
    """Format one low Vir x Vir descendant without empty partitions."""

    operators = [f"L_-{mode}^(1)" for mode in first]
    operators.extend(f"L_-{mode}^(2)" for mode in second)
    operators.append(f"W_({label})^({parity})")
    return " ".join(operators)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--five-quarters",
        action="store_true",
        help="also solve the first genuine level-two descendant reduction",
    )
    parser.add_argument(
        "--b",
        default="b",
        help="embedded-algebra coupling used for descendant reductions",
    )
    parser.add_argument(
        "--higher-label",
        type=sp.Rational,
        help="solve L_1 W_n inside the conjectured W_(n-1) module",
    )
    parser.add_argument(
        "--momentum",
        type=sp.Rational,
        help="exact P specialization for higher-label calculations",
    )
    arguments = parser.parse_args()
    for label in (-sp.Rational(3, 4), sp.Rational(3, 4)):
        for parity in (0, 1):
            decomposition = decompose_l1_image(label, parity)
            print(f"L_1 W_({label})^({parity}) =")
            for lower_label, coefficient in decomposition.items():
                if coefficient:
                    print(f"  {coefficient} W_({lower_label})^({parity})")
    if arguments.five_quarters:
        b = sp.sympify(arguments.b)
        for parity in (0, 1):
            decomposition = decompose_positive_l1_at_five_quarters(parity, b)
            print(f"L_1 W_(5/4)^({parity}) =")
            for (label, first, second), coefficient in decomposition.items():
                print(
                    f"  ({coefficient}) "
                    f"{descendant_name(label, parity, first, second)}"
                )
    if arguments.higher_label is not None:
        b = sp.sympify(arguments.b)
        if arguments.higher_label > sp.Rational(5, 4) and (
            not b.is_Rational or arguments.momentum is None
        ):
            parser.error(
                "labels above 5/4 require exact rational --b and --momentum"
            )
        momentum = (
            branch.P if arguments.momentum is None else arguments.momentum
        )
        for parity in (0, 1):
            decomposition = decompose_positive_l1_single_lower_module(
                arguments.higher_label,
                parity,
                b,
                momentum=momentum,
            )
            print(f"L_1 W_({arguments.higher_label})^({parity}) =")
            for (label, first, second), coefficient in decomposition.items():
                print(
                    f"  ({coefficient}) "
                    f"{descendant_name(label, parity, first, second)}"
                )


if __name__ == "__main__":
    main()
