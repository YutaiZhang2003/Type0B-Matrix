#!/usr/bin/env python3
"""Exact action of the two Ramond zero modes on low branch states.

This is a diagnostic for the decomposition

    F_0 tensor SCA_0 = direct sum of Vir tensor Vir branches.

The branch vectors are imported from ``check_ramond_branching.py`` and
therefore use exactly the raw-state phases employed by the Ramond branching
and three-point checks.  All comparisons are made in one common abstract
Ramond PBW basis.  In particular, the plus and minus free-field
realizations used to construct the two branch sheets are never compared as
if their Fock oscillators were the same.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import sympy as sp

import check_ramond_branching as branch


I = sp.I
SQRT2 = sp.sqrt(2)
P = branch.P
Q = branch.Q


def _add(out, key, coefficient):
    coefficient = sp.factor(sp.cancel(coefficient))
    if coefficient == 0:
        return
    out[key] = sp.factor(sp.cancel(out.get(key, 0) + coefficient))
    if out[key] == 0:
        del out[key]


def flatten_branch(label, parity):
    """Return W_label^parity in the common auxiliary x abstract PBW basis."""

    _, sectors = branch.branch_in_abstract_basis(sp.Rational(label), int(parity))
    answer = {}
    for auxiliary_state, (level, physical_basis, coefficients) in sectors.items():
        for physical_state, coefficient in zip(physical_basis, coefficients):
            if coefficient != 0:
                _add(answer, (auxiliary_state, physical_state), coefficient)
    return answer


def apply_auxiliary_zero(expression):
    """Apply psi_0 on the first tensor factor."""

    answer = {}
    for (auxiliary_state, physical_state), outer in expression.items():
        final, coefficient = branch.apply_auxiliary(0, auxiliary_state)
        if coefficient:
            _add(answer, (final, physical_state), outer * coefficient)
    return answer


def apply_physical_g0_fock(state, realization=-1, momentum=P):
    """Apply the physical free-field G_0 to one Fock state.

    The mode-zero specialization of the free-field formula used by
    ``branch.apply_G_to_state`` is

        G_0 = sum_{m != 0} c_m theta_{-m} - i P theta_0

    in the minus realization.  The plus realization changes both displayed
    signs, as in ``branch.check_transition_and_gram``.  Only finitely many
    summands can act on a fixed Fock state.
    """

    bosons, fermions, _ = state
    answer = {}
    summation_modes = set(bosons)
    summation_modes.update(-mode for mode in fermions)
    for mode in summation_modes:
        final, coefficient = branch.apply_two(
            lambda current, m=mode: branch.apply_c(m, current),
            lambda current, m=mode: branch.apply_fermion(
                -m, current, realization
            ),
            state,
        )
        if coefficient:
            _add(answer, final, coefficient)

    final, coefficient = branch.apply_fermion(0, state, realization)
    if coefficient:
        momentum_coefficient = -I * momentum if realization == -1 else I * momentum
        _add(answer, final, momentum_coefficient * coefficient)
    return answer


def raw_branch(label, parity):
    """Return a branch in its native free-field realization."""

    realization, _, expression = branch.expand_chi_string(label, parity)
    return realization, expression


def apply_auxiliary_zero_raw(expression):
    answer = {}
    for state, outer in expression.items():
        aux_modes, aux_ground, bosons, fermions, physical_ground = state
        final, coefficient = branch.apply_auxiliary(0, (aux_modes, aux_ground))
        if coefficient:
            _add(
                answer,
                (final[0], final[1], bosons, fermions, physical_ground),
                outer * coefficient,
            )
    return answer


def apply_physical_zero_raw(expression, realization, oscillator_only=False):
    answer = {}
    for state, outer in expression.items():
        aux_modes, aux_ground, bosons, fermions, physical_ground = state
        physical_state = (bosons, fermions, physical_ground)
        if oscillator_only:
            physical_answer = apply_physical_g0_fock(
                physical_state, realization=realization, momentum=0
            )
        else:
            physical_answer = apply_physical_g0_fock(
                physical_state, realization=realization
            )
        auxiliary_parity = (len(aux_modes) + aux_ground) % 2
        tensor_sign = (-1) ** auxiliary_parity
        for final, coefficient in physical_answer.items():
            _add(
                answer,
                (aux_modes, aux_ground, final[0], final[1], final[2]),
                outer * tensor_sign * coefficient,
            )
    return answer


def apply_chi(expression, mode, realization):
    """Apply chi_mode=psi_mode-i theta_mode with graded tensor action."""

    answer = {}
    for state, outer in expression.items():
        aux_modes, aux_ground, bosons, fermions, physical_ground = state
        aux_final, aux_coefficient = branch.apply_auxiliary(
            mode, (aux_modes, aux_ground)
        )
        if aux_coefficient:
            _add(
                answer,
                (aux_final[0], aux_final[1], bosons, fermions, physical_ground),
                outer * aux_coefficient,
            )
        physical_final, physical_coefficient = branch.apply_fermion(
            mode, (bosons, fermions, physical_ground), realization
        )
        if physical_coefficient:
            auxiliary_parity = (len(aux_modes) + aux_ground) % 2
            _add(
                answer,
                (
                    aux_modes,
                    aux_ground,
                    physical_final[0],
                    physical_final[1],
                    physical_final[2],
                ),
                outer * (-I) * (-1) ** auxiliary_parity * physical_coefficient,
            )
    return answer


def ordered_chi_state(operators):
    """Build chi_1 ... chi_k |u+,Delta+> from (mode,realization) pairs."""

    expression = {((), 0, (), (), 0): sp.Integer(1)}
    for mode, realization in reversed(tuple(operators)):
        expression = apply_chi(expression, mode, realization)
    return expression


def create_physical_boson(expression, mode):
    """Apply c_{-mode}; ``mode`` is a positive integer."""

    answer = {}
    for state, outer in expression.items():
        aux_modes, aux_ground, bosons, fermions, physical_ground = state
        final, coefficient = branch.apply_c(
            -int(mode), (bosons, fermions, physical_ground)
        )
        if coefficient:
            _add(
                answer,
                (aux_modes, aux_ground, final[0], final[1], final[2]),
                outer * coefficient,
            )
    return answer


def oscillator_string_formula(mode_count, realization, append_opposite_zero):
    """Closed finite formula for (sum c_r theta_-r) W."""

    answer = {}
    for removed_mode in range(1, mode_count + 1):
        operators = [(0, realization)]
        operators.extend(
            (-mode, realization)
            for mode in range(1, mode_count + 1)
            if mode != removed_mode
        )
        if append_opposite_zero:
            operators.append((0, -realization))
        reduced = ordered_chi_state(operators)
        created = create_physical_boson(reduced, removed_mode)
        for key, value in created.items():
            _add(answer, key, -I * (-1) ** removed_mode * value)
    return answer


def abstract_g0_matrix(level):
    """Matrix of G_0 in the abstract PBW basis at one integer level."""

    physical_basis, transition = branch.transition(int(level), -1)
    row = {state: index for index, state in enumerate(physical_basis)}
    fock_action = sp.zeros(len(physical_basis), len(physical_basis))
    for column, state in enumerate(physical_basis):
        for final, coefficient in apply_physical_g0_fock(state).items():
            fock_action[row[final], column] += coefficient
    answer = transition.inv() * fock_action * transition
    return physical_basis, sp.simplify(answer)


def apply_physical_zero(expression):
    """Apply 1 tensor G_0, including the graded tensor-product sign."""

    by_auxiliary = defaultdict(dict)
    for (auxiliary_state, physical_state), coefficient in expression.items():
        by_auxiliary[auxiliary_state][physical_state] = coefficient

    answer = {}
    for auxiliary_state, physical_expression in by_auxiliary.items():
        level = sum(next(iter(physical_expression))[0]) + sum(
            next(iter(physical_expression))[1]
        )
        physical_basis, g0 = abstract_g0_matrix(level)
        vector = sp.Matrix([physical_expression.get(state, 0) for state in physical_basis])
        result = g0 * vector
        auxiliary_parity = (len(auxiliary_state[0]) + auxiliary_state[1]) % 2
        tensor_sign = (-1) ** auxiliary_parity
        for state, coefficient in zip(physical_basis, result):
            if coefficient:
                _add(answer, (auxiliary_state, state), tensor_sign * coefficient)
    return answer


def residual(left, right, coefficient=1):
    """Return left - coefficient*right as a sparse exact dictionary."""

    answer = dict(left)
    for key, value in right.items():
        _add(answer, key, -coefficient * value)
    return answer


def proportionality(left, right):
    """Return the exact proportionality coefficient, or None."""

    shared = [key for key in right if right[key] != 0]
    if not shared:
        raise ValueError("The comparison state is zero.")
    coefficient = sp.factor(sp.cancel(left.get(shared[0], 0) / right[shared[0]]))
    if residual(left, right, coefficient):
        return None
    return coefficient


def _scale_and_add(expressions):
    answer = {}
    for coefficient, expression in expressions:
        for key, value in expression.items():
            _add(answer, key, coefficient * value)
    return answer


def physical_l_minus_one_on_ground(expression):
    answer = {}
    for (auxiliary_state, physical_state), outer in expression.items():
        if physical_state[0] or physical_state[1]:
            raise ValueError("This helper acts only on physical ground states.")
        final = ((1,), (), physical_state[2])
        _add(answer, (auxiliary_state, final), outer)
    return answer


def auxiliary_lf_minus_one_on_ground(expression):
    """Apply L^F_-1=(1/2) psi_-1 psi_0 on an auxiliary ground state."""

    answer = {}
    for (auxiliary_state, physical_state), outer in expression.items():
        if auxiliary_state[0]:
            raise ValueError("This helper acts only on auxiliary ground states.")
        final = ((1,), 1 - auxiliary_state[1])
        _add(answer, (final, physical_state), outer / (2 * SQRT2))
    return answer


def u_minus_one_on_ground(expression):
    """Apply U_-1=psi_-1 G_0+psi_0 G_-1 on tensor ground states."""

    answer = {}
    for (auxiliary_state, physical_state), outer in expression.items():
        auxiliary_modes, auxiliary_ground = auxiliary_state
        if auxiliary_modes or physical_state[0] or physical_state[1]:
            raise ValueError("This helper acts only on tensor ground states.")
        physical_ground = physical_state[2]
        tensor_sign = (-1) ** auxiliary_ground

        # psi_-1 G_0
        _add(
            answer,
            (((1,), auxiliary_ground), ((), (), 1 - physical_ground)),
            outer * tensor_sign * (-I * P / SQRT2),
        )
        # psi_0 G_-1
        _add(
            answer,
            (((), 1 - auxiliary_ground), ((), (1,), physical_ground)),
            outer * tensor_sign / SQRT2,
        )
    return answer


def double_vir_l_minus_one(sheet, parity, copy, b):
    """Apply L_-1^(copy) to W_(sheet/4)^parity exactly."""

    ground = flatten_branch(sp.Rational(sheet, 4), parity)
    physical = physical_l_minus_one_on_ground(ground)
    auxiliary = auxiliary_lf_minus_one_on_ground(ground)
    mixed = u_minus_one_on_ground(ground)
    denominator = 1 / b - b
    if copy == 1:
        return _scale_and_add(
            (
                ((1 / b) / denominator, physical),
                (-(1 / b + 2 * b) / denominator, auxiliary),
                (1 / denominator, mixed),
            )
        )
    if copy == 2:
        return _scale_and_add(
            (
                (-b / denominator, physical),
                ((b + 2 / b) / denominator, auxiliary),
                (-1 / denominator, mixed),
            )
        )
    raise ValueError("copy must be 1 or 2")


def decompose_level_one(expression, output_parity, b):
    """Decompose a level-one vector into high branches and ground descendants."""

    q_substitution = {Q: b + 1 / b}
    columns = [
        {
            key: sp.factor(value.subs(q_substitution))
            for key, value in flatten_branch(sp.Rational(3, 4), output_parity).items()
        },
        {
            key: sp.factor(value.subs(q_substitution))
            for key, value in flatten_branch(-sp.Rational(3, 4), output_parity).items()
        },
    ]
    for sheet in (1, -1):
        for copy in (1, 2):
            columns.append(double_vir_l_minus_one(sheet, output_parity, copy, b))
    expression = {
        key: sp.factor(value.subs(q_substitution)) for key, value in expression.items()
    }
    keys = sorted(set(expression).union(*(set(column) for column in columns)), key=str)
    matrix = sp.Matrix([[column.get(key, 0) for column in columns] for key in keys])
    vector = sp.Matrix([expression.get(key, 0) for key in keys])

    # Each row has a common harmless spin-frame factor.  Removing it puts
    # the system over QQ(b,P), where fraction-free exact elimination is both
    # fast and stable (the generic SymPy expression-domain inverse is not).
    row_factors = {
        0: (SQRT2 * I, 1, SQRT2, I, SQRT2 * I, SQRT2),
        1: (1, SQRT2 * I, I, SQRT2, 1, I),
    }[int(output_parity)]
    for row, factor in enumerate(row_factors):
        for column in range(matrix.cols):
            matrix[row, column] = sp.cancel(matrix[row, column] / factor)
        vector[row] = sp.cancel(vector[row] / factor)

    from sympy.polys.domains import QQ
    from sympy.polys.matrices import DomainMatrix

    field = QQ.frac_field(b, P)
    domain_matrix = DomainMatrix(
        [
            [field.from_sympy(matrix[row, column]) for column in range(matrix.cols)]
            for row in range(matrix.rows)
        ],
        matrix.shape,
        field,
    )
    domain_vector = DomainMatrix(
        [[field.from_sympy(vector[row])] for row in range(vector.rows)],
        vector.shape,
        field,
    )
    numerator, denominator = domain_matrix.solve_den(domain_vector)
    denominator = field.to_sympy(denominator)
    coefficients = numerator.to_Matrix().applyfunc(
        lambda value: field.to_sympy(value) / denominator
    )
    if any(
        sp.factor(value) != 0
        for value in matrix * coefficients - vector
    ):
        raise AssertionError("Level-one Vir tensor Vir decomposition failed.")
    return tuple(sp.factor(sp.cancel(value)) for value in coefficients)


def level_one_decomposition_checks():
    b = sp.symbols("b", nonzero=True)
    labels = (
        "W_(+3/4)",
        "W_(-3/4)",
        "L_-1^(1) W_(+1/4)",
        "L_-1^(2) W_(+1/4)",
        "L_-1^(1) W_(-1/4)",
        "L_-1^(2) W_(-1/4)",
    )
    print("Exact level-one Vir x Vir decompositions (Q=b+b^-1):")
    for sheet in (1, -1):
        for parity in (0, 1):
            state = flatten_branch(sheet * sp.Rational(3, 4), parity)
            for operator, result in (
                ("psi0", apply_auxiliary_zero(state)),
                ("G0", apply_physical_zero(state)),
            ):
                coefficients = decompose_level_one(result, 1 - parity, b)
                terms = [
                    f"({coefficient}) {label}"
                    for coefficient, label in zip(coefficients, labels)
                    if coefficient != 0
                ]
                print(
                    f"  {operator} W_({sheet * 3}/4)^{parity} = "
                    + " + ".join(terms)
                )


def common_pbw_checks():
    """Common-basis checks through |n|=3/4 (fully symbolic in P,Q)."""

    for absolute_label in (sp.Rational(1, 4), sp.Rational(3, 4)):
        print(f"|n|={absolute_label}")
        for sign in (1, -1):
            label = sign * absolute_label
            for parity in (0, 1):
                state = flatten_branch(label, parity)
                reflected = flatten_branch(-label, 1 - parity)
                auxiliary = apply_auxiliary_zero(state)
                physical = apply_physical_zero(state)
                print(
                    f"  n={label}, epsilon={parity}: "
                    f"psi0/reflected={proportionality(auxiliary, reflected)}, "
                    f"G0/reflected={proportionality(physical, reflected)}, "
                    f"components=({len(auxiliary)},{len(physical)})"
                )


def raw_all_level_checks():
    """Check the exact string identities through |n|=5/4.

    These checks remain symbolic in P.  They do not require inversion of the
    level-three physical free-field transition matrix.
    """

    for absolute_label in (sp.Rational(1, 4), sp.Rational(3, 4), sp.Rational(5, 4)):
        for sign in (1, -1):
            label = sign * absolute_label
            realization = -sign
            mode_count = int(2 * absolute_label - sp.Rational(1, 2))
            main_parity = (mode_count + 1) % 2
            opposite_parity = 1 - main_parity
            _, main = raw_branch(label, main_parity)
            _, opposite = raw_branch(label, opposite_parity)

            negative_string = ordered_chi_state(
                [(-mode, realization) for mode in range(1, mode_count + 1)]
            )
            negative_then_opposite_zero = ordered_chi_state(
                [(-mode, realization) for mode in range(1, mode_count + 1)]
                + [(0, -realization)]
            )

            expected_main = dict(negative_string)
            for key, value in opposite.items():
                _add(
                    expected_main,
                    key,
                    -sp.Rational((-1) ** mode_count, 2) * value,
                )
            if residual(apply_auxiliary_zero_raw(main), expected_main):
                raise AssertionError((label, "auxiliary main-copy identity"))
            if residual(
                apply_auxiliary_zero_raw(opposite), negative_then_opposite_zero
            ):
                raise AssertionError((label, "auxiliary opposite-copy identity"))

            for parity, state in ((main_parity, main), (opposite_parity, opposite)):
                if apply_chi(state, 0, realization):
                    raise AssertionError((label, parity, "chi_0 does not annihilate"))
                auxiliary = apply_auxiliary_zero_raw(state)
                physical = apply_physical_zero_raw(state, realization)
                zero_mode_product = apply_auxiliary_zero_raw(physical)
                oscillator = apply_physical_zero_raw(
                    state, realization, oscillator_only=True
                )
                predicted_oscillator = oscillator_string_formula(
                    mode_count,
                    realization,
                    append_opposite_zero=(parity == opposite_parity),
                )
                if residual(oscillator, predicted_oscillator):
                    raise AssertionError((label, parity, "finite oscillator formula"))
                combination = dict(physical)
                for key, value in auxiliary.items():
                    _add(combination, key, sign * P * value)
                if residual(combination, oscillator):
                    raise AssertionError((label, parity, "G0+sP psi0 identity"))
                twice = apply_auxiliary_zero_raw(auxiliary)
                if residual(twice, state, sp.Rational(1, 2)):
                    raise AssertionError((label, parity, "psi0 square"))
                levels = {
                    sum(key[0]) + sum(key[2]) + sum(key[3])
                    for key in physical
                }
                expected_level = int(2 * absolute_label**2 - sp.Rational(1, 8))
                if levels != {expected_level}:
                    raise AssertionError((label, parity, levels))
                expected_output_parity = 1 - parity
                output_parities = {
                    (len(key[0]) + key[1] + len(key[3]) + key[4]) % 2
                    for key in physical
                }
                if output_parities != {expected_output_parity}:
                    raise AssertionError((label, parity, output_parities))

                product_coefficient = proportionality(zero_mode_product, state)
                if absolute_label == sp.Rational(1, 4):
                    expected_product = -sign * P / 2
                    if sp.simplify(product_coefficient - expected_product) != 0:
                        raise AssertionError((label, parity, product_coefficient))
                elif product_coefficient is not None:
                    raise AssertionError((label, parity, "psi0 G0 preserves sheet"))

                _, same_sheet_target = raw_branch(label, expected_output_parity)
                keys = set(auxiliary) | set(physical) | set(same_sheet_target)
                relation_matrix = sp.Matrix(
                    [
                        [
                            auxiliary.get(key, 0),
                            physical.get(key, 0),
                            -same_sheet_target.get(key, 0),
                        ]
                        for key in keys
                    ]
                )
                relation_rank = relation_matrix.rank()
                if absolute_label > sp.Rational(1, 4) and relation_rank != 3:
                    raise AssertionError((label, parity, "unexpected linear relation"))
                print(
                    f"raw n={label}, epsilon={parity}: "
                    f"psi0 terms={len(auxiliary)}, G0 terms={len(physical)}, "
                    f"level={expected_level}, output epsilon={expected_output_parity}, "
                    f"rank(psi0 W,G0 W,W_same^flip)={relation_rank}, "
                    f"psi0*G0/W={product_coefficient}"
                )

    # Independent validation that the free-field operator used above really
    # is the physical Ramond zero mode on every physical level encountered.
    for level in range(4):
        physical_basis = branch.basis(level)
        row = {state: index for index, state in enumerate(physical_basis)}
        g0 = sp.zeros(len(physical_basis))
        for column, state in enumerate(physical_basis):
            for final, coefficient in apply_physical_g0_fock(state).items():
                g0[row[final], column] += coefficient
        expected_square = (level - P**2 / 2) * sp.eye(len(physical_basis))
        if any(sp.expand(value) != 0 for value in g0 * g0 - expected_square):
            raise AssertionError((level, "G0 square"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--common-pbw",
        action="store_true",
        help="also compare the two sheets in a common abstract PBW basis",
    )
    parser.add_argument(
        "--level-one-decomposition",
        action="store_true",
        help="print the exact Vir x Vir decomposition at |n|=3/4",
    )
    arguments = parser.parse_args()
    raw_all_level_checks()
    print(
        "All raw zero-mode/string identities at |n|=1/4,3/4,5/4: "
        "exact symbolic match"
    )
    if arguments.common_pbw:
        common_pbw_checks()
    if arguments.level_one_decomposition:
        level_one_decomposition_checks()


if __name__ == "__main__":
    main()
