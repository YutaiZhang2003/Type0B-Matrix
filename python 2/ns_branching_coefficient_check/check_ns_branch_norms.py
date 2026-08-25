#!/usr/bin/env python3
"""NS free-field to abstract super-Virasoro branching-state converter.

This module reconstructs the local interface formerly supplied by
``agent_notes/check_ns_branch_norms.py``.  It uses the positive-branch
free-field chart

    chi_r = f_r - i theta_r,

where ``f`` is the auxiliary Majorana fermion and ``theta`` is the NS
fermion in the free-field realization of the N=1 superconformal algebra.
At fixed branching label the finite chi string is expanded into all choices
of auxiliary and physical fermions.  Every physical Fock endpoint is then
transported to the abstract SCA PBW basis by inverting the free-field
transition matrix.

Mode labels ending in ``2`` are doubled: ``1`` denotes the mode 1/2,
``3`` denotes 3/2, and so on.  The historical public function name
``abstract_eta_coefficients`` is retained for compatibility; its "eta" is
the physical free-field fermion ``theta``, not the NS--R--R chiral sign.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from itertools import combinations

import sympy as sp


I = sp.I
P, Q = sp.symbols("P Q")
H, CENTRAL_CHARGE = sp.symbols("H CENTRAL_CHARGE")


def add_term(out, state, coefficient):
    """Add one exact coefficient to a sparse Fock expression."""

    coefficient = sp.cancel(coefficient)
    if coefficient == 0:
        return
    out[state] = sp.cancel(out.get(state, 0) + coefficient)
    if out[state] == 0:
        del out[state]


@lru_cache(None)
def partitions(total, largest=None):
    """Integer partitions in nonincreasing order."""

    total = int(total)
    if total < 0:
        return ()
    if total == 0:
        return ((),)
    if largest is None or largest > total:
        largest = total
    answer = []
    for first in range(int(largest), 0, -1):
        for rest in partitions(total - first, first):
            answer.append((first,) + rest)
    return tuple(answer)


@lru_cache(None)
def strict_odd_partitions(total, largest=None):
    """Partitions into distinct positive odd integers, decreasing."""

    total = int(total)
    if total < 0:
        return ()
    if total == 0:
        return ((),)
    if largest is None or largest > total:
        largest = total
    largest = int(largest)
    if largest % 2 == 0:
        largest -= 1
    answer = []
    for first in range(largest, 0, -2):
        for rest in strict_odd_partitions(total - first, first - 2):
            answer.append((first,) + rest)
    return tuple(answer)


@lru_cache(None)
def basis(level2):
    """Return the common NS Fock/PBW basis at twice the level.

    A basis state is ``(bosonic_modes, fermionic_modes2)``.  In the Fock
    interpretation these are ``c_-m`` and ``theta_-r`` creators.  In the
    abstract interpretation the same tuple denotes ``L_-m`` and ``G_-r``.
    Bosonic modes may repeat; fermionic modes are distinct positive odds.
    """

    level2 = int(level2)
    if level2 < 0:
        return ()
    answer = []
    for fermion_level2 in range(level2 + 1):
        boson_level2 = level2 - fermion_level2
        if boson_level2 % 2:
            continue
        for bosons in partitions(boson_level2 // 2):
            for fermions2 in strict_odd_partitions(fermion_level2):
                answer.append((bosons, fermions2))
    return tuple(answer)


def apply_c(mode, state):
    """Apply one Heisenberg mode with [c_m,c_n]=m delta_(m+n,0)."""

    mode = int(mode)
    bosons, fermions2 = state
    if mode < 0:
        created = -mode
        final = tuple(sorted(bosons + (created,), reverse=True))
        return (final, fermions2), sp.Integer(1)
    if mode == 0:
        raise AssertionError("The bosonic zero mode has already been evaluated.")
    count = bosons.count(mode)
    if count == 0:
        return None, sp.Integer(0)
    remaining = list(bosons)
    remaining.remove(mode)
    return (tuple(remaining), fermions2), sp.Integer(mode * count)


def apply_theta(mode2, state):
    """Apply the physical NS fermion theta_(mode2/2)."""

    mode2 = int(mode2)
    if mode2 == 0 or mode2 % 2 == 0:
        raise ValueError("An NS fermion mode must be a nonzero odd half-integer.")
    bosons, fermions2 = state
    if mode2 < 0:
        created = -mode2
        if created in fermions2:
            return None, sp.Integer(0)
        crossings = sum(existing > created for existing in fermions2)
        final = tuple(sorted(fermions2 + (created,), reverse=True))
        return (bosons, final), sp.Integer((-1) ** crossings)
    if mode2 not in fermions2:
        return None, sp.Integer(0)
    position = fermions2.index(mode2)
    remaining = fermions2[:position] + fermions2[position + 1 :]
    return (bosons, remaining), sp.Integer((-1) ** position)


# Historical local code sometimes called the physical free-field fermion eta.
apply_eta = apply_theta


def apply_two(first, second, state):
    """Apply ``first * second`` to a Fock state."""

    middle, coefficient_second = second(state)
    if coefficient_second == 0:
        return None, sp.Integer(0)
    final, coefficient_first = first(middle)
    if coefficient_first == 0:
        return None, sp.Integer(0)
    return final, coefficient_first * coefficient_second


def apply_L_to_state(mode, state, momentum=P):
    r"""Apply a negative abstract L mode in the positive-branch chart.

    The realization is

      L_m = 1/2 sum c_j c_(m-j)
            + 1/2 sum_r (r-m/2) theta_(m-r) theta_r
            + i (Q m/2-P) c_m.

    After pairing the two fermionic summands related by ``r -> m-r``, the
    coefficient used below is ``r/2``.
    """

    mode = int(mode)
    if mode >= 0:
        raise ValueError("This transition builder applies negative L modes only.")
    bosons, fermions2 = state
    out = {}

    bosonic_indices = set(range(mode + 1, 0))
    bosonic_indices.update(bosons)
    bosonic_indices.update(mode - existing for existing in bosons)
    for summation_mode in bosonic_indices:
        if summation_mode in (0, mode):
            continue
        final, coefficient = apply_two(
            lambda current, m=mode - summation_mode: apply_c(m, current),
            lambda current, m=summation_mode: apply_c(m, current),
            state,
        )
        if coefficient:
            add_term(out, final, sp.Rational(1, 2) * coefficient)

    fermionic_indices2 = set(range(2 * mode + 1, 0, 2))
    fermionic_indices2.update(fermions2)
    fermionic_indices2.update(2 * mode - existing for existing in fermions2)
    for summation_mode2 in fermionic_indices2:
        final, coefficient = apply_two(
            lambda current, r=2 * mode - summation_mode2: apply_theta(
                r, current
            ),
            lambda current, r=summation_mode2: apply_theta(r, current),
            state,
        )
        if coefficient:
            add_term(out, final, sp.Rational(summation_mode2, 4) * coefficient)

    final, coefficient = apply_c(mode, state)
    if coefficient:
        add_term(
            out,
            final,
            I * (sp.Rational(mode, 2) * Q - momentum) * coefficient,
        )
    return out


def apply_G_to_state(mode2, state, momentum=P):
    r"""Apply a negative abstract G_(mode2/2) mode in the same chart."""

    mode2 = int(mode2)
    if mode2 >= 0 or mode2 % 2 == 0:
        raise ValueError("This transition builder expects a negative NS G mode.")
    bosons, fermions2 = state
    out = {}

    bosonic_indices = set(range(mode2 // 2 + 1, 0))
    bosonic_indices.update(bosons)
    bosonic_indices.update((mode2 - existing) // 2 for existing in fermions2)
    for summation_mode in bosonic_indices:
        if summation_mode == 0:
            continue
        final, coefficient = apply_two(
            lambda current, m=summation_mode: apply_c(m, current),
            lambda current, r=mode2 - 2 * summation_mode: apply_theta(
                r, current
            ),
            state,
        )
        if coefficient:
            add_term(out, final, coefficient)

    final, coefficient = apply_theta(mode2, state)
    if coefficient:
        add_term(
            out,
            final,
            I * (sp.Rational(mode2, 2) * Q - momentum) * coefficient,
        )
    return out


def apply_to_expression(action, expression):
    """Apply a sparse oscillator action to a sparse expression."""

    out = {}
    for state, outer_coefficient in expression.items():
        for final, inner_coefficient in action(state).items():
            add_term(out, final, outer_coefficient * inner_coefficient)
    return out


def descendant_to_fock(descendant, momentum=P):
    """Map one abstract NS PBW descendant to the free-field Fock basis."""

    virasoro_modes, supercurrent_modes2 = descendant
    expression = {((), ()): sp.Integer(1)}
    for mode2 in reversed(supercurrent_modes2):
        expression = apply_to_expression(
            lambda state, r=-mode2: apply_G_to_state(r, state, momentum),
            expression,
        )
    for mode in reversed(virasoro_modes):
        expression = apply_to_expression(
            lambda state, n=-mode: apply_L_to_state(n, state, momentum),
            expression,
        )
    return expression


@lru_cache(None)
def transition(level2, momentum=P):
    """Return ``(basis, T)`` with T mapping PBW columns to Fock rows."""

    level2 = int(level2)
    ordered_basis = basis(level2)
    matrix = sp.zeros(len(ordered_basis), len(ordered_basis))
    row = {state: index for index, state in enumerate(ordered_basis)}
    for column, descendant in enumerate(ordered_basis):
        for state, coefficient in descendant_to_fock(
            descendant, momentum
        ).items():
            matrix[row[state], column] = coefficient
    return ordered_basis, matrix


@lru_cache(None)
def inverse_transition(level2, momentum=P):
    """Inverse free-field transition matrix at generic Q and momentum."""

    ordered_basis, matrix = transition(int(level2), momentum)
    return ordered_basis, matrix.inv()


def _canonical_physical_modes(physical_modes2):
    modes = tuple(int(mode) for mode in physical_modes2)
    if any(mode <= 0 or mode % 2 == 0 for mode in modes):
        raise ValueError("Physical NS mode labels must be positive odd integers.")
    if len(set(modes)) != len(modes):
        raise ValueError("Physical NS fermion modes must be distinct.")
    if modes != tuple(sorted(modes, reverse=True)):
        raise ValueError("Physical NS modes must be in decreasing order.")
    return modes


@lru_cache(None)
def abstract_eta_coefficients(physical_modes2):
    r"""Expand a pure physical-fermion endpoint in the abstract PBW basis.

    ``physical_modes2=(3,1)`` denotes
    ``theta_-3/2 theta_-1/2 |P>``.  The returned vector ``c`` satisfies

        theta endpoint = sum_I c[I] PBW_basis[I].
    """

    physical_modes2 = _canonical_physical_modes(physical_modes2)
    level2 = sum(physical_modes2)
    ordered_basis, inverse = inverse_transition(level2, P)
    target = sp.zeros(len(ordered_basis), 1)
    target[ordered_basis.index(((), physical_modes2))] = 1
    return ordered_basis, inverse * target


@lru_cache(None)
def inverse_transition_at(level2, q_value, momentum):
    """Transition inverse after substituting exact Q and P values.

    Substitution before inversion avoids the large generic rational-function
    inverse at higher NS levels.  The symbolic interface above remains the
    reference implementation; production grid evaluations use this exact
    specialization when their momenta are already known.
    """

    ordered_basis, matrix = transition(int(level2), P)
    specialized = matrix.subs(
        {Q: sp.sympify(q_value), P: sp.sympify(momentum)}, simultaneous=True
    )
    return ordered_basis, specialized.inv()


@lru_cache(None)
def abstract_eta_coefficients_at(physical_modes2, q_value, momentum):
    """Specialized version of :func:`abstract_eta_coefficients`."""

    physical_modes2 = _canonical_physical_modes(physical_modes2)
    level2 = sum(physical_modes2)
    ordered_basis, inverse = inverse_transition_at(
        level2, sp.sympify(q_value), sp.sympify(momentum)
    )
    target = sp.zeros(len(ordered_basis), 1)
    target[ordered_basis.index(((), physical_modes2))] = 1
    return ordered_basis, inverse * target


def coefficient_in_chi_product(all_modes2, physical_modes2):
    r"""Coefficient of one endpoint in ``prod (f_-r-i theta_-r)``.

    The tensor product is auxiliary-first.  Moving every selected physical
    fermion through later auxiliary fermions gives the Koszul crossing count.
    """

    all_modes2 = _canonical_physical_modes(all_modes2)
    physical_modes2 = _canonical_physical_modes(physical_modes2)
    if not set(physical_modes2).issubset(all_modes2):
        raise ValueError("The selected physical modes are not a subset of the chi string.")
    physical = set(physical_modes2)
    auxiliary_modes2 = tuple(mode for mode in all_modes2 if mode not in physical)
    crossings = sum(
        physical_mode > auxiliary_mode
        for physical_mode in physical_modes2
        for auxiliary_mode in auxiliary_modes2
    )
    return sp.expand((-I) ** len(physical_modes2) * (-1) ** crossings)


def chi_modes(branch_label):
    """Return doubled positive mode labels for a nonnegative NS branch."""

    branch_label = sp.Rational(branch_label)
    count = 2 * branch_label
    if not count.is_integer or count < 0:
        raise ValueError("An NS branch label must be a nonnegative half-integer.")
    count = int(count)
    return tuple(range(2 * count - 1, 0, -2))


@lru_cache(None)
def branch_in_abstract_basis(branch_label):
    """Group the raw NS chi string by auxiliary Fock endpoint."""

    all_modes2 = chi_modes(branch_label)
    sectors = {}
    for physical_count in range(len(all_modes2) + 1):
        for selected in combinations(all_modes2, physical_count):
            physical_modes2 = tuple(sorted(selected, reverse=True))
            physical = set(physical_modes2)
            auxiliary_modes2 = tuple(
                mode for mode in all_modes2 if mode not in physical
            )
            ordered_basis, coefficients = abstract_eta_coefficients(
                physical_modes2
            )
            chi_coefficient = coefficient_in_chi_product(
                all_modes2, physical_modes2
            )
            sectors[auxiliary_modes2] = (
                sum(physical_modes2),
                ordered_basis,
                coefficients * chi_coefficient,
            )
    return all_modes2, sectors


def super_bracket(first, second):
    """Graded bracket of two standard-normalized NS SCA modes."""

    first_kind, first_mode = first
    second_kind, second_mode = second
    answer = []
    if first_kind == "L" and second_kind == "L":
        answer.append(
            (
                first_mode - second_mode,
                (("L", first_mode + second_mode),),
            )
        )
        if first_mode + second_mode == 0:
            answer.append(
                (
                    CENTRAL_CHARGE * (first_mode**3 - first_mode) / 12,
                    (),
                )
            )
    elif first_kind == "L" and second_kind == "G":
        answer.append(
            (
                sp.Rational(first_mode, 2) - second_mode,
                (("G", first_mode + second_mode),),
            )
        )
    elif first_kind == "G" and second_kind == "L":
        answer.append(
            (
                first_mode - sp.Rational(second_mode, 2),
                (("G", first_mode + second_mode),),
            )
        )
    else:
        answer.append((sp.Integer(2), (("L", first_mode + second_mode),)))
        if first_mode + second_mode == 0:
            answer.append(
                (
                    CENTRAL_CHARGE
                    * (first_mode**2 - sp.Rational(1, 4))
                    / 3,
                    (),
                )
            )
    return tuple((sp.expand(coefficient), word) for coefficient, word in answer)


@lru_cache(None)
def highest_weight_expectation(word):
    """Reduce ``<P| word |P>`` with the NS SCA commutators."""

    if not word:
        return sp.Integer(1)
    if word[0][1] < 0 or word[-1][1] > 0:
        return sp.Integer(0)

    for position in range(len(word) - 1):
        first, second = word[position], word[position + 1]
        if first[1] >= 0 and second[1] < 0:
            exchange_sign = -1 if first[0] == second[0] == "G" else 1
            exchanged = word[:position] + (second, first) + word[position + 2 :]
            answer = exchange_sign * highest_weight_expectation(exchanged)
            for coefficient, replacement in super_bracket(first, second):
                reduced = word[:position] + replacement + word[position + 2 :]
                answer += coefficient * highest_weight_expectation(reduced)
            return sp.expand(answer)

    if any(mode != 0 for _, mode in word):
        raise AssertionError(f"Nonzero NS modes were not reduced: {word}")
    if any(kind != "L" for kind, _ in word):
        raise AssertionError(f"An NS G_0 mode cannot occur: {word}")
    return H ** len(word)


def descendant_word(descendant, bra=False):
    """Convert one PBW basis tuple to a signed-mode word."""

    virasoro_modes, supercurrent_modes2 = descendant
    if not bra:
        return tuple(("L", -sp.Integer(mode)) for mode in virasoro_modes) + tuple(
            ("G", -sp.Rational(mode2, 2)) for mode2 in supercurrent_modes2
        )
    return tuple(
        ("G", sp.Rational(mode2, 2))
        for mode2 in reversed(supercurrent_modes2)
    ) + tuple(("L", sp.Integer(mode)) for mode in reversed(virasoro_modes))


@lru_cache(None)
def abstract_gram(level2):
    """Generic NS Shapovalov matrix in the converter's PBW order."""

    ordered_basis = basis(int(level2))
    matrix = sp.zeros(len(ordered_basis), len(ordered_basis))
    for row, left in enumerate(ordered_basis):
        left_word = descendant_word(left, bra=True)
        for column, right in enumerate(ordered_basis):
            matrix[row, column] = highest_weight_expectation(
                left_word + descendant_word(right)
            )
    if any(sp.expand(value) != 0 for value in matrix - matrix.T):
        raise AssertionError(f"NS Gram matrix at twice-level {level2} is not symmetric.")
    return ordered_basis, matrix


@lru_cache(None)
def substituted_gram(level2):
    """NS Gram matrix on h=(Q^2/4-P^2)/2 and c=3/2+3Q^2."""

    highest_weight = (Q**2 / 4 - P**2) / 2
    central_charge = sp.Rational(3, 2) + 3 * Q**2
    return abstract_gram(int(level2))[1].subs(
        {H: highest_weight, CENTRAL_CHARGE: central_charge}, simultaneous=True
    )


def auxiliary_norm(auxiliary_modes2):
    """BPZ norm for f_r^dagger=-f_-r in canonical mode order."""

    return sp.Integer((-1) ** len(auxiliary_modes2))


def branch_norm(branch_label, substitutions=None):
    """Direct bilinear norm of the raw NS chi-string branch."""

    operators, sectors = branch_in_abstract_basis(sp.Rational(branch_label))
    total = sp.Integer(0)
    component_count = 0
    for auxiliary_modes2, (level2, _, coefficients) in sectors.items():
        gram = substituted_gram(level2)
        if substitutions:
            gram = gram.subs(substitutions, simultaneous=True)
            coefficients = coefficients.subs(substitutions, simultaneous=True)
        total += auxiliary_norm(auxiliary_modes2) * (
            coefficients.T * gram * coefficients
        )[0]
        component_count += sum(sp.cancel(value) != 0 for value in coefficients)
    return operators, sectors, component_count, sp.factor(sp.cancel(total))


def audit_endpoints(max_branch=sp.Integer(1)):
    """Check exact round trips for every chi endpoint through max_branch."""

    max_branch = sp.Rational(max_branch)
    checked = 0
    for twice_branch in range(int(2 * max_branch) + 1):
        modes = chi_modes(sp.Rational(twice_branch, 2))
        for count in range(len(modes) + 1):
            for selected in combinations(modes, count):
                selected = tuple(sorted(selected, reverse=True))
                ordered_basis, coefficients = abstract_eta_coefficients(selected)
                _, matrix = transition(sum(selected), P)
                target = sp.zeros(len(ordered_basis), 1)
                target[ordered_basis.index(((), selected))] = 1
                residual = (matrix * coefficients - target).applyfunc(
                    lambda value: sp.factor(sp.cancel(value))
                )
                if any(value != 0 for value in residual):
                    raise AssertionError((selected, residual))
                checked += 1
    return checked


def audit_first_branch():
    """Check the documented normalized n=1/2 branching vector."""

    ordered_basis, coefficients = abstract_eta_coefficients((1,))
    target = ((), (1,))
    coefficient = sp.factor(coefficients[ordered_basis.index(target)])
    expected = I / (Q / 2 + P)
    if sp.factor(sp.cancel(coefficient - expected)) != 0:
        raise AssertionError((coefficient, expected))
    raw_physical = sp.factor(
        coefficient_in_chi_product((1,), (1,)) * coefficient
    )
    if sp.factor(sp.cancel(raw_physical - 1 / (Q / 2 + P))) != 0:
        raise AssertionError(raw_physical)
    raw_norm = branch_norm(sp.Rational(1, 2))[3]
    expected_raw_norm = -4 * P / (Q + 2 * P)
    if sp.factor(sp.cancel(raw_norm - expected_raw_norm)) != 0:
        raise AssertionError((raw_norm, expected_raw_norm))
    return sp.factor(raw_physical)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-branch",
        type=sp.Rational,
        default=sp.Integer(1),
        help="largest nonnegative half-integer NS branch to round-trip",
    )
    arguments = parser.parse_args()
    first = audit_first_branch()
    checked = audit_endpoints(arguments.max_branch)
    print(f"n=1/2 raw G coefficient: {first}")
    print(f"NS free-field/PBW endpoint audit: {checked} exact round trips")


if __name__ == "__main__":
    main()
