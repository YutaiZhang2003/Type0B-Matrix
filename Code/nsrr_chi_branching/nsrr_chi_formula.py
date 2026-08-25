#!/usr/bin/env python3
"""Exact finite chi-path formula for NS--R--R Vir+Vir primaries.

The construction in this file is the literal expansion of the 2016
Ramond highest-weight strings.  It does not assume an ell-product ansatz.
At fixed branch labels every chi mode is assigned either to the auxiliary
Majorana fermion or to the physical free-field realization of the SCA.
The resulting finite sum is contracted with the auxiliary Ising form and
the physical NS--R--R Ward form.

All arithmetic used by the public functions is exact SymPy arithmetic.
The physical Ramond ground labels are, throughout, the Human-Note states
``w^+`` and ``w^-``.  In particular, the physical fermion zero mode and the
free-field-to-PBW transition are constructed directly in that basis; no
second Ramond ground basis or endpoint rephasing is used.

The function ``raw_three_point`` returns the form with the unnormalised NS
chi state ``w_n`` in the first slot.  ``v_three_point`` applies the
normalization of ``v_n`` used in SCblock.tex.
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import itertools
from pathlib import Path
import sys

import sympy as sp


THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[1]
GRID_DIR = ROOT / "python" / "ramond_three_point_grid"
for directory in (GRID_DIR,):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import compute_grid as stored  # noqa: E402


I = sp.I
SQRT2 = sp.sqrt(2)
EIGHTH_MINUS = (1 - I) / SQRT2
EIGHTH_THREE = (-1 + I) / SQRT2
EIGHTH_NEG_THREE = (-1 - I) / SQRT2


def _add_term(expression, state, coefficient):
    coefficient = sp.cancel(coefficient)
    if coefficient == 0:
        return
    expression[state] = sp.cancel(expression.get(state, 0) + coefficient)
    if expression[state] == 0:
        del expression[state]


def ramond_mode_count(branch_label):
    """Return M=2|n|-1/2 for n in Z/2+1/4."""

    branch_label = sp.Rational(branch_label)
    mode_count = 2 * abs(branch_label) - sp.Rational(1, 2)
    if not mode_count.is_integer or mode_count < 0:
        raise ValueError("A Ramond branch label must lie in Z/2+1/4.")
    return int(mode_count)


def ramond_chi_chain(branch_label, parity):
    """Return the ordered tagged modes defining W_n^parity.

    A tag is ``(mode, realization)``.  The realization is ``-sgn(n)`` for
    the 2016 string chi^{-sgn(n)}.  If its parity is not the requested copy,
    the opposite zero mode is appended on the right.
    """

    branch_label = sp.Rational(branch_label)
    parity = int(parity)
    if parity not in (0, 1):
        raise ValueError("The Ramond copy parity must be 0 or 1.")
    if branch_label == 0:
        raise ValueError("There is no Ramond branch at n=0.")
    sign = 1 if branch_label > 0 else -1
    realization = -sign
    mode_count = ramond_mode_count(branch_label)
    chain = [(0, realization)]
    chain.extend((-mode, realization) for mode in range(1, mode_count + 1))
    if len(chain) % 2 != parity:
        chain.append((0, -realization))
    return tuple(chain)


def _fermion_action(mode, modes, ground, zero_sign=1):
    """Act on an auxiliary free-fermion Fock endpoint."""

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


def _physical_ramond_fermion_action(
    mode, modes, ground, realization
):
    """Act with the physical Ramond fermion directly on ``w^+``/``w^-``.

    Ground labels ``0`` and ``1`` mean ``w^+`` and ``w^-``.  The zero-mode
    phases are fixed by the Human-Note convention

        G_0 w^+ = i beta exp(-i*pi/4) w^-,
        G_0 w^- = i beta exp(+i*pi/4) w^+.

    The nonzero modes preserve the ground label and need no phase choice.
    """

    mode = int(mode)
    realization = int(realization)
    if realization not in (-1, 1):
        raise ValueError("realization must be +1 or -1")
    if mode < 0:
        created = -mode
        if created in modes:
            return None, sp.S.Zero
        crossings = sum(existing > created for existing in modes)
        final = tuple(sorted(modes + (created,), reverse=True))
        return (final, ground), sp.Integer((-1) ** crossings)
    if mode > 0:
        if mode not in modes:
            return None, sp.S.Zero
        position = modes.index(mode)
        return (
            (modes[:position] + modes[position + 1 :], ground),
            sp.Integer((-1) ** position),
        )
    phase = EIGHTH_THREE if ground == 0 else EIGHTH_NEG_THREE
    coefficient = (
        (-1) ** len(modes)
        * (-realization)
        * phase
        / SQRT2
    )
    return (modes, 1 - ground), sp.cancel(coefficient)


@lru_cache(None)
def ramond_fock_paths(branch_label, parity):
    """Expand W_n^parity into all auxiliary/physical binary paths."""

    chain = ramond_chi_chain(branch_label, parity)
    expression = {((), 0, (), 0): sp.Integer(1)}
    for mode, realization in reversed(chain):
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

            physical_final, physical_coefficient = _physical_ramond_fermion_action(
                mode,
                physical_modes,
                physical_ground,
                realization,
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
def _partitions(total, largest=None):
    if total == 0:
        return ((),)
    if largest is None or largest > total:
        largest = total
    answer = []
    for first in range(largest, 0, -1):
        for rest in _partitions(total - first, first):
            answer.append((first,) + rest)
    return tuple(answer)


@lru_cache(None)
def _strict_partitions(total, largest=None):
    if total == 0:
        return ((),)
    if largest is None or largest > total:
        largest = total
    answer = []
    for first in range(largest, 0, -1):
        for rest in _strict_partitions(total - first, first - 1):
            answer.append((first,) + rest)
    return tuple(answer)


@lru_cache(None)
def _ramond_w_basis(level):
    """Common Fock/PBW labels with ground 0/1 equal to ``w^+``/``w^-``."""

    return tuple(
        (bosons, fermions, ground)
        for boson_level in range(level + 1)
        for bosons in _partitions(boson_level)
        for fermions in _strict_partitions(level - boson_level)
        for ground in (0, 1)
    )


def _physical_c_action(mode, state):
    bosons, fermions, ground = state
    mode = int(mode)
    if mode < 0:
        created = -mode
        return (
            (tuple(sorted(bosons + (created,), reverse=True)), fermions, ground),
            sp.S.One,
        )
    if mode == 0:
        raise AssertionError("the bosonic zero mode is already evaluated")
    count = bosons.count(mode)
    if count == 0:
        return None, sp.S.Zero
    remaining = list(bosons)
    remaining.remove(mode)
    return (tuple(remaining), fermions, ground), sp.Integer(mode * count)


def _physical_fermion_action(mode, state, realization):
    bosons, fermions, ground = state
    final, coefficient = _physical_ramond_fermion_action(
        mode, fermions, ground, realization
    )
    if coefficient == 0:
        return None, sp.S.Zero
    return (bosons, final[0], final[1]), coefficient


def _apply_two(first, second, state):
    middle, coefficient_second = second(state)
    if coefficient_second == 0:
        return None, sp.S.Zero
    final, coefficient_first = first(middle)
    if coefficient_first == 0:
        return None, sp.S.Zero
    return final, sp.cancel(coefficient_second * coefficient_first)


def _physical_l_action(mode, state, realization, q_value, momentum):
    """Negative physical Virasoro mode in the direct ``w^\pm`` Fock basis."""

    mode = int(mode)
    if mode >= 0:
        raise ValueError("the PBW transition only applies negative L modes")
    bosons, fermions, _ = state
    answer = {}
    bosonic_indices = set(range(mode + 1, 0))
    bosonic_indices.update(bosons)
    bosonic_indices.update(mode - occupied for occupied in bosons)
    for summation_mode in bosonic_indices:
        if summation_mode in (0, mode):
            continue
        final, coefficient = _apply_two(
            lambda current, k=mode - summation_mode: _physical_c_action(k, current),
            lambda current, k=summation_mode: _physical_c_action(k, current),
            state,
        )
        if coefficient:
            _add_term(answer, final, coefficient / 2)

    fermionic_indices = set(range(mode, 1))
    fermionic_indices.update(fermions)
    fermionic_indices.update(mode - occupied for occupied in fermions)
    for summation_mode in fermionic_indices:
        final, coefficient = _apply_two(
            lambda current, r=mode - summation_mode: _physical_fermion_action(
                r, current, realization
            ),
            lambda current, r=summation_mode: _physical_fermion_action(
                r, current, realization
            ),
            state,
        )
        if coefficient:
            _add_term(
                answer,
                final,
                sp.Rational(summation_mode, 2) * coefficient,
            )

    final, coefficient = _physical_c_action(mode, state)
    if coefficient:
        _add_term(
            answer,
            final,
            I
            * (q_value * mode + 2 * realization * momentum)
            * coefficient
            / 2,
        )
    return answer


def _physical_g_action(mode, state, realization, q_value, momentum):
    """Negative physical supercurrent mode in the direct ``w^\pm`` basis."""

    mode = int(mode)
    if mode >= 0:
        raise ValueError("the PBW transition only applies negative G modes")
    bosons, fermions, _ = state
    answer = {}
    bosonic_indices = set(range(mode, 0))
    bosonic_indices.update(bosons)
    bosonic_indices.update(mode - occupied for occupied in fermions)
    for summation_mode in bosonic_indices:
        if summation_mode == 0:
            continue
        final, coefficient = _apply_two(
            lambda current, k=summation_mode: _physical_c_action(k, current),
            lambda current, r=mode - summation_mode: _physical_fermion_action(
                r, current, realization
            ),
            state,
        )
        if coefficient:
            _add_term(answer, final, coefficient)

    final, coefficient = _physical_fermion_action(mode, state, realization)
    if coefficient:
        _add_term(
            answer,
            final,
            I
            * (q_value * mode + realization * momentum)
            * coefficient,
        )
    return answer


def _apply_expression(expression, action):
    answer = {}
    for state, outer in expression.items():
        for final, inner in action(state).items():
            _add_term(answer, final, outer * inner)
    return answer


@lru_cache(None)
def _ramond_w_transition(level, realization, q_value, momentum):
    """Free-field-to-SCA PBW transition entirely in ``w^+``/``w^-``."""

    level = int(level)
    realization = int(realization)
    q_value = sp.sympify(q_value)
    momentum = sp.sympify(momentum)
    ordered_basis = _ramond_w_basis(level)
    row = {state: index for index, state in enumerate(ordered_basis)}
    matrix = sp.zeros(len(ordered_basis), len(ordered_basis))
    for column, (virasoro_modes, supercurrent_modes, ground) in enumerate(
        ordered_basis
    ):
        expression = {((), (), ground): sp.S.One}
        for mode in reversed(supercurrent_modes):
            expression = _apply_expression(
                expression,
                lambda state, mode=mode: _physical_g_action(
                    -mode, state, realization, q_value, momentum
                ),
            )
        for mode in reversed(virasoro_modes):
            expression = _apply_expression(
                expression,
                lambda state, mode=mode: _physical_l_action(
                    -mode, state, realization, q_value, momentum
                ),
            )
        for state, coefficient in expression.items():
            matrix[row[state], column] = coefficient
    if matrix.det() == 0:
        raise ZeroDivisionError(
            f"singular direct w-basis Ramond transition at level {level}"
        )
    return ordered_basis, matrix


@lru_cache(None)
def ramond_path_components(branch_label, parity, q_value, momentum):
    """Map every physical endpoint of a Ramond path to the SCA PBW basis."""

    branch_label = sp.Rational(branch_label)
    realization = -1 if branch_label > 0 else 1
    grouped = defaultdict(lambda: defaultdict(lambda: sp.Integer(0)))
    for state, coefficient in ramond_fock_paths(branch_label, parity):
        aux_modes, aux_ground, physical_modes, physical_ground = state
        grouped[(aux_modes, aux_ground)][
            ((), physical_modes, physical_ground)
        ] += coefficient

    answer = []
    for auxiliary_state, physical_expression in grouped.items():
        one_state = next(iter(physical_expression))
        level = sum(one_state[0]) + sum(one_state[1])
        ordered_basis, transition = _ramond_w_transition(
            level, realization, q_value, momentum
        )
        row = {state: index for index, state in enumerate(ordered_basis)}
        vector = sp.zeros(len(ordered_basis), 1)
        for state, coefficient in physical_expression.items():
            vector[row[state]] += coefficient
        coefficients = transition.inv() * vector
        for index, (virasoro_modes, supercurrent_modes, ground) in enumerate(
            ordered_basis
        ):
            coefficient = sp.cancel(coefficients[index])
            coefficient = sp.cancel(coefficient)
            if coefficient == 0:
                continue
            word = tuple(("L", -mode) for mode in virasoro_modes)
            word += tuple(("G", -mode) for mode in supercurrent_modes)
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
    """Expand the raw NS chi string; negative labels use reflection."""

    branch_label = sp.Rational(branch_label)
    if branch_label == 0:
        return (((), (), sp.Integer(1)),)
    effective_momentum = momentum if branch_label > 0 else -momentum
    branch_label = abs(branch_label)
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
                stored.ns_branch.abstract_eta_coefficients(physical_modes2)
            )
            coefficients = coefficients.subs(
                {
                    stored.ns_branch.Q: q_value,
                    stored.ns_branch.P: effective_momentum,
                },
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
                        tuple(
                            sp.Rational(mode2, 2)
                            for mode2 in auxiliary_modes2
                        ),
                        word,
                        coefficient,
                    )
                )
    return tuple(answer)


def _assemble_exact_number(components):
    return (
        components[0]
        + components[1] * SQRT2
        + I * components[2]
        + I * SQRT2 * components[3]
    )


def raw_three_point(
    n1,
    n2,
    n3,
    epsilon2,
    epsilon3,
    form_parity,
    eta,
    b_value,
    p1,
    p2,
    p3,
):
    """Compute rhohat_f^eta(w_n1,W_n2^eps2,W_n3^eps3).

    The first state is the raw NS chi string.  The auxiliary form parity is
    fixed by total parity and is returned together with the exact value.
    """

    n1, n2, n3 = map(sp.Rational, (n1, n2, n3))
    epsilon2, epsilon3 = int(epsilon2), int(epsilon3)
    form_parity, eta = int(form_parity), int(eta)
    q_value = sp.cancel(b_value + 1 / b_value)
    central_charge = sp.Rational(3, 2) + 3 * q_value**2
    h1 = (q_value**2 / 4 - p1**2) / 2
    h2 = sp.Rational(1, 16) + q_value**2 / 8 - p2**2 / 2
    h3 = sp.Rational(1, 16) + q_value**2 / 8 - p3**2 / 2
    physical = stored.PhysicalNRREvaluator(
        form_parity,
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
    third = ramond_path_components(n3, epsilon3, q_value, p3)
    auxiliary_form_parity = (
        int(2 * n1) + epsilon2 + epsilon3 - form_parity
    ) % 2

    exact_sample = not any(
        sp.sympify(value).free_symbols
        for value in (b_value, p1, p2, p3)
    )
    answer = sp.Integer(0)
    number_components = [sp.Integer(0)] * 4 if exact_sample else None
    for auxiliary1, word1, coefficient1 in first:
        physical_parity1 = stored.state_parity(word1)
        for auxiliary2, ground_a2, word2, ground_p2, coefficient2 in second:
            physical_parity2 = stored.state_parity(word2, ground_p2)
            auxiliary_parity2 = (len(auxiliary2) + ground_a2) % 2
            for auxiliary3, ground_a3, word3, ground_p3, coefficient3 in third:
                auxiliary_parity3 = (len(auxiliary3) + ground_a3) % 2
                auxiliary_value = stored.fermion_value_virasoro(
                    auxiliary_form_parity,
                    auxiliary1,
                    auxiliary2,
                    ground_a2,
                    auxiliary3,
                    ground_a3,
                )
                if auxiliary_value == 0:
                    continue
                physical_value = physical.value(
                    word1, word2, ground_p2, word3, ground_p3
                )
                if physical_value == 0:
                    continue
                koszul = (-1) ** (
                    physical_parity1 * (auxiliary_parity2 + auxiliary_parity3)
                    + physical_parity2 * auxiliary_parity3
                )
                term = (
                    koszul
                    * coefficient1
                    * coefficient2
                    * coefficient3
                    * auxiliary_value
                    * physical_value
                )
                if exact_sample:
                    for index, component in enumerate(
                        stored.quadratic_number_components(term)
                    ):
                        number_components[index] += component
                else:
                    answer += term
    if exact_sample:
        answer = _assemble_exact_number(number_components)
    return auxiliary_form_parity, sp.factor(sp.cancel(answer))


def ell(x, index, b_value):
    return stored.boundary.ell(x, int(index), b_value)


def ns_v_scale(branch_label, b_value, momentum):
    """Coefficient v_n/w_n in the normalization of SCblock.tex."""

    branch_label = sp.Rational(branch_label)
    if branch_label == 0:
        return sp.Integer(1)
    sign = 1 if branch_label > 0 else -1
    magnitude = abs(branch_label)
    q_value = sp.cancel(b_value + 1 / b_value)
    return sp.factor(
        sp.Rational(1, 2 ** int(2 * magnitude))
        * ell(q_value + 2 * sign * momentum, 4 * magnitude, b_value)
    )


def v_three_point(*arguments):
    """Compute rhohat_f^eta(v_n1,W_n2^eps2,W_n3^eps3)."""

    if len(arguments) != 11:
        raise TypeError("v_three_point expects the 11 raw_three_point arguments")
    n1 = arguments[0]
    b_value, p1 = arguments[7], arguments[8]
    auxiliary_parity, raw = raw_three_point(*arguments)
    return auxiliary_parity, sp.factor(
        sp.cancel(ns_v_scale(n1, b_value, p1) * raw)
    )


def ramond_norm(branch_label, parity, b_value, momentum):
    """Closed bilinear norm of the exact raw 2016 chain W_n^parity."""

    branch_label = sp.Rational(branch_label)
    parity = int(parity)
    sign = 1 if branch_label > 0 else -1
    mode_count = ramond_mode_count(branch_label)
    if parity == 0:
        discrete = 2 ** (2 * (mode_count // 2) + 1)
    elif parity == 1:
        discrete = -2 ** (2 * ((mode_count + 1) // 2))
    else:
        raise ValueError("The Ramond copy parity must be 0 or 1.")
    q_value = sp.cancel(b_value + 1 / b_value)
    return sp.factor(
        discrete
        * ell(2 * sign * momentum, 4 * abs(branch_label), b_value)
        / ell(q_value + 2 * sign * momentum, 4 * abs(branch_label), b_value)
    )


def ns_v_norm(branch_label, b_value, momentum):
    """Bilinear norm of v_n in the normalization of SCblock.tex."""

    branch_label = sp.Rational(branch_label)
    if branch_label == 0:
        return sp.Integer(1)
    sign = 1 if branch_label > 0 else -1
    magnitude = abs(branch_label)
    q_value = sp.cancel(b_value + 1 / b_value)
    return sp.factor(
        (-1) ** int(2 * magnitude)
        * sp.Rational(1, 2 ** int(2 * magnitude))
        * ell(2 * sign * momentum, 4 * magnitude, b_value)
        * ell(q_value + 2 * sign * momentum, 4 * magnitude, b_value)
    )


def branching_square(*arguments):
    """Return the root-independent normalized branching coefficient B^2."""

    if len(arguments) != 11:
        raise TypeError("branching_square expects the 11 raw_three_point arguments")
    n1, n2, n3, epsilon2, epsilon3 = arguments[:5]
    b_value, p1, p2, p3 = arguments[7:]
    auxiliary_parity, value = v_three_point(*arguments)
    denominator = (
        ns_v_norm(n1, b_value, p1)
        * ramond_norm(n2, epsilon2, b_value, p2)
        * ramond_norm(n3, epsilon3, b_value, p3)
    )
    return auxiliary_parity, sp.factor(sp.cancel(value**2 / denominator))


def master_reduction_factor(n2, n3, epsilon2, epsilon3, form_parity, eta):
    """Universal raw factor relative to (epsilon3,f)=(0,0), for n2,n3>0."""

    n2, n3 = sp.Rational(n2), sp.Rational(n3)
    if n2 <= 0 or n3 <= 0:
        raise ValueError("The displayed master reduction uses positive branches.")
    mode_count2 = ramond_mode_count(n2)
    mode_count3 = ramond_mode_count(n3)
    parity_scale3 = sp.Pow(
        2, sp.Rational((-1) ** (mode_count3 + 1), 2)
    )
    return sp.simplify(
        parity_scale3 ** int(epsilon3)
        * (
            int(eta)
            * (-1) ** (mode_count2 + 1 + int(epsilon2))
            * EIGHTH_MINUS
        )
        ** int(form_parity)
        * (-1) ** (int(epsilon3) * int(form_parity))
    )


def hard_crossed_master(epsilon2, eta, b_value, p1, p2, p3):
    """Closed first crossed master at (n1,n2,n3)=(0,3/4,3/4)."""

    epsilon2, eta = int(epsilon2), int(eta)
    if epsilon2 not in (0, 1) or eta not in (1, -1):
        raise ValueError("epsilon2 must be 0 or 1 and eta must be +/-1")
    q_value = sp.cancel(b_value + 1 / b_value)
    x_pp = q_value / 2 + p1 + p2 + p3
    x_mm = q_value / 2 + p1 - p2 - p3
    if eta == 1:
        master_zero = -(1 + I) * ell(x_pp, 3, b_value) * ell(
            x_mm, -3, b_value
        ) / (
            ell(q_value + 2 * p2, 3, b_value)
            * ell(q_value + 2 * p3, 3, b_value)
        )
    else:
        d2 = sp.Pow(2, -sp.Rational(1, 8)) * ell(
            q_value + 2 * p2, 3, b_value
        )
        d3 = sp.Pow(2, -sp.Rational(1, 8)) * ell(
            q_value + 2 * p3, 3, b_value
        )
        e2 = ell(q_value + 2 * p2, 2, b_value)
        e3 = ell(q_value + 2 * p3, 2, b_value)
        crossed = ell(x_pp, 2, b_value) * ell(x_mm, -2, b_value)
        master_zero = -(1 - I) * (
            crossed**2 + 2 * crossed * (e2 * e3 + 1) + d2 * d3
        ) / (d2 * d3)
    if epsilon2:
        return sp.factor(sp.cancel(I * SQRT2 * eta * master_zero))
    return sp.factor(sp.cancel(master_zero))
