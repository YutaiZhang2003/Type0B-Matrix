#!/usr/bin/env python3
"""Finite-field transition solves for the high NS and Ramond branches.

This is a profiling prototype, not the production three-point evaluator.  It
implements exactly the same free-field realization as
``check_ns_branch_norms.py`` and ``check_ramond_branching.py``, but it

* substitutes ``Q`` and ``P`` before constructing a transition matrix;
* works over a prime field, so intermediate rational functions never grow;
* uses the ground/parity block of the Ramond transition matrix; and
* solves all requested Fock right-hand sides in one Gauss--Jordan pass.

Repeating the calculation at independent primes, combining the residues by
CRT, and rationally reconstructing only the final three-point scalar gives an
exact algorithm.  The finite-field calculation therefore measures the useful
cost of the PBW fallback without confusing it with SymPy expression swell.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations
from pathlib import Path
import importlib.util
import sys
import time

import numpy as np
import sympy as sp


ROOT = Path(__file__).resolve().parents[3]
PYTHON_DIR = ROOT / "python 2"
RAMOND_DIR = PYTHON_DIR / "ramond_branching_coefficient_check"
if str(RAMOND_DIR) not in sys.path:
    sys.path.insert(0, str(RAMOND_DIR))

import check_ramond_branching as ramond_reference  # noqa: E402


def _load_ns_reference():
    source = (
        PYTHON_DIR
        / "ns_branching_coefficient_check"
        / "check_ns_branch_norms.py"
    )
    specification = importlib.util.spec_from_file_location("ns_reference", source)
    if specification is None or specification.loader is None:
        raise ImportError(source)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


ns_reference = _load_ns_reference()


def rational_mod(value, prime: int) -> int:
    """Map a rational number to ``GF(prime)``."""

    value = sp.Rational(value)
    numerator = int(value.p) % prime
    denominator = int(value.q) % prime
    if denominator == 0:
        raise ZeroDivisionError(f"denominator vanishes modulo {prime}")
    return numerator * pow(denominator, -1, prime) % prime


def roots(prime: int) -> tuple[int, int]:
    """Return fixed roots of -1 and 2 in ``GF(prime)``."""

    if prime % 8 != 1 or not sp.isprime(prime):
        raise ValueError("the profiling prime must be prime and congruent to 1 mod 8")
    root_i = int(sp.sqrt_mod(-1, prime, all_roots=False))
    root_two = int(sp.sqrt_mod(2, prime, all_roots=False))
    return root_i, root_two


def add_term(out, state, coefficient, prime):
    coefficient %= prime
    if coefficient == 0:
        return
    final = (out.get(state, 0) + coefficient) % prime
    if final:
        out[state] = final
    elif state in out:
        del out[state]


def apply_c(mode, state, prime):
    bosons = state[0]
    if mode < 0:
        created = -mode
        final = (tuple(sorted(bosons + (created,), reverse=True)),) + state[1:]
        return final, 1
    if mode == 0:
        raise AssertionError("the bosonic zero mode is already evaluated")
    count = bosons.count(mode)
    if not count:
        return None, 0
    remaining = list(bosons)
    remaining.remove(mode)
    return (tuple(remaining),) + state[1:], mode * count % prime


def apply_ns_fermion(mode2, state, prime):
    bosons, fermions = state
    if mode2 < 0:
        created = -mode2
        if created in fermions:
            return None, 0
        crossings = sum(existing > created for existing in fermions)
        final = tuple(sorted(fermions + (created,), reverse=True))
        return (bosons, final), (-1 if crossings % 2 else 1) % prime
    if mode2 not in fermions:
        return None, 0
    position = fermions.index(mode2)
    final = fermions[:position] + fermions[position + 1 :]
    return (bosons, final), (-1 if position % 2 else 1) % prime


def apply_two(first, second, state, prime):
    middle, coefficient_second = second(state)
    if not coefficient_second:
        return None, 0
    final, coefficient_first = first(middle)
    if not coefficient_first:
        return None, 0
    return final, coefficient_first * coefficient_second % prime


def apply_ns_L(mode, state, q_value, momentum, root_i, prime):
    if mode >= 0:
        raise ValueError(mode)
    bosons, fermions = state
    out = {}
    inv2 = pow(2, -1, prime)
    inv4 = pow(4, -1, prime)

    indices = set(range(mode + 1, 0))
    indices.update(bosons)
    indices.update(mode - existing for existing in bosons)
    for summation_mode in indices:
        if summation_mode in (0, mode):
            continue
        final, coefficient = apply_two(
            lambda current, m=mode - summation_mode: apply_c(m, current, prime),
            lambda current, m=summation_mode: apply_c(m, current, prime),
            state,
            prime,
        )
        if coefficient:
            add_term(out, final, inv2 * coefficient, prime)

    indices2 = set(range(2 * mode + 1, 0, 2))
    indices2.update(fermions)
    indices2.update(2 * mode - existing for existing in fermions)
    for summation_mode2 in indices2:
        final, coefficient = apply_two(
            lambda current, r=2 * mode - summation_mode2: apply_ns_fermion(
                r, current, prime
            ),
            lambda current, r=summation_mode2: apply_ns_fermion(r, current, prime),
            state,
            prime,
        )
        if coefficient:
            add_term(out, final, summation_mode2 * inv4 * coefficient, prime)

    final, coefficient = apply_c(mode, state, prime)
    if coefficient:
        linear = mode * inv2 * q_value - momentum
        add_term(out, final, root_i * linear * coefficient, prime)
    return out


def apply_ns_G(mode2, state, q_value, momentum, root_i, prime):
    if mode2 >= 0 or mode2 % 2 == 0:
        raise ValueError(mode2)
    bosons, fermions = state
    out = {}
    inv2 = pow(2, -1, prime)
    indices = set(range(mode2 // 2 + 1, 0))
    indices.update(bosons)
    indices.update((mode2 - existing) // 2 for existing in fermions)
    for summation_mode in indices:
        if summation_mode == 0:
            continue
        final, coefficient = apply_two(
            lambda current, m=summation_mode: apply_c(m, current, prime),
            lambda current, r=mode2 - 2 * summation_mode: apply_ns_fermion(
                r, current, prime
            ),
            state,
            prime,
        )
        if coefficient:
            add_term(out, final, coefficient, prime)
    final, coefficient = apply_ns_fermion(mode2, state, prime)
    if coefficient:
        linear = mode2 * inv2 * q_value - momentum
        add_term(out, final, root_i * linear * coefficient, prime)
    return out


def apply_expression(action, expression, prime):
    out = {}
    for state, outer in expression.items():
        for final, inner in action(state).items():
            add_term(out, final, outer * inner, prime)
    return out


def ns_descendant_to_fock(descendant, q_value, momentum, root_i, prime):
    virasoro_modes, supercurrent_modes2 = descendant
    expression = {((), ()): 1}
    for mode2 in reversed(supercurrent_modes2):
        expression = apply_expression(
            lambda state, r=-mode2: apply_ns_G(
                r, state, q_value, momentum, root_i, prime
            ),
            expression,
            prime,
        )
    for mode in reversed(virasoro_modes):
        expression = apply_expression(
            lambda state, n=-mode: apply_ns_L(
                n, state, q_value, momentum, root_i, prime
            ),
            expression,
            prime,
        )
    return expression


def ns_transition(level2, q_value, momentum, root_i, prime):
    basis = ns_reference.basis(level2)
    row = {state: index for index, state in enumerate(basis)}
    matrix = np.zeros((len(basis), len(basis)), dtype=np.int64)
    for column, descendant in enumerate(basis):
        for state, coefficient in ns_descendant_to_fock(
            descendant, q_value, momentum, root_i, prime
        ).items():
            matrix[row[state], column] = coefficient
    return basis, matrix


def apply_r_fermion(mode, state, realization, root_two, prime):
    bosons, fermions, ground = state
    if mode < 0:
        created = -mode
        if created in fermions:
            return None, 0
        crossings = sum(existing > created for existing in fermions)
        final = tuple(sorted(fermions + (created,), reverse=True))
        return (bosons, final, ground), (-1 if crossings % 2 else 1) % prime
    if mode > 0:
        if mode not in fermions:
            return None, 0
        position = fermions.index(mode)
        final = fermions[:position] + fermions[position + 1 :]
        return (bosons, final, ground), (-1 if position % 2 else 1) % prime
    zero_sign = 1 if realization == -1 else -1
    sign = -1 if len(fermions) % 2 else 1
    coefficient = zero_sign * sign * pow(root_two, -1, prime)
    return (bosons, fermions, 1 - ground), coefficient % prime


def apply_r_L(mode, state, realization, q_value, momentum, root_i, root_two, prime):
    if mode >= 0:
        raise ValueError(mode)
    bosons, fermions, _ = state
    out = {}
    inv2 = pow(2, -1, prime)
    indices = set(range(mode + 1, 0))
    indices.update(bosons)
    indices.update(mode - existing for existing in bosons)
    for summation_mode in indices:
        if summation_mode in (0, mode):
            continue
        final, coefficient = apply_two(
            lambda current, m=mode - summation_mode: apply_c(m, current, prime),
            lambda current, m=summation_mode: apply_c(m, current, prime),
            state,
            prime,
        )
        if coefficient:
            add_term(out, final, inv2 * coefficient, prime)

    indices = set(range(mode, 1))
    indices.update(fermions)
    indices.update(mode - existing for existing in fermions)
    for summation_mode in indices:
        final, coefficient = apply_two(
            lambda current, r=mode - summation_mode: apply_r_fermion(
                r, current, realization, root_two, prime
            ),
            lambda current, r=summation_mode: apply_r_fermion(
                r, current, realization, root_two, prime
            ),
            state,
            prime,
        )
        if coefficient:
            add_term(out, final, summation_mode * inv2 * coefficient, prime)

    final, coefficient = apply_c(mode, state, prime)
    if coefficient:
        momentum_term = mode * q_value
        momentum_term += -2 * momentum if realization == -1 else 2 * momentum
        add_term(out, final, root_i * inv2 * momentum_term * coefficient, prime)
    return out


def apply_r_G(mode, state, realization, q_value, momentum, root_i, root_two, prime):
    if mode >= 0:
        raise ValueError(mode)
    bosons, fermions, _ = state
    out = {}
    indices = set(range(mode, 0))
    indices.update(bosons)
    indices.update(mode - existing for existing in fermions)
    for summation_mode in indices:
        if summation_mode == 0:
            continue
        final, coefficient = apply_two(
            lambda current, m=summation_mode: apply_c(m, current, prime),
            lambda current, r=mode - summation_mode: apply_r_fermion(
                r, current, realization, root_two, prime
            ),
            state,
            prime,
        )
        if coefficient:
            add_term(out, final, coefficient, prime)
    final, coefficient = apply_r_fermion(
        mode, state, realization, root_two, prime
    )
    if coefficient:
        momentum_term = mode * q_value
        momentum_term += -momentum if realization == -1 else momentum
        add_term(out, final, root_i * momentum_term * coefficient, prime)
    return out


def r_descendant_to_fock(
    descendant, realization, q_value, momentum, root_i, root_two, prime
):
    virasoro_modes, supercurrent_modes, ground = descendant
    expression = {((), (), ground): 1}
    for mode in reversed(supercurrent_modes):
        expression = apply_expression(
            lambda state, r=-mode: apply_r_G(
                r,
                state,
                realization,
                q_value,
                momentum,
                root_i,
                root_two,
                prime,
            ),
            expression,
            prime,
        )
    for mode in reversed(virasoro_modes):
        expression = apply_expression(
            lambda state, n=-mode: apply_r_L(
                n,
                state,
                realization,
                q_value,
                momentum,
                root_i,
                root_two,
                prime,
            ),
            expression,
            prime,
        )
    return expression


def ramond_transition_block(
    level,
    parity,
    realization,
    q_value,
    momentum,
    root_i,
    root_two,
    prime,
    ground=0,
):
    full_basis = ramond_reference.basis(level)
    basis = tuple(
        state
        for state in full_basis
        if state[2] == int(ground) and len(state[1]) % 2 == int(parity)
    )
    row = {state: index for index, state in enumerate(basis)}
    matrix = np.zeros((len(basis), len(basis)), dtype=np.int64)
    for column, descendant in enumerate(basis):
        for state, coefficient in r_descendant_to_fock(
            descendant,
            realization,
            q_value,
            momentum,
            root_i,
            root_two,
            prime,
        ).items():
            if state in row:
                matrix[row[state], column] = coefficient
    return basis, matrix


def ramond_transition_sector(
    level, parity, realization, q_value, momentum, root_i, root_two, prime
):
    """Full Ramond total-parity block, retaining both ground vectors.

    Negative supercurrent modes can exchange the two Ramond grounds through
    their zero-mode term.  A fixed-ground submatrix is sufficient for a few
    profiling targets, but a general chi endpoint must use this genuine
    conserved-parity block.
    """

    full_basis = ramond_reference.basis(level)
    basis = tuple(
        state
        for state in full_basis
        if (len(state[1]) + state[2]) % 2 == int(parity)
    )
    row = {state: index for index, state in enumerate(basis)}
    matrix = np.zeros((len(basis), len(basis)), dtype=np.int64)
    for column, descendant in enumerate(basis):
        for state, coefficient in r_descendant_to_fock(
            descendant,
            realization,
            q_value,
            momentum,
            root_i,
            root_two,
            prime,
        ).items():
            if state in row:
                matrix[row[state], column] = coefficient
    return basis, matrix


def solve_mod(matrix, right_hand_sides, prime):
    """Solve ``matrix * X = right_hand_sides`` by vectorized elimination."""

    matrix = np.asarray(matrix, dtype=np.int64)
    rhs = np.asarray(right_hand_sides, dtype=np.int64)
    if rhs.ndim == 1:
        rhs = rhs[:, None]
    rows, columns = matrix.shape
    if rows != columns or rhs.shape[0] != rows:
        raise ValueError((matrix.shape, rhs.shape))
    augmented = np.concatenate((matrix.copy(), rhs.copy()), axis=1) % prime
    for column in range(columns):
        candidates = np.flatnonzero(augmented[column:, column])
        if not len(candidates):
            raise ZeroDivisionError(f"singular matrix modulo {prime}")
        pivot = column + int(candidates[0])
        if pivot != column:
            augmented[[column, pivot], :] = augmented[[pivot, column], :]
        inverse = pow(int(augmented[column, column]), -1, prime)
        augmented[column, :] = augmented[column, :] * inverse % prime
        factors = augmented[:, column].copy()
        factors[column] = 0
        augmented = (
            augmented - factors[:, None] * augmented[column : column + 1, :]
        ) % prime
    return augmented[:, columns:]


def ns_target_rhs(basis, modes2):
    rhs = np.zeros((len(basis), 1), dtype=np.int64)
    rhs[basis.index(((), tuple(sorted(modes2, reverse=True)))), 0] = 1
    return rhs


def ramond_target_rhs(basis, modes, ground=0):
    rhs = np.zeros((len(basis), 1), dtype=np.int64)
    rhs[
        basis.index(((), tuple(sorted(modes, reverse=True)), int(ground))), 0
    ] = 1
    return rhs


def _sympy_mod(expression, prime, root_i, root_two):
    expression = sp.expand(expression)
    value = expression.subs({sp.I: root_i, sp.sqrt(2): root_two})
    numerator, denominator = map(int, sp.fraction(sp.cancel(value)))
    return numerator % prime * pow(denominator % prime, -1, prime) % prime


def low_level_check(prime=1_000_033):
    """Compare both modular builders entrywise with the reference matrices."""

    root_i, root_two = roots(prime)
    q_rational = sp.Rational(13, 6)
    p_rational = sp.Rational(2, 5)
    q_value = rational_mod(q_rational, prime)
    momentum = rational_mod(p_rational, prime)

    old_q = ns_reference.Q
    ns_reference.Q = q_rational
    try:
        ns_basis, expected_ns = ns_reference.transition(9, p_rational)
    finally:
        ns_reference.Q = old_q
        ns_reference.transition.cache_clear()
    calculated_basis, calculated_ns = ns_transition(
        9, q_value, momentum, root_i, prime
    )
    assert calculated_basis == ns_basis
    for row in range(len(ns_basis)):
        for column in range(len(ns_basis)):
            assert calculated_ns[row, column] == _sympy_mod(
                expected_ns[row, column], prime, root_i, root_two
            )

    old_q = ramond_reference.Q
    ramond_reference.Q = q_rational
    try:
        full_basis, expected_r = ramond_reference.transition(3, -1)
        expected_r = expected_r.subs(ramond_reference.P, p_rational)
    finally:
        ramond_reference.Q = old_q
        ramond_reference.transition.cache_clear()
    block_basis, calculated_r = ramond_transition_block(
        3, 1, -1, q_value, momentum, root_i, root_two, prime
    )
    indices = [full_basis.index(state) for state in block_basis]
    for row, full_row in enumerate(indices):
        for column, full_column in enumerate(indices):
            assert calculated_r[row, column] == _sympy_mod(
                expected_r[full_row, full_column], prime, root_i, root_two
            )
    return True


def benchmark(prime=1_000_033):
    root_i, root_two = roots(prime)
    q_value = rational_mod(sp.Rational(13, 6), prime)
    momentum = rational_mod(sp.Rational(2, 5), prime)
    rows = []

    for label, modes2 in (
        ("v_2", (7, 5, 3, 1)),
        ("v_5/2", (9, 7, 5, 3, 1)),
    ):
        level2 = sum(modes2)
        start = time.perf_counter()
        basis, matrix = ns_transition(
            level2, q_value, momentum, root_i, prime
        )
        built = time.perf_counter() - start
        start = time.perf_counter()
        solution = solve_mod(matrix, ns_target_rhs(basis, modes2), prime)
        solved = time.perf_counter() - start
        assert np.all(matrix @ solution % prime == ns_target_rhs(basis, modes2))
        rows.append((label, level2, len(basis), int(np.count_nonzero(matrix)), built, solved))

    for label, modes in (
        ("W_7/4", (3, 2, 1)),
        ("W_9/4", (4, 3, 2, 1)),
    ):
        level = sum(modes)
        parity = len(modes) % 2
        start = time.perf_counter()
        basis, matrix = ramond_transition_block(
            level,
            parity,
            -1,
            q_value,
            momentum,
            root_i,
            root_two,
            prime,
        )
        built = time.perf_counter() - start
        start = time.perf_counter()
        rhs = ramond_target_rhs(basis, modes)
        solution = solve_mod(matrix, rhs, prime)
        solved = time.perf_counter() - start
        assert np.all(matrix @ solution % prime == rhs)
        rows.append((label, level, len(basis), int(np.count_nonzero(matrix)), built, solved))
    return rows


def _subsets(items):
    for size in range(len(items) + 1):
        yield from combinations(items, size)


def benchmark_complete_branches(prime=1_000_033):
    """Time every distinct physical endpoint in each requested chi string.

    Endpoints of the same grade are put in a multi-column right-hand side,
    so elimination is performed once per grade rather than once per path.
    """

    root_i, root_two = roots(prime)
    q_value = rational_mod(sp.Rational(13, 6), prime)
    momentum = rational_mod(sp.Rational(2, 5), prime)
    reports = []

    for label, modes2 in (
        ("v_2", (7, 5, 3, 1)),
        ("v_5/2", (9, 7, 5, 3, 1)),
    ):
        by_grade = {}
        for subset in _subsets(modes2):
            by_grade.setdefault(sum(subset), []).append(subset)
        build_time = 0.0
        solve_time = 0.0
        components = 0
        maximum_dimension = 0
        for level2, targets in sorted(by_grade.items()):
            start = time.perf_counter()
            basis, matrix = ns_transition(
                level2, q_value, momentum, root_i, prime
            )
            build_time += time.perf_counter() - start
            maximum_dimension = max(maximum_dimension, len(basis))
            rhs = np.concatenate(
                [ns_target_rhs(basis, target) for target in targets], axis=1
            )
            start = time.perf_counter()
            solution = solve_mod(matrix, rhs, prime)
            solve_time += time.perf_counter() - start
            assert np.all(matrix @ solution % prime == rhs)
            components += int(np.count_nonzero(solution))
        reports.append(
            (
                label,
                len(tuple(_subsets(modes2))),
                len(by_grade),
                maximum_dimension,
                components,
                build_time,
                solve_time,
            )
        )

    for label, modes in (
        ("W_7/4", (1, 2, 3)),
        ("W_9/4", (1, 2, 3, 4)),
    ):
        by_grade_and_parity = {}
        for subset in _subsets(modes):
            key = (sum(subset), len(subset) % 2)
            by_grade_and_parity.setdefault(key, []).append(subset)
        build_time = 0.0
        solve_time = 0.0
        components = 0
        maximum_dimension = 0
        for (level, parity), targets in sorted(by_grade_and_parity.items()):
            start = time.perf_counter()
            basis, matrix = ramond_transition_block(
                level,
                parity,
                -1,
                q_value,
                momentum,
                root_i,
                root_two,
                prime,
            )
            build_time += time.perf_counter() - start
            maximum_dimension = max(maximum_dimension, len(basis))
            rhs = np.concatenate(
                [ramond_target_rhs(basis, target) for target in targets], axis=1
            )
            start = time.perf_counter()
            solution = solve_mod(matrix, rhs, prime)
            solve_time += time.perf_counter() - start
            assert np.all(matrix @ solution % prime == rhs)
            components += int(np.count_nonzero(solution))
        reports.append(
            (
                label,
                len(tuple(_subsets(modes))),
                len(by_grade_and_parity),
                maximum_dimension,
                components,
                build_time,
                solve_time,
            )
        )
    return reports


def main():
    print("entrywise low-level comparison:", low_level_check())
    print("label grade dimension nnz build_s solve_s")
    for label, grade, dimension, nonzero, built, solved in benchmark():
        print(
            f"{label:7s} {grade:5d} {dimension:9d} {nonzero:7d} "
            f"{built:8.3f} {solved:8.3f}"
        )
    print("\ncomplete branch: label targets systems max_dim unique_components build_s solve_s")
    for row in benchmark_complete_branches():
        label, paths, systems, dimension, components, built, solved = row
        print(
            f"{label:7s} {paths:5d} {systems:7d} {dimension:7d} "
            f"{components:10d} {built:8.3f} {solved:8.3f}"
        )


if __name__ == "__main__":
    main()
