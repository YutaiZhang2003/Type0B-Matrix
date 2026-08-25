#!/usr/bin/env python3
"""Exact level-by-level Ramond free-field reflection intertwiner.

The 2013 reflection construction defines the reflected oscillators by

    psi^R_m(P) = R(P)^(-1) psi_m R(P)

and characterizes ``R`` by intertwining the two Feigin--Fuchs
realizations.  A common implementation forms both free-field/PBW
transition matrices and multiplies them.  That is unnecessary.  If
``A_s(X_n;ell)`` is the matrix of a positive Ramond generator from Fock
level ``ell`` to ``ell-n`` in realization ``s=+/-``, the level block of
the reflection map is the unique solution of

    A_-(X_n;ell) R_ell = R_(ell-n) A_+(X_n;ell),  X=L,G, n>0.       (1)

At a generic momentum the stacked matrix on the left has full column
rank.  Equation (1) therefore determines ``R_ell`` from lower levels.
This file implements (1) directly, both symbolically at small levels and
over a prime field at larger levels.  No super-Virasoro descendant state,
Gram matrix, or branching state ``W_n`` is constructed.

The explicit matrix still has the size of a Fock level.  Consequently this
is an exact audit/fallback for the reflected kernel, not the asymptotically
preferred screening-chart algorithm: if ``d_ell`` is one Ramond parity
block, time is O(d_ell^3) and memory O(d_ell^2), with
``d_ell = exp(O(sqrt(ell)))``.  The chart-adapted Coulomb-gas evaluation
described in ``README.md`` avoids forming this matrix altogether.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from pathlib import Path
import math
import sys
import time

import numpy as np
import sympy as sp


ROOT = Path(__file__).resolve().parents[3]
RAMOND_REFERENCE = ROOT / "python" / "ramond_branching_coefficient_check"
PROFILE = ROOT / "python" / "ramond_screening_algorithm" / "profile"
for directory in (RAMOND_REFERENCE, PROFILE):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import check_ramond_branching as reference  # noqa: E402
import modular_transition as modular  # noqa: E402


P = reference.P
Q = reference.Q


def fock_norm(state) -> int:
    """Diagonal oscillator norm in the unnormalised Fock basis."""

    bosons = state[0]
    answer = 1
    for mode, multiplicity in Counter(bosons).items():
        answer *= int(mode) ** multiplicity * math.factorial(multiplicity)
    return answer


@lru_cache(None)
def parity_basis(level: int, parity: int):
    """One conserved total-parity block of a Ramond Fock level."""

    return tuple(
        state
        for state in reference.basis(int(level))
        if (len(state[1]) + state[2]) % 2 == int(parity)
    )


def _symbolic_negative_matrix(level, kind, mode, realization):
    """Matrix of X_-mode from ``level-mode`` to ``level``."""

    high = reference.basis(int(level))
    low = reference.basis(int(level - mode))
    row = {state: index for index, state in enumerate(high)}
    matrix = sp.zeros(len(high), len(low))
    for column, state in enumerate(low):
        if kind == "L":
            expression = reference.apply_L_to_state(
                -int(mode), state, int(realization)
            )
        elif kind == "G":
            expression = reference.apply_G_to_state(
                -int(mode), state, int(realization)
            )
        else:
            raise ValueError(kind)
        for final, coefficient in expression.items():
            matrix[row[final], column] = coefficient
    return matrix


def _bpz(expression):
    """BPZ involution in the repository's analytically continued P-frame."""

    # The 2013 Hermitian momentum is p=i P.  Thus complex conjugation sends
    # i -> -i and P -> -P, while Q is fixed.
    expression = sp.conjugate(expression)
    return expression.xreplace(
        {
            sp.conjugate(P): -P,
            sp.conjugate(Q): Q,
        }
    )


def symbolic_positive_matrix(level, kind, mode, realization):
    """Positive-mode Fock matrix obtained by BPZ from a negative mode."""

    high = reference.basis(int(level))
    low = reference.basis(int(level - mode))
    negative = _symbolic_negative_matrix(level, kind, mode, realization)
    high_metric = sp.diag(*(fock_norm(state) for state in high))
    low_metric = sp.diag(*(fock_norm(state) for state in low))
    return low_metric.inv() * negative.applyfunc(_bpz).T * high_metric


@lru_cache(None)
def symbolic_reflection_block(level: int) -> sp.Matrix:
    """Low-level rational-function block of the reflection recurrence.

    This exact symbolic version is useful for exposing the local reflected
    kernel.  It is deliberately not the high-level engine; expression swell
    makes ``reflection_blocks_mod`` the scalable exact implementation.
    """

    level = int(level)
    if level < 0:
        raise ValueError(level)
    if level == 0:
        return sp.eye(2)

    target = []
    source = []
    for mode in range(1, level + 1):
        for kind in ("L", "G"):
            target.append(
                symbolic_positive_matrix(level, kind, mode, -1)
            )
            source.append(
                symbolic_reflection_block(level - mode)
                * symbolic_positive_matrix(level, kind, mode, +1)
            )
    left = target[0]
    right = source[0]
    for part in target[1:]:
        left = left.col_join(part)
    for part in source[1:]:
        right = right.col_join(part)
    independent_rows = left.T.rref()[1]
    square_left = left[list(independent_rows), :]
    square_right = right[list(independent_rows), :]
    answer = (square_left.inv() * square_right).applyfunc(sp.cancel)
    residual = (left * answer - right).applyfunc(sp.simplify)
    if any(value != 0 for value in residual):
        raise AssertionError(
            f"The level-{level} symbolic reflection equations did not close."
        )
    return answer


def symbolic_level_one() -> sp.Matrix:
    """Solve (1) at level one without a PBW transition matrix."""

    return symbolic_reflection_block(1)


def _mod_negative_matrix(
    level,
    parity,
    kind,
    mode,
    realization,
    q_value,
    momentum,
    root_i,
    root_two,
    prime,
):
    """One parity block of a negative generator over GF(prime)."""

    lower_parity = int(parity) if kind == "L" else 1 - int(parity)
    high = parity_basis(int(level), int(parity))
    low = parity_basis(int(level - mode), lower_parity)
    row = {state: index for index, state in enumerate(high)}
    matrix = np.zeros((len(high), len(low)), dtype=np.int64)
    for column, state in enumerate(low):
        if kind == "L":
            expression = modular.apply_r_L(
                -int(mode),
                state,
                int(realization),
                q_value,
                momentum,
                root_i,
                root_two,
                prime,
            )
        else:
            expression = modular.apply_r_G(
                -int(mode),
                state,
                int(realization),
                q_value,
                momentum,
                root_i,
                root_two,
                prime,
            )
        for final, coefficient in expression.items():
            matrix[row[final], column] = int(coefficient) % prime
    return low, high, matrix


def modular_positive_matrix(
    level,
    parity,
    kind,
    mode,
    realization,
    q_value,
    momentum,
    root_i,
    root_two,
    prime,
):
    """Positive generator matrix, using i->-i and P->-P under BPZ."""

    low, high, bpz_negative = _mod_negative_matrix(
        level,
        parity,
        kind,
        mode,
        realization,
        q_value,
        (-momentum) % prime,
        (-root_i) % prime,
        root_two,
        prime,
    )
    low_norm = np.array([fock_norm(state) % prime for state in low], dtype=np.int64)
    high_norm = np.array(
        [fock_norm(state) % prime for state in high], dtype=np.int64
    )
    inverse_low = np.array(
        [pow(int(value), -1, prime) for value in low_norm], dtype=np.int64
    )
    matrix = bpz_negative.T.copy()
    matrix = matrix * high_norm[None, :] % prime
    matrix = inverse_low[:, None] * matrix % prime
    return low, high, matrix


def independent_rows_mod(matrix, prime):
    """Indices of a maximal independent row set over GF(prime)."""

    transposed = np.asarray(matrix, dtype=np.int64).T.copy() % prime
    rows, columns = transposed.shape
    pivot_columns = []
    pivot_row = 0
    for column in range(columns):
        candidates = np.flatnonzero(transposed[pivot_row:, column])
        if not len(candidates):
            continue
        selected = pivot_row + int(candidates[0])
        if selected != pivot_row:
            transposed[[pivot_row, selected], :] = transposed[
                [selected, pivot_row], :
            ]
        inverse = pow(int(transposed[pivot_row, column]), -1, prime)
        transposed[pivot_row, :] = transposed[pivot_row, :] * inverse % prime
        for row in range(rows):
            if row == pivot_row:
                continue
            factor = int(transposed[row, column])
            if factor:
                transposed[row, :] = (
                    transposed[row, :] - factor * transposed[pivot_row, :]
                ) % prime
        pivot_columns.append(column)
        pivot_row += 1
        if pivot_row == rows:
            break
    return tuple(pivot_columns)


def reflection_blocks_mod(
    maximum_level,
    q_value,
    momentum,
    root_i,
    root_two,
    prime,
):
    """Return all R_(level,parity) through ``maximum_level`` over GF(p)."""

    blocks = {
        (0, 0): np.ones((1, 1), dtype=np.int64),
        (0, 1): np.ones((1, 1), dtype=np.int64),
    }
    for level in range(1, int(maximum_level) + 1):
        for parity in (0, 1):
            left_parts = []
            right_parts = []
            for mode in range(1, level + 1):
                for kind in ("L", "G"):
                    lower_parity = parity if kind == "L" else 1 - parity
                    _, high, left = modular_positive_matrix(
                        level,
                        parity,
                        kind,
                        mode,
                        -1,
                        q_value,
                        momentum,
                        root_i,
                        root_two,
                        prime,
                    )
                    _, _, source = modular_positive_matrix(
                        level,
                        parity,
                        kind,
                        mode,
                        +1,
                        q_value,
                        momentum,
                        root_i,
                        root_two,
                        prime,
                    )
                    right = blocks[(level - mode, lower_parity)] @ source % prime
                    left_parts.append(left)
                    right_parts.append(right)
            stacked_left = np.concatenate(left_parts, axis=0) % prime
            stacked_right = np.concatenate(right_parts, axis=0) % prime
            pivots = independent_rows_mod(stacked_left, prime)
            if len(pivots) != len(high):
                raise ZeroDivisionError(
                    f"reflection equations lost rank at level={level}, "
                    f"parity={parity}: {len(pivots)}/{len(high)}"
                )
            square_left = stacked_left[list(pivots), :]
            square_right = stacked_right[list(pivots), :]
            block = modular.solve_mod(square_left, square_right, prime)
            if np.any((stacked_left @ block - stacked_right) % prime):
                raise AssertionError((level, parity, "intertwining residual"))
            blocks[(level, parity)] = block
    return blocks


def reflect_fock_expression_mod(expression, blocks, prime):
    """Apply precomputed plus-to-minus reflection blocks to a sparse vector.

    ``expression`` is a mapping

        (boson_partition, strict_fermion_partition, ground_index) -> coefficient

    in the plus free-field chart.  Coefficients and the returned values are
    residues modulo ``prime``.  States may have different levels and total
    parities; the function groups them before applying the corresponding
    ``blocks[(level, parity)]``.  Within each block both input and output use
    the public ``parity_basis(level, parity)`` ordering.

    This is the interface needed by a chi-string evaluator.  Its output is
    still a free-oscillator expression: in particular, reflected fermion
    strings generally contain bosonic modes and must subsequently be sent to
    the Heisenberg/Pfaffian--Selberg contraction, not to a scalar covariance.
    """

    grouped = {}
    for state, coefficient in expression.items():
        bosons, fermions, ground = state
        level = sum(bosons) + sum(fermions)
        parity = (len(fermions) + int(ground)) % 2
        basis = parity_basis(level, parity)
        row = {item: index for index, item in enumerate(basis)}
        key = (level, parity)
        if key not in blocks:
            raise KeyError(f"missing reflection block {key}")
        vector = grouped.setdefault(
            key, np.zeros((len(basis), 1), dtype=np.int64)
        )
        vector[row[state], 0] = (
            int(vector[row[state], 0]) + int(coefficient)
        ) % prime

    answer = {}
    for key, vector in grouped.items():
        level, parity = key
        basis = parity_basis(level, parity)
        reflected = blocks[key] @ vector % prime
        for state, coefficient in zip(basis, reflected[:, 0]):
            coefficient = int(coefficient) % prime
            if coefficient:
                answer[state] = (answer.get(state, 0) + coefficient) % prime
                if answer[state] == 0:
                    del answer[state]
    return answer


def audit_level_one_symbolic():
    """Independent comparison with S_-(P) S_+(P)^(-1)."""

    calculated = symbolic_level_one()
    _, minus = reference.transition(1, -1)
    _, plus = reference.transition(1, +1)
    expected = minus * plus.inv()
    residual = (calculated - expected).applyfunc(sp.factor)
    if any(value != 0 for value in residual):
        raise AssertionError(f"level-one reflection mismatch:\n{residual}")
    return calculated


def audit_level_two_symbolic():
    """Check the level-two current/fermion column quoted in README.md."""

    calculated = symbolic_reflection_block(2)
    ordered = reference.basis(2)
    source = ordered.index(((), (2,), 0))
    denominator_one = 4 * P**2 - 6 * P * Q + 2 * Q**2 + 1
    denominator_two = 4 * P**2 - 10 * P * Q + 4 * Q**2 + 9
    denominator = denominator_one * denominator_two
    expected = {
        ((2,), (), 1): (
            4
            * sp.sqrt(2)
            * sp.I
            * Q
            * (4 * P**2 - 4 * P * Q + Q**2 + 3)
            / denominator
        ),
        ((1, 1), (), 1): (
            4 * sp.sqrt(2) * Q * (4 * P + Q) / denominator
        ),
        ((1,), (1,), 0): (
            12 * sp.I * Q * (4 * P**2 - 2 * P * Q + 1) / denominator
        ),
        ((), (2,), 0): -(
            16 * P**4
            - 44 * P**2 * Q**2
            + 40 * P**2
            + 36 * P * Q**3
            + 16 * P * Q
            - 8 * Q**4
            - 18 * Q**2
            + 9
        )
        / denominator,
    }
    for row, state in enumerate(ordered):
        residual = sp.simplify(
            sp.cancel(
                calculated[row, source]
                - expected.get(state, sp.Integer(0))
            )
        )
        if residual != 0:
            raise AssertionError((state, "level-two reflected psi", residual))
    return calculated[:, source]


def audit_modular(maximum_level=3, prime=1_000_033):
    """Compare the recurrence with transition matrices only as an audit."""

    root_i, root_two = modular.roots(prime)
    q_value = modular.rational_mod(sp.Rational(13, 6), prime)
    momentum = modular.rational_mod(sp.Rational(2, 5), prime)
    blocks = reflection_blocks_mod(
        maximum_level,
        q_value,
        momentum,
        root_i,
        root_two,
        prime,
    )
    for level in range(maximum_level + 1):
        for parity in (0, 1):
            basis_minus, minus = modular.ramond_transition_sector(
                level,
                parity,
                -1,
                q_value,
                momentum,
                root_i,
                root_two,
                prime,
            )
            basis_plus, plus = modular.ramond_transition_sector(
                level,
                parity,
                +1,
                q_value,
                momentum,
                root_i,
                root_two,
                prime,
            )
            if basis_minus != parity_basis(level, parity) or basis_plus != basis_minus:
                raise AssertionError("basis ordering changed")
            expected = modular.solve_mod(plus.T, minus.T, prime).T
            if np.any((blocks[(level, parity)] - expected) % prime):
                raise AssertionError((level, parity, "transition audit"))

            # Also audit the public sparse-vector interface and its declared
            # basis orientation on every column of the complete block.
            for column, state in enumerate(basis_minus):
                sparse = reflect_fock_expression_mod(
                    {state: 1}, blocks, prime
                )
                expected_sparse = {
                    final: int(expected[row, column]) % prime
                    for row, final in enumerate(basis_minus)
                    if int(expected[row, column]) % prime
                }
                if sparse != expected_sparse:
                    raise AssertionError(
                        (level, parity, column, "sparse reflection API")
                    )
    return blocks


def benchmark(maximum_level=6, prime=1_000_033):
    root_i, root_two = modular.roots(prime)
    q_value = modular.rational_mod(sp.Rational(13, 6), prime)
    momentum = modular.rational_mod(sp.Rational(2, 5), prime)
    start = time.perf_counter()
    blocks = reflection_blocks_mod(
        maximum_level,
        q_value,
        momentum,
        root_i,
        root_two,
        prime,
    )
    elapsed = time.perf_counter() - start
    dimensions = tuple(len(parity_basis(maximum_level, p)) for p in (0, 1))
    return elapsed, dimensions, blocks


def main():
    level_one = audit_level_one_symbolic()
    print("symbolic level-one reflection: exact")
    print("level-one denominator:", sp.factor(sp.denom(level_one[0, 0])))
    audit_level_two_symbolic()
    print("symbolic level-two reflected psi/current column: exact")
    audit_modular(3)
    print("modular levels 0..3 against S_- S_+^-1: exact")
    elapsed, dimensions, _ = benchmark(6)
    print(
        "W_7/4 endpoint level=6 parity dimensions="
        f"{dimensions} recurrence_time={elapsed:.3f}s"
    )


if __name__ == "__main__":
    main()
