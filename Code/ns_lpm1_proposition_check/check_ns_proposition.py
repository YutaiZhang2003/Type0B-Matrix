#!/usr/bin/env python3
"""Numerical state-level test of the NS L_{+/-1} branching proposition.

The code constructs the positive NS branch primaries from the ordered
chi strings and tests, at a generic rational (b,P), that

    L_1 v_n

is a level-(4n-3) double-Virasoro descendant of v_{n-1}, and that

    L_-1 v_n

is in the sum of the level-one descendant space of v_n and the
level-(4n-1) descendant space of v_{n-1}.

Half-integral fermion modes are stored in twice-mode units.  All Grassmann
signs, including the tensor sign in U_m=sum_r psi_(m-r)G_r, are applied
before the numerical span calculation.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import numpy as np
import sympy as sp


HERE = Path(__file__).resolve().parent
NS_HELPERS = HERE.parents[1] / "agent_notes"
sys.path.insert(0, str(NS_HELPERS))

import check_ns_branch_norms as exact_ns  # noqa: E402


# (auxiliary fermion modes2, physical bosons, physical fermion modes2)
State = tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
Expression = dict[State, complex]

B_VALUE = sp.Rational(3, 2)
P_VALUE = sp.Rational(2, 5)
Q_VALUE = B_VALUE + 1 / B_VALUE
ZERO_TOLERANCE = 1.0e-14
RANK_TOLERANCE = 1.0e-11


@lru_cache(None)
def as_complex(value) -> complex:
    return complex(sp.N(value, 30))


def add_term(out, state, coefficient) -> None:
    value = out.get(state, 0.0j) + complex(coefficient)
    if abs(value) <= ZERO_TOLERANCE:
        out.pop(state, None)
    else:
        out[state] = value


def linear_combination(*terms: tuple[complex, Expression]) -> Expression:
    answer: Expression = {}
    for coefficient, expression in terms:
        for state, value in expression.items():
            add_term(answer, state, coefficient * value)
    return answer


def apply_to_expression(action, expression: Expression) -> Expression:
    answer: Expression = {}
    for state, outer in expression.items():
        for final, inner in action(state).items():
            add_term(answer, final, outer * inner)
    return answer


@lru_cache(None)
def partitions(total: int, largest: int | None = None):
    if total == 0:
        return ((),)
    if largest is None or largest > total:
        largest = total
    answer = []
    for first in range(largest, 0, -1):
        for rest in partitions(total - first, first):
            answer.append((first,) + rest)
    return tuple(answer)


def apply_c(mode: int, state):
    bosons, fermions = state
    if mode < 0:
        return (tuple(sorted(bosons + (-mode,), reverse=True)), fermions), 1.0
    count = bosons.count(mode)
    if not count:
        return None, 0.0
    remaining = list(bosons)
    remaining.remove(mode)
    return (tuple(remaining), fermions), float(mode * count)


def apply_fermion(mode2: int, state):
    bosons, fermions = state
    if mode2 < 0:
        created = -mode2
        if created in fermions:
            return None, 0.0
        crossings = sum(existing > created for existing in fermions)
        return (
            bosons,
            tuple(sorted(fermions + (created,), reverse=True)),
        ), float((-1) ** crossings)
    if mode2 not in fermions:
        return None, 0.0
    position = fermions.index(mode2)
    return (
        bosons,
        fermions[:position] + fermions[position + 1 :],
    ), float((-1) ** position)


def apply_auxiliary(mode2: int, modes: tuple[int, ...]):
    if mode2 < 0:
        created = -mode2
        if created in modes:
            return None, 0.0
        crossings = sum(existing > created for existing in modes)
        return tuple(sorted(modes + (created,), reverse=True)), float(
            (-1) ** crossings
        )
    if mode2 not in modes:
        return None, 0.0
    position = modes.index(mode2)
    return modes[:position] + modes[position + 1 :], float((-1) ** position)


def apply_two(first, second, state):
    middle, second_coefficient = second(state)
    if not second_coefficient:
        return None, 0.0
    final, first_coefficient = first(middle)
    if not first_coefficient:
        return None, 0.0
    return final, second_coefficient * first_coefficient


@lru_cache(None)
def apply_physical_l(mode: int, state) -> dict:
    """Apply the physical NS Virasoro mode L_mode."""

    bosons, fermions = state
    answer = {}

    bosonic_indices = set(bosons)
    bosonic_indices.update(mode - occupied for occupied in bosons)
    if mode < 0:
        bosonic_indices.update(range(mode + 1, 0))
    for summation_mode in bosonic_indices:
        if summation_mode in (0, mode):
            continue
        final, coefficient = apply_two(
            lambda current, k=mode - summation_mode: apply_c(k, current),
            lambda current, k=summation_mode: apply_c(k, current),
            state,
        )
        if coefficient:
            add_term(answer, final, coefficient / 2)

    fermionic_indices2 = set(fermions)
    fermionic_indices2.update(2 * mode - occupied for occupied in fermions)
    if mode < 0:
        fermionic_indices2.update(range(2 * mode + 1, 0, 2))
    for summation_mode2 in fermionic_indices2:
        final, coefficient = apply_two(
            lambda current, r2=2 * mode - summation_mode2: apply_fermion(
                r2, current
            ),
            lambda current, r2=summation_mode2: apply_fermion(r2, current),
            state,
        )
        if coefficient:
            add_term(answer, final, summation_mode2 * coefficient / 4)

    final, coefficient = apply_c(mode, state)
    if coefficient:
        add_term(
            answer,
            final,
            as_complex(sp.I * (sp.Rational(mode, 2) * Q_VALUE - P_VALUE))
            * coefficient,
        )
    return answer


@lru_cache(None)
def apply_physical_g(mode2: int, state) -> dict:
    """Apply G_(mode2/2), where mode2 is odd."""

    bosons, fermions = state
    answer = {}
    bosonic_indices = set(bosons)
    bosonic_indices.update((mode2 - occupied) // 2 for occupied in fermions)
    if mode2 < 0:
        bosonic_indices.update(range(mode2 // 2 + 1, 0))
    for summation_mode in bosonic_indices:
        if summation_mode == 0:
            continue
        final, coefficient = apply_two(
            lambda current, k=summation_mode: apply_c(k, current),
            lambda current, r2=mode2 - 2 * summation_mode: apply_fermion(
                r2, current
            ),
            state,
        )
        if coefficient:
            add_term(answer, final, coefficient)

    final, coefficient = apply_fermion(mode2, state)
    if coefficient:
        add_term(
            answer,
            final,
            as_complex(sp.I * (sp.Rational(mode2, 2) * Q_VALUE - P_VALUE))
            * coefficient,
        )
    return answer


def apply_l(mode: int, expression: Expression) -> Expression:
    def action(state: State) -> Expression:
        auxiliary, bosons, fermions = state
        return {
            (auxiliary, final[0], final[1]): coefficient
            for final, coefficient in apply_physical_l(
                mode, (bosons, fermions)
            ).items()
        }

    return apply_to_expression(action, expression)


def apply_lf(mode: int, expression: Expression) -> Expression:
    def action(state: State) -> Expression:
        auxiliary, bosons, fermions = state
        indices2 = set(auxiliary)
        indices2.update(2 * mode - occupied for occupied in auxiliary)
        if mode < 0:
            indices2.update(range(2 * mode + 1, 0, 2))
        answer: Expression = {}
        for summation_mode2 in indices2:
            middle, right = apply_auxiliary(summation_mode2, auxiliary)
            if not right:
                continue
            final, left = apply_auxiliary(2 * mode - summation_mode2, middle)
            if not left:
                continue
            add_term(
                answer,
                (final, bosons, fermions),
                summation_mode2 * right * left / 4,
            )
        return answer

    return apply_to_expression(action, expression)


def auxiliary_level2(modes: tuple[int, ...]) -> int:
    return sum(modes)


def physical_level2(state) -> int:
    return 2 * sum(state[0]) + sum(state[1])


def apply_u(mode: int, expression: Expression) -> Expression:
    """Apply U_mode directly, including the odd tensor-product sign."""

    def action(state: State) -> Expression:
        auxiliary, bosons, fermions = state
        physical = (bosons, fermions)
        lower = 2 * mode - auxiliary_level2(auxiliary)
        upper = physical_level2(physical)
        if lower % 2 == 0:
            lower += 1
        if upper % 2 == 0:
            upper -= 1
        answer: Expression = {}
        for r2 in range(lower, upper + 1, 2):
            auxiliary_final, auxiliary_coefficient = apply_auxiliary(
                2 * mode - r2, auxiliary
            )
            if not auxiliary_coefficient:
                continue
            for physical_final, physical_coefficient in apply_physical_g(
                r2, physical
            ).items():
                add_term(
                    answer,
                    (auxiliary_final, physical_final[0], physical_final[1]),
                    (-1) ** len(auxiliary)
                    * auxiliary_coefficient
                    * physical_coefficient,
                )
        return answer

    return apply_to_expression(action, expression)


def _apply_double_virasoro(
    copy: int, mode: int, expression: Expression
) -> Expression:
    denominator = as_complex(1 / B_VALUE - B_VALUE)
    b = as_complex(B_VALUE)
    physical = apply_l(mode, expression)
    auxiliary = apply_lf(mode, expression)
    mixed = apply_u(mode, expression)
    if copy == 1:
        return linear_combination(
            ((1 / b) / denominator, physical),
            (-(1 / b + 2 * b) / denominator, auxiliary),
            (1 / denominator, mixed),
        )
    if copy == 2:
        return linear_combination(
            (-b / denominator, physical),
            ((b + 2 / b) / denominator, auxiliary),
            (-1 / denominator, mixed),
        )
    raise ValueError("copy must be 1 or 2")


@lru_cache(None)
def double_virasoro_on_state(copy: int, mode: int, state: State):
    return tuple(
        _apply_double_virasoro(copy, mode, {state: 1.0 + 0.0j}).items()
    )


def apply_double_virasoro(copy: int, mode: int, expression: Expression):
    return apply_to_expression(
        lambda state: dict(double_virasoro_on_state(copy, mode, state)), expression
    )


def raw_branch(label: sp.Rational) -> Expression:
    label = sp.Rational(label)
    if label < 0 or 2 * label not in sp.S.Integers:
        raise ValueError("This check uses nonnegative half-integral NS labels.")
    if label == 0:
        return {((), (), ()): 1.0 + 0.0j}
    all_modes2 = tuple(range(int(4 * label - 1), 0, -2))
    answer: Expression = {}
    for physical_count in range(len(all_modes2) + 1):
        for physical_modes2 in combinations(all_modes2, physical_count):
            physical = tuple(sorted(physical_modes2, reverse=True))
            physical_set = set(physical)
            auxiliary = tuple(
                mode for mode in all_modes2 if mode not in physical_set
            )
            crossings = sum(
                physical_mode > auxiliary_mode
                for physical_mode in physical
                for auxiliary_mode in auxiliary
            )
            coefficient = (-1j) ** len(physical) * (-1) ** crossings
            add_term(answer, (auxiliary, (), physical), coefficient)
    return answer


def double_virasoro_descendant(
    primary: Expression,
    first_partition: tuple[int, ...],
    second_partition: tuple[int, ...],
) -> Expression:
    answer = primary
    for mode in reversed(second_partition):
        answer = apply_double_virasoro(2, -mode, answer)
    for mode in reversed(first_partition):
        answer = apply_double_virasoro(1, -mode, answer)
    return answer


def descendant_columns(primary: Expression, level: int):
    answer = []
    for first_level in range(level + 1):
        for first_partition in partitions(first_level):
            for second_partition in partitions(level - first_level):
                answer.append(
                    double_virasoro_descendant(
                        primary, first_partition, second_partition
                    )
                )
    return answer


def max_abs(expression: Expression) -> float:
    return max((abs(value) for value in expression.values()), default=0.0)


def span_residual(target: Expression, columns: list[Expression]):
    keys = sorted(set(target).union(*(set(column) for column in columns)), key=repr)
    matrix = np.zeros((len(keys), len(columns)), dtype=np.complex128)
    vector = np.asarray([target.get(key, 0) for key in keys], dtype=np.complex128)
    for column_index, column in enumerate(columns):
        matrix[:, column_index] = [column.get(key, 0) for key in keys]
    column_norms = np.linalg.norm(matrix, axis=0)
    if np.any(column_norms == 0):
        raise AssertionError("A descendant column vanished at the sample point.")
    normalized = matrix / column_norms
    coefficients, _, rank, singular_values = np.linalg.lstsq(
        normalized, vector, rcond=RANK_TOLERANCE
    )
    residual = normalized @ coefficients - vector
    absolute = float(np.linalg.norm(residual))
    relative = absolute / float(np.linalg.norm(vector))
    original_coefficients = coefficients / column_norms
    smallest = float(singular_values[rank - 1]) if rank else 0.0
    return (
        len(keys),
        int(rank),
        absolute,
        relative,
        smallest,
        original_coefficients,
    )


@dataclass
class CheckResult:
    n: str
    statement: str
    relative_level: int
    rows: int
    columns: int
    rank: int
    absolute_residual: float
    relative_residual: float
    smallest_retained_singular_value: float
    level_one_coefficients: list[str]
    passed: bool


def check_one(n: sp.Rational):
    high = raw_branch(n)
    low = raw_branch(n - 1)

    plus_level = int(4 * n - 3)
    plus_columns = descendant_columns(low, plus_level)
    plus_data = span_residual(apply_l(1, high), plus_columns)

    minus_level = int(4 * n - 1)
    same_branch_columns = [
        double_virasoro_descendant(high, (1,), ()),
        double_virasoro_descendant(high, (), (1,)),
    ]
    minus_columns = same_branch_columns + descendant_columns(low, minus_level)
    minus_data = span_residual(apply_l(-1, high), minus_columns)

    identity = linear_combination(
        (1, apply_l(-1, high)),
        (-1, same_branch_columns[0]),
        (-1, same_branch_columns[1]),
        (1, apply_lf(-1, high)),
    )

    results = []
    for statement, level, columns, data, leading_count in (
        ("L_1 support", plus_level, plus_columns, plus_data, 0),
        ("L_-1 support", minus_level, minus_columns, minus_data, 2),
    ):
        rows, rank, absolute, relative, smallest, coefficients = data
        results.append(
            CheckResult(
                n=str(n),
                statement=statement,
                relative_level=level,
                rows=rows,
                columns=len(columns),
                rank=rank,
                absolute_residual=absolute,
                relative_residual=relative,
                smallest_retained_singular_value=smallest,
                level_one_coefficients=[
                    f"{value.real:.15g}{value.imag:+.15g}j"
                    for value in coefficients[:leading_count]
                ],
                passed=(rank == len(columns) and relative < 1.0e-9),
            )
        )
    return results, max_abs(identity)


def backend_consistency_check() -> float:
    maximum = 0.0
    substitutions = {exact_ns.Q: Q_VALUE, exact_ns.P: P_VALUE}
    for level2 in range(7):
        for state in exact_ns.basis(level2):
            for mode in (-1, -2, -3):
                calculated = apply_physical_l(mode, state)
                exact = exact_ns.apply_L_to_state(mode, state, exact_ns.P)
                exact_numeric = {
                    final: as_complex(value.subs(substitutions))
                    for final, value in exact.items()
                }
                for key in set(calculated) | set(exact_numeric):
                    maximum = max(
                        maximum,
                        abs(calculated.get(key, 0) - exact_numeric.get(key, 0)),
                    )
            for mode2 in (-1, -3, -5):
                calculated = apply_physical_g(mode2, state)
                exact = exact_ns.apply_G_to_state(mode2, state, exact_ns.P)
                exact_numeric = {
                    final: as_complex(value.subs(substitutions))
                    for final, value in exact.items()
                }
                for key in set(calculated) | set(exact_numeric):
                    maximum = max(
                        maximum,
                        abs(calculated.get(key, 0) - exact_numeric.get(key, 0)),
                    )
    return maximum


def highest_weight_check() -> float:
    maximum = 0.0
    for label in (
        sp.Rational(0),
        sp.Rational(1, 2),
        sp.Rational(1),
        sp.Rational(3, 2),
        sp.Rational(2),
        sp.Rational(5, 2),
    ):
        onset = int(2 * label**2)
        primary = raw_branch(label)
        for copy in (1, 2):
            for mode in range(1, onset + 2):
                maximum = max(
                    maximum,
                    max_abs(apply_double_virasoro(copy, mode, primary)),
                )
    return maximum


def run_checks():
    results = []
    identities = []
    for n in (
        sp.Rational(1),
        sp.Rational(3, 2),
        sp.Rational(2),
        sp.Rational(5, 2),
    ):
        pair, identity = check_one(n)
        results.extend(pair)
        identities.append((str(n), identity))
        print(
            f"n={n}: "
            + "; ".join(
                f"{item.statement} level={item.relative_level}, "
                f"rank={item.rank}/{item.columns}, "
                f"relative residual={item.relative_residual:.3e}"
                + (
                    f", level-one coefficients={item.level_one_coefficients}"
                    if item.level_one_coefficients
                    else ""
                )
                for item in pair
            )
            + f"; inverse-identity max residual={identity:.3e}",
            flush=True,
        )
    return results, identities


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path)
    parser.add_argument("--strict", action="store_true")
    arguments = parser.parse_args()

    backend = backend_consistency_check()
    highest = highest_weight_check()
    print(f"negative-mode backend max residual={backend:.3e}")
    print(f"highest-weight max residual={highest:.3e}")
    results, identities = run_checks()
    passed = all(item.passed for item in results)
    payload = {
        "sample": {"b": str(B_VALUE), "P": str(P_VALUE), "Q": str(Q_VALUE)},
        "rank_tolerance": RANK_TOLERANCE,
        "negative_mode_backend_max_residual": backend,
        "highest_weight_max_residual": highest,
        "results": [asdict(item) for item in results],
        "inverse_identity_max_residuals": [
            {"n": n, "max_residual": residual} for n, residual in identities
        ],
        "passed": passed,
    }
    if arguments.json:
        arguments.json.write_text(json.dumps(payload, indent=2) + "\n")
    if arguments.strict and not passed:
        raise SystemExit("At least one NS proposition check failed.")
    print("All requested NS support tests passed." if passed else "A test failed.")


if __name__ == "__main__":
    main()
