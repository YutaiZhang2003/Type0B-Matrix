#!/usr/bin/env python3
"""Numerical state-level test of the proposition around SCblock Eq. (7.13).

At a generic rational value of ``b`` and ``P``, the script constructs the raw
Ramond branch states from their ordered chi strings in the free-field Fock
space.  It then checks, for both parity copies, that

    L_1 v_n^alpha

belongs to the level-(4 n - 3) Vir x Vir descendant space of
``v_(n-1)^alpha``, and that

    L_-1 v_n^alpha

belongs to the sum of the level-one descendant space of ``v_n^alpha`` and
the level-(4 n - 1) descendant space of the lower branch.  It separately
tests the stronger displayed equation, which fixes the coefficients of
``L_-1^(1) v_n^alpha`` and ``L_-1^(2) v_n^alpha`` to one.

The oscillator actions are evaluated directly.  In particular, ``U_m`` is
implemented from ``sum_r psi_(m-r) G_r`` with the graded tensor-product sign.
No branching proposition is used while constructing either tested vector.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import sympy as sp


HERE = Path(__file__).resolve().parent
RAMOND_HELPERS = HERE.parent / "ramond_branching_coefficient_check"
sys.path.insert(0, str(RAMOND_HELPERS))

import check_ramond_branching as branch  # noqa: E402


State = tuple[tuple[int, ...], int, tuple[int, ...], tuple[int, ...], int]
Expression = dict[State, complex]


B_VALUE = sp.Rational(3, 2)
P_VALUE = sp.Rational(2, 5)
Q_VALUE = B_VALUE + 1 / B_VALUE
REALIZATION = -1
ZERO_TOLERANCE = 1.0e-14
RANK_TOLERANCE = 1.0e-11


@lru_cache(None)
def as_complex(value) -> complex:
    return complex(sp.N(value, 30))


def add_term(out: Expression, state: State, coefficient: complex) -> None:
    value = out.get(state, 0.0j) + complex(coefficient)
    if abs(value) <= ZERO_TOLERANCE:
        out.pop(state, None)
    else:
        out[state] = value


def scaled(expression: Expression, coefficient: complex) -> Expression:
    return {
        state: coefficient * value
        for state, value in expression.items()
        if abs(coefficient * value) > ZERO_TOLERANCE
    }


def linear_combination(*terms: tuple[complex, Expression]) -> Expression:
    answer: Expression = {}
    for coefficient, expression in terms:
        for state, value in expression.items():
            add_term(answer, state, coefficient * value)
    return answer


def apply_to_expression(action, expression: Expression) -> Expression:
    answer: Expression = {}
    for state, outer_coefficient in expression.items():
        for final, inner_coefficient in action(state).items():
            add_term(answer, final, outer_coefficient * inner_coefficient)
    return answer


def physical_level(state) -> int:
    return sum(state[0]) + sum(state[1])


def auxiliary_level(state) -> int:
    return sum(state[0])


@lru_cache(None)
def apply_physical_l(mode: int, state) -> dict:
    """Apply the free-field physical SCA Virasoro mode L_mode."""

    bosons, fermions, _ = state
    answer = {}

    # In (1/2) sum_{k != 0,mode} c_k c_(mode-k), a term can act only if
    # an annihilator meets an occupied mode, or if both modes are creators.
    bosonic_indices = set(bosons)
    bosonic_indices.update(mode - occupied for occupied in bosons)
    if mode < 0:
        bosonic_indices.update(range(mode + 1, 0))
    for summation_mode in bosonic_indices:
        if summation_mode in (0, mode):
            continue
        final, coefficient = branch.apply_two(
            lambda current, k=mode - summation_mode: branch.apply_c(k, current),
            lambda current, k=summation_mode: branch.apply_c(k, current),
            state,
        )
        if coefficient:
            add_term(
                answer,
                final,
                as_complex(sp.Rational(1, 2) * coefficient),
            )

    # The same finite-support argument applies to
    # (1/2) sum_r r eta_(mode-r) eta_r.
    fermionic_indices = set(fermions)
    fermionic_indices.update(mode - occupied for occupied in fermions)
    if mode < 0:
        fermionic_indices.update(range(mode, 1))
    for summation_mode in fermionic_indices:
        final, coefficient = branch.apply_two(
            lambda current, r=mode - summation_mode: branch.apply_fermion(
                r, current, REALIZATION
            ),
            lambda current, r=summation_mode: branch.apply_fermion(
                r, current, REALIZATION
            ),
            state,
        )
        if coefficient:
            add_term(
                answer,
                final,
                as_complex(sp.Rational(summation_mode, 2) * coefficient),
            )

    final, coefficient = branch.apply_c(mode, state)
    if coefficient:
        add_term(
            answer,
            final,
            as_complex(
                sp.I
                * sp.Rational(1, 2)
                * (Q_VALUE * mode + 2 * REALIZATION * P_VALUE)
                * coefficient
            ),
        )
    return answer


@lru_cache(None)
def apply_physical_g(mode: int, state) -> dict:
    """Apply the free-field physical supercurrent mode G_mode."""

    bosons, fermions, _ = state
    answer = {}

    # For c_k eta_(mode-k), either eta annihilates an occupied fermion,
    # eta is the zero mode, c annihilates an occupied boson, or both modes
    # create.  These four cases give the following complete finite set.
    bosonic_indices = set(bosons)
    bosonic_indices.update(mode - occupied for occupied in fermions)
    if mode != 0:
        bosonic_indices.add(mode)
    if mode < 0:
        bosonic_indices.update(range(mode, 0))
    for summation_mode in bosonic_indices:
        if summation_mode == 0:
            continue
        final, coefficient = branch.apply_two(
            lambda current, k=summation_mode: branch.apply_c(k, current),
            lambda current, r=mode - summation_mode: branch.apply_fermion(
                r, current, REALIZATION
            ),
            state,
        )
        if coefficient:
            add_term(answer, final, as_complex(coefficient))

    final, coefficient = branch.apply_fermion(mode, state, REALIZATION)
    if coefficient:
        add_term(
            answer,
            final,
            as_complex(
                sp.I
                * (Q_VALUE * mode + REALIZATION * P_VALUE)
                * coefficient
            ),
        )
    return answer


def apply_l(mode: int, expression: Expression) -> Expression:
    """Apply 1 tensor L_mode; this even operator has no tensor sign."""

    def action(state: State) -> Expression:
        auxiliary_modes, auxiliary_ground, bosons, fermions, physical_ground = state
        answer: Expression = {}
        for final, coefficient in apply_physical_l(
            mode, (bosons, fermions, physical_ground)
        ).items():
            answer[
                (
                    auxiliary_modes,
                    auxiliary_ground,
                    final[0],
                    final[1],
                    final[2],
                )
            ] = coefficient
        return answer

    return apply_to_expression(action, expression)


def apply_lf(mode: int, expression: Expression) -> Expression:
    """Apply the auxiliary free-fermion Virasoro mode L_mode^F."""

    def action(state: State) -> Expression:
        modes, ground, bosons, fermions, physical_ground = state
        indices = set(modes)
        indices.update(mode - occupied for occupied in modes)
        if mode < 0:
            indices.update(range(mode, 1))
        answer: Expression = {}
        for summation_mode in indices:
            middle, coefficient_right = branch.apply_auxiliary(
                summation_mode, (modes, ground)
            )
            if not coefficient_right:
                continue
            final, coefficient_left = branch.apply_auxiliary(
                mode - summation_mode, middle
            )
            if not coefficient_left:
                continue
            coefficient = (
                0.5
                * summation_mode
                * as_complex(coefficient_right)
                * as_complex(coefficient_left)
            )
            add_term(
                answer,
                (final[0], final[1], bosons, fermions, physical_ground),
                coefficient,
            )
        return answer

    return apply_to_expression(action, expression)


def apply_u(mode: int, expression: Expression) -> Expression:
    """Apply U_mode=sum_r psi_(mode-r) G_r with its Grassmann sign."""

    def action(state: State) -> Expression:
        modes, ground, bosons, fermions, physical_ground = state
        aux_state = (modes, ground)
        phys_state = (bosons, fermions, physical_ground)
        aux_parity = (len(modes) + ground) % 2

        # G_r lowers physical level by r and psi_(mode-r) lowers auxiliary
        # level by mode-r.  Nonnegative intermediate levels therefore force
        # mode-aux_level <= r <= physical_level.
        lower = mode - auxiliary_level(aux_state)
        upper = physical_level(phys_state)
        answer: Expression = {}
        for r in range(lower, upper + 1):
            auxiliary_final, auxiliary_coefficient = branch.apply_auxiliary(
                mode - r, aux_state
            )
            if not auxiliary_coefficient:
                continue
            for physical_final, physical_coefficient in apply_physical_g(
                r, phys_state
            ).items():
                coefficient = (
                    (-1) ** aux_parity
                    * as_complex(auxiliary_coefficient)
                    * physical_coefficient
                )
                add_term(
                    answer,
                    (
                        auxiliary_final[0],
                        auxiliary_final[1],
                        physical_final[0],
                        physical_final[1],
                        physical_final[2],
                    ),
                    coefficient,
                )
        return answer

    return apply_to_expression(action, expression)


def _apply_double_virasoro(
    copy: int, mode: int, expression: Expression
) -> Expression:
    denominator = as_complex(1 / B_VALUE - B_VALUE)
    physical = apply_l(mode, expression)
    auxiliary = apply_lf(mode, expression)
    mixed = apply_u(mode, expression)
    b = as_complex(B_VALUE)
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
    """Cached elementary double-Virasoro action on one Fock basis state."""

    answer = _apply_double_virasoro(copy, mode, {state: 1.0 + 0.0j})
    return tuple(answer.items())


def apply_double_virasoro(copy: int, mode: int, expression: Expression) -> Expression:
    return apply_to_expression(
        lambda state: dict(double_virasoro_on_state(copy, mode, state)), expression
    )


def raw_branch(branch_label: sp.Rational, parity: int) -> Expression:
    """Return the ordered chi-string state in the common oscillator basis."""

    _, _, expression = branch.expand_chi_string(branch_label, parity)
    return {state: as_complex(value) for state, value in expression.items()}


def double_virasoro_descendant(
    primary: Expression, first_partition: tuple[int, ...], second_partition: tuple[int, ...]
) -> Expression:
    """Construct L_-A^(1) L_-B^(2) primary in displayed operator order."""

    answer = primary
    for mode in reversed(second_partition):
        answer = apply_double_virasoro(2, -mode, answer)
    for mode in reversed(first_partition):
        answer = apply_double_virasoro(1, -mode, answer)
    return answer


def descendant_columns(primary: Expression, relative_level: int) -> list[Expression]:
    columns = []
    for first_level in range(relative_level + 1):
        for first_partition in branch.partitions(first_level):
            for second_partition in branch.partitions(relative_level - first_level):
                columns.append(
                    double_virasoro_descendant(
                        primary, first_partition, second_partition
                    )
                )
    return columns


def vector_norm(expression: Expression) -> float:
    return float(np.linalg.norm(np.asarray(list(expression.values()), dtype=np.complex128)))


def max_abs(expression: Expression) -> float:
    return max((abs(value) for value in expression.values()), default=0.0)


def span_residual(target: Expression, columns: list[Expression]):
    keys = sorted(
        set(target).union(*(set(column) for column in columns)), key=repr
    )
    matrix = np.zeros((len(keys), len(columns)), dtype=np.complex128)
    vector = np.asarray([target.get(key, 0.0j) for key in keys], dtype=np.complex128)
    for column_index, column in enumerate(columns):
        matrix[:, column_index] = [column.get(key, 0.0j) for key in keys]

    column_norms = np.linalg.norm(matrix, axis=0)
    if np.any(column_norms == 0):
        raise AssertionError("A claimed descendant column vanished at the sample point.")
    normalized = matrix / column_norms
    coefficients, _, rank, singular_values = np.linalg.lstsq(
        normalized, vector, rcond=RANK_TOLERANCE
    )
    residual = normalized @ coefficients - vector
    absolute = float(np.linalg.norm(residual))
    relative = absolute / float(np.linalg.norm(vector))
    smallest = float(singular_values[rank - 1]) if rank else 0.0
    original_coefficients = coefficients / column_norms
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
    parity: int
    statement: str
    relative_level: int
    rows: int
    columns: int
    rank: int
    absolute_residual: float
    relative_residual: float
    smallest_retained_singular_value: float
    leading_coefficients: list[str]
    passed: bool


def check_one(n: sp.Rational, parity: int) -> tuple[list[CheckResult], float]:
    high = raw_branch(n, parity)
    low = raw_branch(n - 1, parity)

    # First line of the proposition.
    l_plus_target = apply_l(1, high)
    plus_level = int(4 * n - 3)
    plus_columns = descendant_columns(low, plus_level)
    plus_data = span_residual(l_plus_target, plus_columns)

    # Second line of the proposition.  The first two terms have fixed unit
    # coefficients, so only their residual may be fitted to the lower branch.
    l_minus = apply_l(-1, high)
    first = apply_double_virasoro(1, -1, high)
    second = apply_double_virasoro(2, -1, high)
    minus_target = linear_combination((1, l_minus), (-1, first), (-1, second))
    minus_level = int(4 * n - 1)
    minus_columns = descendant_columns(low, minus_level)
    displayed_minus_data = span_residual(minus_target, minus_columns)

    # The prose immediately before the display allows arbitrary level-one
    # coefficients in the v_n module.  Test that weaker support statement
    # independently of the unit coefficients written in the display.
    same_branch_columns = [
        double_virasoro_descendant(high, (1,), ()),
        double_virasoro_descendant(high, (), (1,)),
    ]
    support_minus_columns = same_branch_columns + minus_columns
    support_minus_data = span_residual(l_minus, support_minus_columns)

    # Independent oscillator check of L_-1=L_-1^(1)+L_-1^(2)-L_-1^F.
    identity_residual = linear_combination(
        (1, l_minus),
        (-1, first),
        (-1, second),
        (1, apply_lf(-1, high)),
    )

    results = []
    for statement, level, column_count, data, leading_count in (
        ("L_1 support", plus_level, len(plus_columns), plus_data, 0),
        (
            "L_-1 support with free level-one coefficients",
            minus_level,
            len(support_minus_columns),
            support_minus_data,
            2,
        ),
        (
            "displayed L_-1 formula with unit coefficients",
            minus_level,
            len(minus_columns),
            displayed_minus_data,
            0,
        ),
    ):
        rows, rank, absolute, relative, smallest, coefficients = data
        results.append(
            CheckResult(
                n=str(n),
                parity=parity,
                statement=statement,
                relative_level=level,
                rows=rows,
                columns=column_count,
                rank=rank,
                absolute_residual=absolute,
                relative_residual=relative,
                smallest_retained_singular_value=smallest,
                leading_coefficients=[
                    f"{value.real:.15g}{value.imag:+.15g}j"
                    for value in coefficients[:leading_count]
                ],
                passed=(rank == column_count and relative < 1.0e-10),
            )
        )
    return results, max_abs(identity_residual)


def backend_consistency_check() -> float:
    """Compare the generalized negative-mode code with the older exact code."""

    maximum = 0.0
    substitutions = {branch.Q: Q_VALUE, branch.P: P_VALUE}
    for level in range(4):
        for physical_state in branch.basis(level):
            for mode in (-1, -2):
                comparisons = (
                    (
                        apply_physical_l(mode, physical_state),
                        branch.apply_L_to_state(
                            mode,
                            physical_state,
                            -1,
                            momentum=branch.P,
                        ),
                    ),
                    (
                        apply_physical_g(mode, physical_state),
                        branch.apply_G_to_state(
                            mode,
                            physical_state,
                            -1,
                            momentum=branch.P,
                        ),
                    ),
                )
                for calculated, exact in comparisons:
                    exact_numeric = {
                        state: as_complex(value.subs(substitutions))
                        for state, value in exact.items()
                    }
                    keys = set(calculated) | set(exact_numeric)
                    maximum = max(
                        maximum,
                        *(abs(calculated.get(key, 0) - exact_numeric.get(key, 0)) for key in keys),
                    )
    return maximum


def highest_weight_check() -> float:
    """Check directly that every branch state used is Vir x Vir primary."""

    labels = {
        sp.Rational(3, 4),
        sp.Rational(5, 4),
        sp.Rational(7, 4),
        sp.Rational(9, 4),
        -sp.Rational(1, 4),
        sp.Rational(1, 4),
    }
    maximum = 0.0
    for label in labels:
        onset = int(2 * label**2 - sp.Rational(1, 8))
        for parity in (0, 1):
            primary = raw_branch(label, parity)
            for copy in (1, 2):
                for mode in range(1, onset + 2):
                    maximum = max(
                        maximum,
                        max_abs(apply_double_virasoro(copy, mode, primary)),
                    )
    return maximum


def run_checks():
    results = []
    identity_residuals = []
    for n in (
        sp.Rational(3, 4),
        sp.Rational(5, 4),
        sp.Rational(7, 4),
        sp.Rational(9, 4),
    ):
        for parity in (0, 1):
            pair, identity_residual = check_one(n, parity)
            results.extend(pair)
            identity_residuals.append((str(n), parity, identity_residual))
            print(
                f"n={n}, alpha={parity}: "
                + "; ".join(
                    f"{item.statement} level={item.relative_level}, "
                    f"rank={item.rank}/{item.columns}, "
                    f"relative residual={item.relative_residual:.3e}"
                    + (
                        f", leading coefficients={item.leading_coefficients}"
                        if item.leading_coefficients
                        else ""
                    )
                    for item in pair
                )
                + f"; inverse-identity max residual={identity_residual:.3e}",
                flush=True,
            )
    return results, identity_residuals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json",
        type=Path,
        help="optionally save the machine-readable results to this path",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit nonzero if any of the three tested statements fails",
    )
    arguments = parser.parse_args()

    backend_residual = backend_consistency_check()
    highest_weight_residual = highest_weight_check()
    print(f"negative-mode backend max residual={backend_residual:.3e}")
    print(f"highest-weight max residual={highest_weight_residual:.3e}")
    results, identity_residuals = run_checks()
    passed = all(item.passed for item in results)
    payload = {
        "sample": {"b": str(B_VALUE), "P": str(P_VALUE), "Q": str(Q_VALUE)},
        "rank_tolerance": RANK_TOLERANCE,
        "negative_mode_backend_max_residual": backend_residual,
        "highest_weight_max_residual": highest_weight_residual,
        "results": [asdict(item) for item in results],
        "inverse_identity_max_residuals": [
            {"n": n, "parity": parity, "max_residual": residual}
            for n, parity, residual in identity_residuals
        ],
        "passed": passed,
    }
    if arguments.json:
        arguments.json.write_text(json.dumps(payload, indent=2) + "\n")
    if arguments.strict and not passed:
        raise SystemExit("At least one proposition check failed.")
    if passed:
        print("All requested state-level span tests passed.")
    else:
        print("At least one tested statement failed; see the residuals above.")


if __name__ == "__main__":
    main()
