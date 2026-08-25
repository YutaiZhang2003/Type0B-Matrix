#!/usr/bin/env python3
"""Direct numerical test of the reflected Ramond L_{+/-1} proposition.

The negative-label branch states are kept in their native ``+1`` free-field
realization.  Thus the physical SCA modes and both embedded Virasoro algebras
act in the same oscillator chart as the states.  Only the boundary state
``v_(1/4)^alpha`` needed at ``n=-3/4`` is transported through the abstract
Ramond PBW basis into that chart.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import sympy as sp


HERE = Path(__file__).resolve().parent
BACKEND_PATH = HERE / "check_proposition_7_13.py"


def load_backend():
    specification = importlib.util.spec_from_file_location(
        "ramond_negative_lpm1_backend", BACKEND_PATH
    )
    if specification is None or specification.loader is None:
        raise ImportError(BACKEND_PATH)
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


backend = load_backend()


def configure(realization: int, b_value, p_value) -> None:
    """Choose one free-field chart and clear every parameter-dependent cache."""

    backend.REALIZATION = realization
    backend.B_VALUE = sp.sympify(b_value)
    backend.P_VALUE = sp.sympify(p_value)
    backend.Q_VALUE = backend.B_VALUE + 1 / backend.B_VALUE
    backend.as_complex.cache_clear()
    backend.apply_physical_l.cache_clear()
    backend.apply_physical_g.cache_clear()
    backend.double_virasoro_on_state.cache_clear()


def branch_in_configured_realization(label, parity):
    """Return ``v_label^parity`` in the currently configured oscillator chart."""

    label = sp.Rational(label)
    native_realization, _, raw_expression = backend.branch.expand_chi_string(
        label, parity
    )
    if native_realization == backend.REALIZATION:
        return {
            state: backend.as_complex(coefficient)
            for state, coefficient in raw_expression.items()
        }

    substitutions = {
        backend.branch.Q: backend.Q_VALUE,
        backend.branch.P: backend.P_VALUE,
    }
    _, sectors = backend.branch.branch_in_abstract_basis(
        label, parity, substitutions=substitutions
    )
    answer = {}
    for auxiliary_state, (level, ordered_basis, abstract_coefficients) in sectors.items():
        fixed_basis, fixed_transition = backend.branch.transition(
            level, backend.REALIZATION
        )
        if fixed_basis != ordered_basis:
            raise AssertionError("The two Ramond PBW bases disagree.")
        fock_coefficients = fixed_transition.subs(
            substitutions, simultaneous=True
        ) * abstract_coefficients
        for state, coefficient in zip(ordered_basis, fock_coefficients):
            if coefficient != 0:
                answer[
                    (
                        auxiliary_state[0],
                        auxiliary_state[1],
                        state[0],
                        state[1],
                        state[2],
                    )
                ] = backend.as_complex(coefficient)
    return answer


def fit_actions(n, parity):
    """Fit both physical Virasoro actions to the proposed descendant spaces."""

    high = branch_in_configured_realization(n, parity)
    toward_zero = branch_in_configured_realization(n - sp.sign(n), parity)

    plus_level = int(4 * abs(n) - 3)
    plus_columns = backend.descendant_columns(toward_zero, plus_level)
    plus_target = backend.apply_l(1, high)
    plus_data = backend.span_residual(plus_target, plus_columns)

    toward_level = int(4 * abs(n) - 1)
    toward_columns = backend.descendant_columns(toward_zero, toward_level)
    same_columns = [
        backend.double_virasoro_descendant(high, (1,), ()),
        backend.double_virasoro_descendant(high, (), (1,)),
    ]
    minus_columns = same_columns + toward_columns
    minus_target = backend.apply_l(-1, high)
    minus_data = backend.span_residual(minus_target, minus_columns)

    identity_residual = backend.linear_combination(
        (1, minus_target),
        (-1, same_columns[0]),
        (-1, same_columns[1]),
        (1, backend.apply_lf(-1, high)),
    )

    onset = int(2 * n**2 - sp.Rational(1, 8))
    highest_weight_residual = 0.0
    for copy in (1, 2):
        for mode in range(1, onset + 2):
            highest_weight_residual = max(
                highest_weight_residual,
                backend.max_abs(
                    backend.apply_double_virasoro(copy, mode, high)
                ),
            )

    return {
        "n": str(n),
        "parity": parity,
        "L1": result_data(plus_level, len(plus_columns), plus_data),
        "L-1": result_data(toward_level, len(minus_columns), minus_data),
        "L1_coefficients": [complex(value) for value in plus_data[5]],
        "L-1_coefficients": [complex(value) for value in minus_data[5]],
        "highest_weight_max_residual": highest_weight_residual,
        "inverse_identity_max_residual": backend.max_abs(identity_residual),
    }


def result_data(relative_level, column_count, data):
    rows, rank, absolute, relative, smallest, _ = data
    return {
        "relative_level": relative_level,
        "rows": rows,
        "columns": column_count,
        "rank": rank,
        "absolute_residual": absolute,
        "relative_residual": relative,
        "smallest_retained_singular_value": smallest,
        "passed": rank == column_count and relative < 1.0e-10,
    }


def encode_complex(values):
    return [f"{value.real:.17g}{value.imag:+.17g}j" for value in values]


def run_family(labels):
    results = []
    timings = {}
    for n in labels:
        started = time.perf_counter()
        for parity in (0, 1):
            result = fit_actions(n, parity)
            results.append(result)
            print(
                f"n={n}, alpha={parity}: "
                f"L1 rank={result['L1']['rank']}/{result['L1']['columns']}, "
                f"residual={result['L1']['relative_residual']:.3e}; "
                f"L-1 rank={result['L-1']['rank']}/{result['L-1']['columns']}, "
                f"residual={result['L-1']['relative_residual']:.3e}; "
                f"highest-weight residual={result['highest_weight_max_residual']:.3e}; "
                f"inverse-identity residual={result['inverse_identity_max_residual']:.3e}",
                flush=True,
            )
        timings[str(n)] = time.perf_counter() - started
        print(f"n={n}: both parities in {timings[str(n)]:.3f} s", flush=True)
    return results, timings


def coefficient_differences(negative_results, positive_results):
    positive_by_key = {
        (sp.Rational(item["n"]), item["parity"]): item for item in positive_results
    }
    comparisons = []
    for negative in negative_results:
        n = sp.Rational(negative["n"])
        reflected = positive_by_key[(-n, negative["parity"])]
        plus_difference = max(
            (
                abs(left - right)
                for left, right in zip(
                    negative["L1_coefficients"], reflected["L1_coefficients"]
                )
            ),
            default=0.0,
        )
        minus_difference = max(
            (
                abs(left - right)
                for left, right in zip(
                    negative["L-1_coefficients"], reflected["L-1_coefficients"]
                )
            ),
            default=0.0,
        )
        comparisons.append(
            {
                "n": str(n),
                "parity": negative["parity"],
                "L1_max_coefficient_difference": plus_difference,
                "L-1_max_coefficient_difference": minus_difference,
            }
        )
    return comparisons


def serializable_results(results):
    answer = []
    for item in results:
        copied = dict(item)
        copied["L1_coefficients"] = encode_complex(copied["L1_coefficients"])
        copied["L-1_coefficients"] = encode_complex(copied["L-1_coefficients"])
        answer.append(copied)
    return answer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path)
    arguments = parser.parse_args()

    labels = tuple(
        -value
        for value in (
            sp.Rational(3, 4),
            sp.Rational(5, 4),
            sp.Rational(7, 4),
            sp.Rational(9, 4),
        )
    )
    b_value = sp.Rational(7, 5)
    p_value = sp.Rational(11, 23)

    total_started = time.perf_counter()
    configure(1, b_value, p_value)
    negative_results, timings = run_family(labels)
    direct_time = time.perf_counter() - total_started

    reflection_started = time.perf_counter()
    configure(-1, b_value, -p_value)
    positive_results, _ = run_family(tuple(-value for value in labels))
    comparisons = coefficient_differences(negative_results, positive_results)
    reflection_time = time.perf_counter() - reflection_started
    total_time = time.perf_counter() - total_started

    maximum_coefficient_difference = max(
        max(
            item["L1_max_coefficient_difference"],
            item["L-1_max_coefficient_difference"],
        )
        for item in comparisons
    )
    passed = (
        all(
            item[action]["passed"]
            and item["highest_weight_max_residual"] < 1.0e-10
            and item["inverse_identity_max_residual"] < 1.0e-10
            for item in negative_results
            for action in ("L1", "L-1")
        )
        and maximum_coefficient_difference < 1.0e-9
    )
    print(
        f"direct negative-family time={direct_time:.3f} s; "
        f"reflection-comparison time={reflection_time:.3f} s; "
        f"total={total_time:.3f} s"
    )
    print(
        "maximum coefficient difference under (n,P)->(-n,-P): "
        f"{maximum_coefficient_difference:.3e}"
    )
    print("PASS" if passed else "FAIL")

    payload = {
        "sample": {
            "b": str(b_value),
            "P": str(p_value),
            "Q": str(b_value + 1 / b_value),
        },
        "direct_negative_results": serializable_results(negative_results),
        "direct_timings_seconds": timings,
        "direct_total_seconds": direct_time,
        "reflection_comparison_seconds": reflection_time,
        "total_seconds": total_time,
        "reflection_coefficient_comparisons": comparisons,
        "maximum_reflection_coefficient_difference": maximum_coefficient_difference,
        "passed": passed,
    }
    if arguments.json:
        arguments.json.write_text(json.dumps(payload, indent=2) + "\n")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
