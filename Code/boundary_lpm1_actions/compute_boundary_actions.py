#!/usr/bin/env python3
"""Solve the boundary L_{+/-1} actions in the double-Virasoro basis."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import sympy as sp


HERE = Path(__file__).resolve().parent
PYTHON_ROOT = HERE.parent


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


ramond = load_module(
    "ramond_boundary_backend",
    PYTHON_ROOT / "ramond_lpm1_proposition_check" / "check_proposition_7_13.py",
)
ns = load_module(
    "ns_boundary_backend",
    PYTHON_ROOT / "ns_lpm1_proposition_check" / "check_ns_proposition.py",
)


def ns_boundary_states():
    vacuum = ns.raw_branch(sp.Rational(0))
    half = ns.raw_branch(sp.Rational(1, 2))
    plus_coefficient = complex(sp.N(ns.P_VALUE + ns.Q_VALUE / 2, 30))
    minus_coefficient = complex(sp.N(ns.Q_VALUE / 2 - ns.P_VALUE, 30))
    plus = {state: plus_coefficient * value for state, value in half.items()}
    minus = {
        ((1,), (), ()): minus_coefficient,
        ((), (), (1,)): -1j * plus_coefficient,
    }
    return vacuum, plus, minus


def descendants(module, states):
    return [
        module.double_virasoro_descendant(state, partition_1, partition_2)
        for state in states
        for partition_1, partition_2 in (((1,), ()), ((), (1,)))
    ]


def ramond_branch_in_common_realization(label, parity):
    substitutions = {
        ramond.branch.Q: ramond.Q_VALUE,
        ramond.branch.P: ramond.P_VALUE,
    }
    _, sectors = ramond.branch.branch_in_abstract_basis(
        label, parity, substitutions=substitutions
    )
    answer = {}
    for auxiliary_state, (level, ordered_basis, abstract_coefficients) in sectors.items():
        fixed_basis, fixed_transition = ramond.branch.transition(level, -1)
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
                ] = complex(sp.N(coefficient, 30))
    return answer


def print_solution(label, module, state, columns):
    positive = module.apply_l(1, state)
    print(f"{label}: max |L1|={module.max_abs(positive):.3e}")
    data = module.span_residual(module.apply_l(-1, state), columns)
    coefficients = data[5]
    print(
        f"  L-1 rank={data[1]}/{len(columns)}, residual={data[3]:.3e}, "
        f"coefficients={[complex(value) for value in coefficients]}"
    )


def main():
    vacuum, ns_plus, ns_minus = ns_boundary_states()
    print_solution("NS n=0", ns, vacuum, descendants(ns, (vacuum,)))
    ns_columns = descendants(ns, (ns_plus, ns_minus))
    print_solution("NS n=1/2", ns, ns_plus, ns_columns)
    print_solution("NS n=-1/2", ns, ns_minus, ns_columns)

    for parity in (0, 1):
        r_plus = ramond_branch_in_common_realization(sp.Rational(1, 4), parity)
        r_minus = ramond_branch_in_common_realization(-sp.Rational(1, 4), parity)
        r_three_quarters = ramond_branch_in_common_realization(
            sp.Rational(3, 4), parity
        )
        r_minus_three_quarters = ramond_branch_in_common_realization(
            -sp.Rational(3, 4), parity
        )
        plus_columns = descendants(ramond, (r_plus,)) + [r_minus_three_quarters]
        minus_columns = descendants(ramond, (r_minus,)) + [r_three_quarters]
        print_solution(
            f"R n=1/4 alpha={parity}", ramond, r_plus, plus_columns
        )
        print_solution(
            f"R n=-1/4 alpha={parity}", ramond, r_minus, minus_columns
        )


if __name__ == "__main__":
    main()
