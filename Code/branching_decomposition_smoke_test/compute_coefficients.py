#!/usr/bin/env python3
"""Compute all L_{+/-1} branch-decomposition coefficients at one generic point.

The script uses the state-level Ramond and NS oscillator implementations in
the neighboring proposition checks.  It solves the complete, overdetermined
double-Virasoro descendant systems and writes both a machine-readable JSON
file and full Markdown coefficient tables.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

PROGRAM_START = time.perf_counter()

import sympy as sp


HERE = Path(__file__).resolve().parent
PYTHON_ROOT = HERE.parent

B_VALUE = sp.sqrt(sp.Rational(3, 2))
P_VALUE = sp.Rational(1, 3)
Q_VALUE = sp.simplify(B_VALUE + 1 / B_VALUE)
C_VALUE = sp.simplify(sp.Rational(3, 2) + 3 * Q_VALUE**2)


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


ramond = load_module(
    "ramond_lpm1_smoke_backend",
    PYTHON_ROOT
    / "ramond_lpm1_proposition_check"
    / "check_proposition_7_13.py",
)
ns = load_module(
    "ns_lpm1_smoke_backend",
    PYTHON_ROOT / "ns_lpm1_proposition_check" / "check_ns_proposition.py",
)


def configure(module) -> None:
    module.B_VALUE = B_VALUE
    module.P_VALUE = P_VALUE
    module.Q_VALUE = Q_VALUE
    for cached in (
        module.as_complex,
        module.apply_physical_l,
        module.apply_physical_g,
        module.double_virasoro_on_state,
    ):
        cached.cache_clear()


configure(ramond)
configure(ns)


def partition_pairs(level: int, partition_function):
    return tuple(
        (first, second)
        for first_level in range(level + 1)
        for first in partition_function(first_level)
        for second in partition_function(level - first_level)
    )


def encode_complex(value) -> dict[str, float]:
    value = complex(value)
    real = 0.0 if abs(value.real) < 5.0e-13 else float(value.real)
    imaginary = 0.0 if abs(value.imag) < 5.0e-13 else float(value.imag)
    return {"real": real, "imag": imaginary}


def coefficient_entry(first, second, coefficient):
    return {
        "A": list(first),
        "B": list(second),
        "coefficient": encode_complex(coefficient),
    }


def solve_ramond(n: sp.Rational, parity: int):
    high = ramond.raw_branch(n, parity)
    low = ramond.raw_branch(n - 1, parity)

    plus_level = int(4 * n - 3)
    plus_pairs = partition_pairs(plus_level, ramond.branch.partitions)
    plus_columns = [
        ramond.double_virasoro_descendant(low, first, second)
        for first, second in plus_pairs
    ]
    plus = ramond.span_residual(ramond.apply_l(1, high), plus_columns)

    minus_level = int(4 * n - 1)
    same_branch = [
        ramond.double_virasoro_descendant(high, (1,), ()),
        ramond.double_virasoro_descendant(high, (), (1,)),
    ]
    lower_pairs = partition_pairs(minus_level, ramond.branch.partitions)
    lower_columns = [
        ramond.double_virasoro_descendant(low, first, second)
        for first, second in lower_pairs
    ]
    minus = ramond.span_residual(
        ramond.apply_l(-1, high), same_branch + lower_columns
    )

    inverse_identity = ramond.linear_combination(
        (1, ramond.apply_l(-1, high)),
        (-1, same_branch[0]),
        (-1, same_branch[1]),
        (1, ramond.apply_lf(-1, high)),
    )

    return {
        "n": str(n),
        "alpha": parity,
        "L_1": {
            "relative_level": plus_level,
            "rows": plus[0],
            "columns": len(plus_columns),
            "rank": plus[1],
            "absolute_residual": plus[2],
            "relative_residual": plus[3],
            "smallest_retained_singular_value": plus[4],
            "coefficients": [
                coefficient_entry(first, second, coefficient)
                for (first, second), coefficient in zip(plus_pairs, plus[5])
            ],
        },
        "L_minus_1": {
            "lower_branch_relative_level": minus_level,
            "rows": minus[0],
            "columns": len(same_branch) + len(lower_columns),
            "rank": minus[1],
            "absolute_residual": minus[2],
            "relative_residual": minus[3],
            "smallest_retained_singular_value": minus[4],
            "same_branch_coefficients": {
                "copy_1": encode_complex(minus[5][0]),
                "copy_2": encode_complex(minus[5][1]),
            },
            "lower_branch_coefficients": [
                coefficient_entry(first, second, coefficient)
                for (first, second), coefficient in zip(
                    lower_pairs, minus[5][2:]
                )
            ],
        },
        "inverse_identity_max_residual": ramond.max_abs(inverse_identity),
    }


def solve_ns(n: sp.Rational):
    high = ns.raw_branch(n)
    low = ns.raw_branch(n - 1)

    plus_level = int(4 * n - 3)
    plus_pairs = partition_pairs(plus_level, ns.partitions)
    plus_columns = [
        ns.double_virasoro_descendant(low, first, second)
        for first, second in plus_pairs
    ]
    plus = ns.span_residual(ns.apply_l(1, high), plus_columns)

    minus_level = int(4 * n - 1)
    same_branch = [
        ns.double_virasoro_descendant(high, (1,), ()),
        ns.double_virasoro_descendant(high, (), (1,)),
    ]
    lower_pairs = partition_pairs(minus_level, ns.partitions)
    lower_columns = [
        ns.double_virasoro_descendant(low, first, second)
        for first, second in lower_pairs
    ]
    minus = ns.span_residual(ns.apply_l(-1, high), same_branch + lower_columns)

    inverse_identity = ns.linear_combination(
        (1, ns.apply_l(-1, high)),
        (-1, same_branch[0]),
        (-1, same_branch[1]),
        (1, ns.apply_lf(-1, high)),
    )

    return {
        "n": str(n),
        "L_1": {
            "relative_level": plus_level,
            "rows": plus[0],
            "columns": len(plus_columns),
            "rank": plus[1],
            "absolute_residual": plus[2],
            "relative_residual": plus[3],
            "smallest_retained_singular_value": plus[4],
            "coefficients": [
                coefficient_entry(first, second, coefficient)
                for (first, second), coefficient in zip(plus_pairs, plus[5])
            ],
        },
        "L_minus_1": {
            "lower_branch_relative_level": minus_level,
            "rows": minus[0],
            "columns": len(same_branch) + len(lower_columns),
            "rank": minus[1],
            "absolute_residual": minus[2],
            "relative_residual": minus[3],
            "smallest_retained_singular_value": minus[4],
            "same_branch_coefficients": {
                "copy_1": encode_complex(minus[5][0]),
                "copy_2": encode_complex(minus[5][1]),
            },
            "lower_branch_coefficients": [
                coefficient_entry(first, second, coefficient)
                for (first, second), coefficient in zip(
                    lower_pairs, minus[5][2:]
                )
            ],
        },
        "inverse_identity_max_residual": ns.max_abs(inverse_identity),
    }


def coefficient_text(value: dict[str, float]) -> str:
    real = value["real"]
    imaginary = value["imag"]
    if imaginary == 0:
        return f"{real:.11g}"
    if real == 0:
        return f"{imaginary:.11g}i"
    return f"{real:.11g}{imaginary:+.11g}i"


def partition_text(partition) -> str:
    return "empty" if not partition else "(" + ",".join(map(str, partition)) + ")"


def append_case_tables(lines: list[str], title: str, result: dict) -> None:
    lines.extend((f"### {title}", ""))
    plus = result["L_1"]
    lines.extend(
        (
            f"`L_1`: level {plus['relative_level']}, rank "
            f"{plus['rank']}/{plus['columns']}, relative residual "
            f"{plus['relative_residual']:.3e}.",
            "",
            "| A | B | coefficient |",
            "|---|---|---:|",
        )
    )
    for entry in plus["coefficients"]:
        lines.append(
            f"| {partition_text(entry['A'])} | {partition_text(entry['B'])} | "
            f"`{coefficient_text(entry['coefficient'])}` |"
        )

    minus = result["L_minus_1"]
    lines.extend(
        (
            "",
            f"`L_-1`: lower-branch level {minus['lower_branch_relative_level']}, "
            f"rank {minus['rank']}/{minus['columns']}, relative residual "
            f"{minus['relative_residual']:.3e}.",
            "",
            "| term | A | B | coefficient |",
            "|---|---|---|---:|",
            "| same branch, copy 1 | (1) | empty | `"
            + coefficient_text(minus["same_branch_coefficients"]["copy_1"])
            + "` |",
            "| same branch, copy 2 | empty | (1) | `"
            + coefficient_text(minus["same_branch_coefficients"]["copy_2"])
            + "` |",
        )
    )
    for entry in minus["lower_branch_coefficients"]:
        lines.append(
            f"| lower branch | {partition_text(entry['A'])} | "
            f"{partition_text(entry['B'])} | "
            f"`{coefficient_text(entry['coefficient'])}` |"
        )
    lines.append("")


def make_markdown(payload: dict) -> str:
    lines = [
        "# Branch-decomposition coefficient smoke test",
        "",
        "The generic point is",
        "",
        "\\[",
        "P=\\frac13,\\qquad c=14,\\qquad b=\\sqrt{\\frac32},"
        "\\qquad Q=\\frac5{\\sqrt6}.",
        "\\]",
        "",
        "At this point the two embedded central charges are",
        "",
        "\\[c^{(1)}=-24,\\qquad c^{(2)}=\\frac{77}{2}.\\]",
        "",
        "All systems below are overdetermined state-level systems. Every one",
        "has full column rank. Partitions `A` and `B` label",
        "\\(L_{-A}^{(1)}L_{-B}^{(2)}\\) in the order used in the main notes.",
        "The displayed decimals are truncated; `coefficients.json` retains the",
        "full double-precision values.",
        "",
        "## Summary",
        "",
        "| sector | n | alpha | L1 rank | L1 residual | L-1 rank | L-1 residual | time (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in payload["ramond"]:
        lines.append(
            f"| R | {result['n']} | {result['alpha']} | "
            f"{result['L_1']['rank']}/{result['L_1']['columns']} | "
            f"{result['L_1']['relative_residual']:.3e} | "
            f"{result['L_minus_1']['rank']}/{result['L_minus_1']['columns']} | "
            f"{result['L_minus_1']['relative_residual']:.3e} | "
            f"{result['runtime_seconds']:.3f} |"
        )
    for result in payload["ns"]:
        lines.append(
            f"| NS | {result['n']} | - | "
            f"{result['L_1']['rank']}/{result['L_1']['columns']} | "
            f"{result['L_1']['relative_residual']:.3e} | "
            f"{result['L_minus_1']['rank']}/{result['L_minus_1']['columns']} | "
            f"{result['L_minus_1']['relative_residual']:.3e} | "
            f"{result['runtime_seconds']:.3f} |"
        )

    lines.extend(
        (
            "",
            f"Total coefficient-solver wall time: "
            f"`{payload['solver_runtime_seconds']:.3f} s`.",
            "",
            f"Process wall time through completion of all solves (including "
            f"imports and backend setup): "
            f"`{payload['process_runtime_through_solves_seconds']:.3f} s`.",
            "",
            "Per-case times are measured in one sequential run. Later cases",
            "reuse cached elementary mode actions from earlier cases.",
        )
    )

    lines.extend(("", "## Ramond coefficients", ""))
    for result in payload["ramond"]:
        append_case_tables(
            lines,
            f"n = {result['n']}, alpha = {result['alpha']}",
            result,
        )
    lines.extend(("## NS coefficients", ""))
    for result in payload["ns"]:
        append_case_tables(lines, f"n = {result['n']}", result)
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, default=HERE)
    arguments = parser.parse_args()

    total_start = time.perf_counter()
    ramond_results = []
    for n in (
        sp.Rational(3, 4),
        sp.Rational(5, 4),
        sp.Rational(7, 4),
        sp.Rational(9, 4),
    ):
        for parity in (0, 1):
            case_start = time.perf_counter()
            result = solve_ramond(n, parity)
            result["runtime_seconds"] = time.perf_counter() - case_start
            ramond_results.append(result)
            print(
                f"R n={n}, alpha={parity}: "
                f"L1 {result['L_1']['rank']}/{result['L_1']['columns']} "
                f"res={result['L_1']['relative_residual']:.3e}; "
                f"L-1 {result['L_minus_1']['rank']}/"
                f"{result['L_minus_1']['columns']} "
                f"res={result['L_minus_1']['relative_residual']:.3e}; "
                f"time={result['runtime_seconds']:.3f}s",
                flush=True,
            )

    ns_results = []
    for n in (
        sp.Rational(1),
        sp.Rational(3, 2),
        sp.Rational(2),
        sp.Rational(5, 2),
    ):
        case_start = time.perf_counter()
        result = solve_ns(n)
        result["runtime_seconds"] = time.perf_counter() - case_start
        ns_results.append(result)
        print(
            f"NS n={n}: L1 {result['L_1']['rank']}/"
            f"{result['L_1']['columns']} "
            f"res={result['L_1']['relative_residual']:.3e}; "
            f"L-1 {result['L_minus_1']['rank']}/"
            f"{result['L_minus_1']['columns']} "
            f"res={result['L_minus_1']['relative_residual']:.3e}; "
            f"time={result['runtime_seconds']:.3f}s",
            flush=True,
        )

    b1_squared = sp.simplify(2 * B_VALUE**2 / (1 - B_VALUE**2))
    b2_inverse_squared = sp.simplify(2 / (B_VALUE**2 - 1))
    c1 = sp.simplify(
        1 + 6 * (sp.sqrt(b1_squared) + 1 / sp.sqrt(b1_squared)) ** 2
    )
    b2 = 1 / sp.sqrt(b2_inverse_squared)
    c2 = sp.simplify(1 + 6 * (b2 + 1 / b2) ** 2)

    solver_runtime = time.perf_counter() - total_start
    process_runtime = time.perf_counter() - PROGRAM_START
    payload = {
        "point": {
            "P": str(P_VALUE),
            "c": str(C_VALUE),
            "b": str(B_VALUE),
            "Q": str(Q_VALUE),
            "c_1": str(c1),
            "c_2": str(c2),
        },
        "ramond": ramond_results,
        "ns": ns_results,
        "solver_runtime_seconds": solver_runtime,
        "process_runtime_through_solves_seconds": process_runtime,
    }
    output = arguments.output_directory
    output.mkdir(parents=True, exist_ok=True)
    (output / "coefficients.json").write_text(json.dumps(payload, indent=2) + "\n")
    (output / "COEFFICIENTS.md").write_text(make_markdown(payload))

    all_results = ramond_results + ns_results
    full_rank = all(
        result[mode]["rank"] == result[mode]["columns"]
        for result in all_results
        for mode in ("L_1", "L_minus_1")
    )
    max_residual = max(
        result[mode]["relative_residual"]
        for result in all_results
        for mode in ("L_1", "L_minus_1")
    )
    print(f"full column rank: {full_rank}")
    print(f"maximum relative residual: {max_residual:.3e}")
    print(f"total coefficient-solver wall time: {solver_runtime:.3f}s")
    print(f"process wall time through all solves: {process_runtime:.3f}s")
    if not full_rank or max_residual >= 1.0e-9:
        raise SystemExit("Smoke test failed.")


if __name__ == "__main__":
    main()
