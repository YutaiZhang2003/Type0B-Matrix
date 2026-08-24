#!/usr/bin/env python3
"""Fixed-q diagnostic for the three-variable Ramond enlarged block.

The implementation follows the first L_1 Ward recursion and the double-
Virasoro formula in the Ramond section of SCblock.tex.  The ordinary Virasoro
theta blocks are evaluated by the central-charge recursion; no PBW block sum
is used in the production calculation.

"Level N" always means total plumbing level

    level(q_1) + level(q_2) + level(q_3) <= N.

The benchmark evaluates the canonical even block with eta=eta'=+ and
(eta_1,eta_2,eta_3)=(+,+,+) at one generic numerical point.  It does not
produce a coefficient-by-coefficient q-expansion; use compute_q_expansion.py
for that calculation.
"""

from __future__ import annotations

import argparse
import cmath
import concurrent.futures
import json
import math
import os
import platform
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy import sparse


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BRANCHING = ROOT / "python" / "ramond_branching_recursion"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BRANCHING) not in sys.path:
    sys.path.insert(0, str(BRANCHING))

from compute_target import (  # noqa: E402
    BranchWeights,
    FreeFieldModule,
    TOLERANCE,
    norm_product,
    ordinary_factor,
    solve_ns_l1,
    solve_ramond_lminus,
)
from direct_state_check import (  # noqa: E402
    AuxiliaryThreePoint,
    DirectBranchingCoefficient,
)
import plumbing.ccy_genus2_block as ccy_module  # noqa: E402
from plumbing.ccy_genus2_block import (  # noqa: E402
    ccy_genus2_block,
    genus2_vacuum_seed_schottky,
)


class RamondAuxiliaryThreePoint(AuxiliaryThreePoint):
    """Auxiliary NS-R-R form with the chiral phase fixed in the main notes."""

    def base_value(self, states):
        ground2 = states[1][1]
        ground3 = states[2][1]
        if ground2 != ground3:
            return 0.0j
        return 1.0 + 0.0j if ground2 == 0 else 1.0j


def continuous_b_square_rs(r: int, s: int, weight: complex) -> complex:
    """Choose the non-cancelling continuation of the c-recursion root.

    On the real slice reached by the first Virasoro copy, the principal
    square root can jump between the two algebraic roots.  For example, for
    (r,s)=(2,1) it can select the identically zero root once h<-1/2 and turn a
    pole at infinity into a catastrophic finite floating-point contribution.
    The main notes prescribe continuous continuation of the b_{r,s} branch.
    Selecting the root with larger modulus implements that continuation on
    this benchmark path and avoids subtractive cancellation.
    """

    weight = complex(weight)
    radical = (r - s) ** 2 + 4.0 * (r * s - 1.0) * weight + 4.0 * weight * weight
    root = cmath.sqrt(radical)
    denominator = 1.0 - r * r
    first = (r * s - 1.0 + 2.0 * weight + root) / denominator
    second = (r * s - 1.0 + 2.0 * weight - root) / denominator
    return first if abs(first) >= abs(second) else second


# Every c-recursion helper resolves b_{r,s} through this module global.
ccy_module.b_square_rs_from_h = continuous_b_square_rs


def encode(value: complex) -> dict[str, float]:
    value = complex(value)
    return {"real": float(value.real), "imag": float(value.imag)}


def show(label: Fraction) -> str:
    label = Fraction(label)
    return str(label.numerator) if label.denominator == 1 else f"{label.numerator}/{label.denominator}"


def ns_labels(cutoff: int) -> tuple[Fraction, ...]:
    twice_cutoff = 2 * int(cutoff)
    bound = math.isqrt(twice_cutoff)
    return tuple(Fraction(index, 2) for index in range(-bound, bound + 1))


def ramond_labels(cutoff: int) -> tuple[Fraction, ...]:
    twice_cutoff = 2 * int(cutoff)
    answer: list[Fraction] = []
    numerator = 1
    while (numerator * numerator - 1) // 4 <= twice_cutoff:
        label = Fraction(numerator, 4)
        answer.extend((-label, label))
        numerator += 2
    return tuple(sorted(answer))


def base_twice_level(labels: tuple[Fraction, Fraction, Fraction]) -> int:
    n1, n2, n3 = labels
    value = 4 * n1 * n1 + 4 * n2 * n2 - Fraction(1, 4) + 4 * n3 * n3 - Fraction(1, 4)
    if value.denominator != 1:
        raise AssertionError(f"nonintegral twice-level at {labels}")
    return int(value)


class BranchingGrid:
    """Close the first Ward identity on the finite level-cutoff lattice."""

    def __init__(self, b: float, momenta: tuple[float, float, float], cutoff: int):
        self.b = float(b)
        self.momenta = tuple(float(value) for value in momenta)
        self.cutoff = int(cutoff)
        self.weights = BranchWeights(self.b, self.momenta)
        self.modules = (
            FreeFieldModule("NS", self.b, self.momenta[0]),
            FreeFieldModule("R", self.b, self.momenta[1]),
            FreeFieldModule("R", self.b, self.momenta[2]),
        )
        self.ns = ns_labels(self.cutoff)
        self.r = ramond_labels(self.cutoff)
        self.ns_actions: dict[Fraction, tuple] = {}
        self.r_actions: dict[tuple[int, int, Fraction], tuple] = {}
        self.action_diagnostics: list[dict[str, object]] = []
        self.direct = DirectBranchingCoefficient(self.b, self.momenta)
        self.direct.auxiliary_form = RamondAuxiliaryThreePoint(
            self.direct.free_modules
        )

    def build_actions(self) -> None:
        for label in self.ns:
            if abs(label) < 1:
                self.ns_actions[label] = ()
                continue
            terms, fit = solve_ns_l1(self.modules[0], label)
            self.ns_actions[label] = tuple(terms)
            self.action_diagnostics.append(
                {
                    "sector": "NS",
                    "slot": 1,
                    "label": show(label),
                    "parity": 0,
                    "relative_residual": float(fit["relative_residual"]),
                }
            )
        for slot in (1, 2):
            for parity in (0, 1):
                for label in self.r:
                    terms, fit = solve_ramond_lminus(
                        self.modules[slot], label, parity
                    )
                    self.r_actions[(slot, parity, label)] = tuple(terms)
                    self.action_diagnostics.append(
                        {
                            "sector": "R",
                            "slot": slot + 1,
                            "label": show(label),
                            "parity": parity,
                            "relative_residual": float(fit["relative_residual"]),
                        }
                    )

    def solve(self, alpha2: int, alpha3: int) -> tuple[dict[tuple[Fraction, Fraction, Fraction], complex], dict[str, object]]:
        alpha2 = int(alpha2)
        alpha3 = int(alpha3)
        n1_parity = (alpha2 + alpha3) % 2
        first_labels = tuple(
            label for label in self.ns if int(2 * label) % 2 == n1_parity
        )
        unknowns = tuple(
            (first, second, third)
            for first in first_labels
            for second in self.r
            for third in self.r
        )
        index = {labels: position for position, labels in enumerate(unknowns)}
        row_indices: list[int] = []
        column_indices: list[int] = []
        entries: list[complex] = []
        rhs: list[complex] = []

        def append(equation: dict[tuple[Fraction, Fraction, Fraction], complex], value: complex) -> None:
            row_norm = math.sqrt(sum(abs(coefficient) ** 2 for coefficient in equation.values()))
            if row_norm <= TOLERANCE:
                return
            row = len(rhs)
            for labels, coefficient in equation.items():
                row_indices.append(row)
                column_indices.append(index[labels])
                entries.append(complex(coefficient) / row_norm)
            rhs.append(complex(value) / row_norm)

        for labels in unknowns:
            equation: dict[tuple[Fraction, Fraction, Fraction], complex] = {}
            action_sets = (
                (1.0, 0, self.ns_actions[labels[0]]),
                (-1.0, 1, self.r_actions[(1, alpha2, labels[1])]),
                (-1.0, 2, self.r_actions[(2, alpha3, labels[2])]),
            )
            for ward_sign, slot, actions in action_sets:
                for term in actions:
                    changed, coefficient = ordinary_factor(
                        self.weights, labels, slot, term
                    )
                    if changed not in index:
                        raise AssertionError(
                            f"Ward action left the finite lattice: {labels} -> {changed}"
                        )
                    equation[changed] = equation.get(changed, 0.0j) + ward_sign * coefficient
            append(equation, 0.0j)

        anchor_first = (Fraction(0),) if n1_parity == 0 else (Fraction(-1, 2), Fraction(1, 2))
        anchor_r = (
            Fraction(-3, 4),
            Fraction(-1, 4),
            Fraction(1, 4),
            Fraction(3, 4),
        )
        anchors: dict[tuple[Fraction, Fraction, Fraction], complex] = {}
        for first in anchor_first:
            for second in anchor_r:
                for third in anchor_r:
                    labels = (first, second, third)
                    if labels not in index:
                        continue
                    value = self.direct.raw(labels, alpha2, alpha3, eta=1)
                    anchors[labels] = value
                    append({labels: 1.0 + 0.0j}, value)

        matrix = sparse.coo_matrix(
            (np.asarray(entries, dtype=np.complex128), (row_indices, column_indices)),
            shape=(len(rhs), len(unknowns)),
        ).tocsr()
        vector = np.asarray(rhs, dtype=np.complex128)
        column_norms = np.sqrt(np.asarray(abs(matrix).power(2).sum(axis=0)).ravel())
        if np.any(column_norms == 0):
            raise AssertionError("a Ward-system column is identically zero")
        scaled = (matrix @ sparse.diags(1.0 / column_norms)).toarray()
        scaled_values, _, rank, singular_values = np.linalg.lstsq(
            scaled, vector, rcond=1.0e-13
        )
        values = scaled_values / column_norms
        residual = matrix @ values - vector
        residual_norm = float(np.linalg.norm(residual))
        vector_norm = float(np.linalg.norm(vector))
        solution = {
            labels: complex(values[position])
            for labels, position in index.items()
        }
        diagnostic = {
            "alpha_2": alpha2,
            "alpha_3": alpha3,
            "unknowns": len(unknowns),
            "equations": int(matrix.shape[0]),
            "anchors": len(anchors),
            "rank": int(rank),
            "full_column_rank": bool(rank == len(unknowns)),
            "absolute_residual": residual_norm,
            "relative_residual": residual_norm / max(vector_norm, 1.0),
            "smallest_retained_singular_value": float(singular_values[rank - 1]),
            "condition_number": float(singular_values[0] / singular_values[rank - 1]),
        }
        return solution, diagnostic


def ordinary_pair_task(task: tuple) -> tuple[tuple[Fraction, Fraction, Fraction], int, complex, complex, float]:
    labels, remaining, b, momenta, q_values = task
    started = time.perf_counter()
    weights = BranchWeights(b, momenta)
    values: list[complex] = []
    for copy in (0, 1):
        h1, h2, h3 = weights.triple(labels, copy)
        value = ccy_genus2_block(
            c=weights.central_charges[copy],
            h1=h1,
            h2=h2,
            h3=h3,
            q1=q_values[0],
            q2=q_values[1],
            q3=q_values[2],
            order=remaining,
            include_vacuum_seed=False,
            collision_aware=False,
        ).value
        values.append(value)
    return labels, remaining, values[0], values[1], time.perf_counter() - started


def run(cutoff: int, workers: int, output: Path) -> dict[str, object]:
    b = 7.0 / 5.0
    momenta = (11.0 / 23.0, 13.0 / 29.0, 17.0 / 31.0)
    q_values = (0.019, 0.023, 0.029)
    started_total = time.perf_counter()

    branching = BranchingGrid(b, momenta, cutoff)
    started = time.perf_counter()
    branching.build_actions()
    action_seconds = time.perf_counter() - started
    print(f"branch actions: {action_seconds:.6f} s", flush=True)

    started = time.perf_counter()
    raw_grids: dict[tuple[int, int], dict[tuple[Fraction, Fraction, Fraction], complex]] = {}
    ward_diagnostics: list[dict[str, object]] = []
    for alpha2 in (0, 1):
        for alpha3 in (0, 1):
            values, diagnostic = branching.solve(alpha2, alpha3)
            raw_grids[(alpha2, alpha3)] = values
            ward_diagnostics.append(diagnostic)
            print(
                f"Ward grid ({alpha2},{alpha3}): rank {diagnostic['rank']}/{diagnostic['unknowns']}, "
                f"relative residual {diagnostic['relative_residual']:.3e}",
                flush=True,
            )
    ward_seconds = time.perf_counter() - started
    print(f"branching Ward systems: {ward_seconds:.6f} s", flush=True)

    triples = tuple(
        labels
        for labels in (
            (first, second, third)
            for first in branching.ns
            for second in branching.r
            for third in branching.r
        )
        if base_twice_level(labels) <= 2 * cutoff
    )
    tasks = [
        (
            labels,
            (2 * cutoff - base_twice_level(labels)) // 2,
            b,
            momenta,
            q_values,
        )
        for labels in triples
    ]
    started = time.perf_counter()
    ordinary_results: dict[tuple[Fraction, Fraction, Fraction], tuple[int, complex, complex, float]] = {}
    if workers == 1:
        iterator = map(ordinary_pair_task, tasks)
        executor = None
    else:
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=workers)
        iterator = executor.map(ordinary_pair_task, tasks, chunksize=1)
    try:
        for count, (labels, remaining, first, second, seconds) in enumerate(iterator, start=1):
            ordinary_results[labels] = (remaining, first, second, seconds)
            if count % 50 == 0 or count == len(tasks):
                print(f"Virasoro pairs: {count}/{len(tasks)}", flush=True)
    finally:
        if executor is not None:
            executor.shutdown()
    virasoro_seconds = time.perf_counter() - started
    print(f"ordinary Virasoro c-recursions: {virasoro_seconds:.6f} s", flush=True)

    started = time.perf_counter()
    reduced_value = 0.0j
    rows: list[dict[str, object]] = []
    by_base_level: dict[int, complex] = {}
    for labels in triples:
        n1, n2, n3 = labels
        base2 = base_twice_level(labels)
        remaining, first_vir, second_vir, task_seconds = ordinary_results[labels]
        alpha_pairs = (
            ((0, 0), (1, 1))
            if int(2 * n1) % 2 == 0
            else ((0, 1), (1, 0))
        )
        branching_sum = 0.0j
        branch_rows: list[dict[str, object]] = []
        for alpha2, alpha3 in alpha_pairs:
            raw = raw_grids[(alpha2, alpha3)][labels]
            normalized = raw / norm_product(
                labels, alpha2, alpha3, b, momenta
            )
            exponent = (
                int(2 * n1) * alpha2
                + int(2 * n1) * alpha3
                + alpha2 * alpha3
            )
            sign = (-1) ** exponent
            term = sign * normalized * normalized
            branching_sum += term
            branch_rows.append(
                {
                    "alpha_2": alpha2,
                    "alpha_3": alpha3,
                    "B": encode(normalized),
                    "signed_B_squared": encode(term),
                }
            )
        q_prefactor = (
            q_values[0] ** float(2 * n1 * n1)
            * q_values[1] ** float(2 * n2 * n2 - Fraction(1, 8))
            * q_values[2] ** float(2 * n3 * n3 - Fraction(1, 8))
        )
        contribution = q_prefactor * branching_sum * first_vir * second_vir
        reduced_value += contribution
        by_base_level[base2] = by_base_level.get(base2, 0.0j) + contribution
        rows.append(
            {
                "n": [show(value) for value in labels],
                "base_twice_level": base2,
                "maximum_descendant_level": remaining,
                "branching_terms": branch_rows,
                "virasoro_copy_1": encode(first_vir),
                "virasoro_copy_2": encode(second_vir),
                "virasoro_task_seconds": float(task_seconds),
                "reduced_contribution": encode(contribution),
            }
        )
    assembly_seconds = time.perf_counter() - started

    started = time.perf_counter()
    vacuum_seed = genus2_vacuum_seed_schottky(
        *q_values,
        max_word_len=7,
        oscillator_level_max=cutoff,
    )
    vacuum_seconds = time.perf_counter() - started
    value = vacuum_seed * vacuum_seed * reduced_value
    total_seconds = time.perf_counter() - started_total

    result: dict[str, object] = {
        "calculation": "full three-variable enlarged Ramond block",
        "block": "widehat F_0^(+,+)",
        "total_level_cutoff": cutoff,
        "parameters": {
            "b": b,
            "P": list(momenta),
            "q": list(q_values),
            "eta_tubes": [1, 1, 1],
            "three_point_eta": [1, 1],
            "fermion_parity": 0,
        },
        "method": {
            "branching": "first L1 Ward identity with direct low-level boundary anchors",
            "ordinary_blocks": "CCY central-charge recursion",
            "pbw_block_sum_used": False,
            "virasoro_workers": workers,
            "vacuum_word_length": 7,
            "vacuum_oscillator_level": cutoff,
        },
        "counts": {
            "branch_label_triples": len(triples),
            "ordinary_virasoro_blocks": 2 * len(triples),
            "action_decompositions": len(branching.action_diagnostics),
        },
        "timing_seconds": {
            "branch_action_decompositions": action_seconds,
            "branching_ward_systems": ward_seconds,
            "ordinary_virasoro_c_recursions": virasoro_seconds,
            "assembly": assembly_seconds,
            "vacuum_seed": vacuum_seconds,
            "total": total_seconds,
            "sum_of_individual_virasoro_task_times": float(
                sum(value[3] for value in ordinary_results.values())
            ),
        },
        "diagnostics": {
            "maximum_action_fit_relative_residual": max(
                item["relative_residual"] for item in branching.action_diagnostics
            ),
            "ward_systems": ward_diagnostics,
        },
        "reduced_value_without_two_vacuum_seeds": encode(reduced_value),
        "virasoro_vacuum_seed": encode(vacuum_seed),
        "full_value": encode(value),
        "contribution_by_base_twice_level": [
            {"base_twice_level": level, "value": encode(by_base_level[level])}
            for level in sorted(by_base_level)
        ],
        "branch_triples": rows,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"full value: {value.real:+.15e}{value.imag:+.15e}j", flush=True)
    print(f"total runtime: {total_seconds:.6f} s", flush=True)
    print(f"wrote {output}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff", type=int, default=10)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--json", type=Path, default=HERE / "level10_results.json")
    arguments = parser.parse_args()
    if arguments.cutoff < 0:
        raise ValueError("cutoff must be nonnegative")
    if arguments.workers < 1:
        raise ValueError("workers must be positive")
    run(arguments.cutoff, arguments.workers, arguments.json)


if __name__ == "__main__":
    main()
