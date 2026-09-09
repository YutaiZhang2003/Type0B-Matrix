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

import mpmath as mp
import numpy as np
from scipy import sparse
from scipy import linalg as scipy_linalg


HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
REPOSITORY = CODE_ROOT.parent
BRANCHING = CODE_ROOT / "ramond_branching_recursion"

# Yuchen's runtime keeps the Ramond modules beside this folder and imports the
# CCY plumbing package from the StringMC checkout.  Keep that dependency route
# intact; do not fall back to the provisional local PBW/anchor implementation.
STRING_MC_ROOT = Path(
    os.environ.get(
        "TYPE0B_STRINGMC_ROOT",
        REPOSITORY.parent / "Project" / "StringMC",
    )
).expanduser()
for directory in (REPOSITORY, STRING_MC_ROOT, BRANCHING):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from compute_target import (  # noqa: E402
    BranchWeights,
    FreeFieldModule,
    TOLERANCE,
    complex_number,
    norm_product,
    ordinary_factor,
    set_multiprecision,
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

    def __init__(
        self,
        b: float,
        momenta: tuple[complex, complex, complex],
        cutoff: int,
        primary_parity: int = 0,
        mp_dps: int = 0,
    ):
        self.mp_dps = int(mp_dps)
        if self.mp_dps and self.mp_dps < 30:
            raise ValueError("mp_dps must be 0 or at least 30")
        set_multiprecision(self.mp_dps)
        self.b = float(b)
        self.momenta = tuple(complex(value) for value in momenta)
        self.cutoff = int(cutoff)
        self.primary_parity = int(primary_parity)
        if self.primary_parity not in (0, 1):
            raise ValueError("primary_parity must be 0 or 1")
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
        # The direct low-state oracle is independently certified in its
        # original binary64 implementation.  Multiprecision is used for the
        # action decompositions and finite Ward solve, while those boundary
        # data are deliberately kept unchanged.
        if self.mp_dps:
            set_multiprecision(0)
        try:
            self.direct = DirectBranchingCoefficient(
                self.b,
                self.momenta,
                primary_parity=self.primary_parity,
            )
            self.direct.auxiliary_form = RamondAuxiliaryThreePoint(
                self.direct.free_modules
            )
        finally:
            if self.mp_dps:
                set_multiprecision(self.mp_dps)

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

    def solve(
        self,
        alpha2: int,
        alpha3: int,
        *,
        eta: int = 1,
        form_parity: int = 0,
    ) -> tuple[dict[tuple[Fraction, Fraction, Fraction], complex], dict[str, object]]:
        alpha2 = int(alpha2)
        alpha3 = int(alpha3)
        eta = int(eta)
        form_parity = int(form_parity)
        if eta not in (-1, 1):
            raise ValueError("eta must be +/-1")
        if form_parity not in (0, 1):
            raise ValueError("form_parity must be zero or one")
        n1_parity = (form_parity - alpha2 - alpha3) % 2
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
        mp_rows: list[dict[int, object]] = []
        mp_rhs: list[object] = []

        def append(equation: dict[tuple[Fraction, Fraction, Fraction], complex], value: complex) -> None:
            if self.mp_dps:
                row_norm = mp.sqrt(
                    mp.fsum(abs(coefficient) ** 2 for coefficient in equation.values())
                )
                if row_norm <= mp.power(10, -max(20, self.mp_dps - 20)):
                    return
                mp_rows.append(
                    {
                        index[labels]: coefficient / row_norm
                        for labels, coefficient in equation.items()
                    }
                )
                mp_rhs.append(complex_number(value) / row_norm)
                return
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
                    if self.mp_dps:
                        set_multiprecision(0)
                    try:
                        value = self.direct.raw(
                            labels, alpha2, alpha3, eta=eta
                        )
                    finally:
                        if self.mp_dps:
                            set_multiprecision(self.mp_dps)
                    anchors[labels] = value
                    append({labels: 1.0 + 0.0j}, value)

        if self.mp_dps:
            solution, diagnostic = self._solve_multiprecision(
                alpha2,
                alpha3,
                unknowns,
                mp_rows,
                mp_rhs,
                len(anchors),
            )
            diagnostic["eta"] = eta
            diagnostic["form_parity"] = form_parity
            return solution, diagnostic

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
            "eta": eta,
            "form_parity": form_parity,
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

    def _solve_multiprecision(
        self,
        alpha2: int,
        alpha3: int,
        unknowns: tuple[tuple[Fraction, Fraction, Fraction], ...],
        rows: list[dict[int, object]],
        rhs: list[object],
        anchor_count: int,
    ) -> tuple[
        dict[tuple[Fraction, Fraction, Fraction], complex],
        dict[str, object],
    ]:
        """Solve the normalized Ward grid by multiprecision refinement.

        A binary64 QR factorization is only a preconditioner.  Residuals and
        coefficient updates are accumulated at ``self.mp_dps`` decimal
        digits, so the returned result solves the multiprecision Ward matrix,
        rather than merely repeating the binary64 least-squares calculation.
        """

        row_count = len(rows)
        column_count = len(unknowns)
        column_norms = [
            mp.sqrt(
                mp.fsum(abs(row.get(column, 0)) ** 2 for row in rows)
            )
            for column in range(column_count)
        ]
        if any(norm == 0 for norm in column_norms):
            raise AssertionError("a Ward-system column is identically zero")

        scaled_rows = [
            {
                column: coefficient / column_norms[column]
                for column, coefficient in row.items()
            }
            for row in rows
        ]
        shadow = np.zeros((row_count, column_count), dtype=np.complex128)
        for row_index, row in enumerate(scaled_rows):
            for column, coefficient in row.items():
                shadow[row_index, column] = complex(coefficient)
        shadow_rhs = np.asarray(
            [complex(value) for value in rhs], dtype=np.complex128
        )

        initial, _, rank, singular_values = np.linalg.lstsq(
            shadow, shadow_rhs, rcond=1.0e-13
        )
        if rank != column_count:
            raise np.linalg.LinAlgError(
                f"multiprecision Ward grid has rank {rank}/{column_count}"
            )
        q_factor, r_factor = scipy_linalg.qr(
            shadow, mode="economic", check_finite=False
        )
        scaled_solution = [complex_number(value) for value in initial]
        # Boundary anchors are intentionally retained from the independent
        # binary64 PBW oracle.  Asking the iterative correction to fall below
        # their inherited uncertainty is neither attainable nor meaningful;
        # 1e-22 is already safely beyond the accuracy tested by the q-series
        # comparison.
        requested_tolerance = mp.power(
            10, -min(22, max(18, self.mp_dps - 15))
        )
        relative_correction = mp.inf
        refinement_iterations = 0

        for iteration in range(1, 31):
            residual = [
                rhs[row_index]
                - mp.fsum(
                    coefficient * scaled_solution[column]
                    for column, coefficient in row.items()
                )
                for row_index, row in enumerate(scaled_rows)
            ]
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                projected = q_factor.conj().T @ np.asarray(
                    [complex(value) for value in residual], dtype=np.complex128
                )
            if not np.all(np.isfinite(projected)):
                raise FloatingPointError(
                    "multiprecision Ward refinement produced a non-finite correction"
                )
            correction = scipy_linalg.solve_triangular(
                r_factor, projected, check_finite=False
            )
            correction_norm = mp.sqrt(
                mp.fsum(abs(value) ** 2 for value in correction)
            )
            solution_norm = mp.sqrt(
                mp.fsum(abs(value) ** 2 for value in scaled_solution)
            )
            relative_correction = correction_norm / max(solution_norm, 1)
            scaled_solution = [
                value + complex_number(delta)
                for value, delta in zip(scaled_solution, correction)
            ]
            refinement_iterations = iteration
            if relative_correction <= requested_tolerance:
                break
        else:
            raise FloatingPointError(
                "multiprecision Ward refinement did not converge: "
                f"relative correction {float(relative_correction):.3e}"
            )

        values = [
            scaled_solution[column] / column_norms[column]
            for column in range(column_count)
        ]
        residual = [
            mp.fsum(
                coefficient * values[column]
                for column, coefficient in row.items()
            )
            - rhs[row_index]
            for row_index, row in enumerate(rows)
        ]
        residual_norm = mp.sqrt(mp.fsum(abs(value) ** 2 for value in residual))
        vector_norm = mp.sqrt(mp.fsum(abs(value) ** 2 for value in rhs))
        solution = {
            labels: complex(values[position])
            for position, labels in enumerate(unknowns)
        }
        diagnostic = {
            "alpha_2": alpha2,
            "alpha_3": alpha3,
            "unknowns": column_count,
            "equations": row_count,
            "anchors": anchor_count,
            "rank": int(rank),
            "full_column_rank": True,
            "absolute_residual": float(residual_norm),
            "relative_residual": float(residual_norm / max(vector_norm, 1)),
            "smallest_retained_singular_value": float(singular_values[-1]),
            "condition_number": float(singular_values[0] / singular_values[-1]),
            "solver": f"mixed-precision-qr-refinement-{self.mp_dps}dps",
            "refinement_iterations": refinement_iterations,
            "relative_final_correction": float(relative_correction),
            "boundary_anchor_backend": "certified-direct-pbw-complex128",
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


def run(
    cutoff: int,
    workers: int,
    output: Path,
    *,
    primary_parity: int = 0,
) -> dict[str, object]:
    b = 7.0 / 5.0
    momenta = (11.0 / 23.0, 13.0 / 29.0, 17.0 / 31.0)
    q_values = (0.019, 0.023, 0.029)
    started_total = time.perf_counter()

    branching = BranchingGrid(
        b, momenta, cutoff, primary_parity=primary_parity
    )
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
                (int(2 * n1) + primary_parity) * alpha2
                + (int(2 * n1) + primary_parity) * alpha3
                + alpha2 * alpha3
            )
            sign = (-1) ** exponent
            human_enlarged_sign = (-1) ** int(2 * n1)
            term = (
                human_enlarged_sign * sign * normalized * normalized
            )
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
            "primary_parity": int(primary_parity),
        },
        "method": {
            "branching": "first L1 Ward identity with direct low-level boundary anchors",
            "ordinary_blocks": "CCY central-charge recursion",
            "pbw_block_sum_used": False,
            "virasoro_workers": workers,
            "vacuum_word_length": 7,
            "vacuum_oscillator_level": cutoff,
            "human_note_signs": {
                "enlarged_first_tube": "(-1)^(A+mathsf_A)",
                "double_virasoro_branch": "(-1)^(2*n_1)",
            },
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
    parser.add_argument("--primary-parity", type=int, choices=(0, 1), default=0)
    parser.add_argument("--json", type=Path, default=HERE / "level10_results.json")
    arguments = parser.parse_args()
    if arguments.cutoff < 0:
        raise ValueError("cutoff must be nonnegative")
    if arguments.workers < 1:
        raise ValueError("workers must be positive")
    run(
        arguments.cutoff,
        arguments.workers,
        arguments.json,
        primary_parity=arguments.primary_parity,
    )


if __name__ == "__main__":
    main()
