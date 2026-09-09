#!/usr/bin/env python3
"""Reduce finite-q table periods to the genus-two fundamental domain.

Every production table row retains two pieces of atlas provenance:

* a target period matrix already in Gottschling's fundamental domain; and
* the exact finite-depth ``Sp(4,Z)`` marking used to enter the plumbing frame.

The finite-q period is first transported back with the inverse atlas marking,
after removing its symmetric integral B-period branch.  It is consequently
already close to the target fundamental representative.  A short symplectic
wall search then handles rows whose finite-q correction crosses a boundary.
The returned matrix is accepted only after exact symplectic and numerical
fundamental-domain certificates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:
    from bolza_torus_plumbing_reach import enumerate_symplectic_words, transform_omega
    from genus2_hybrid_period_map import period_max_residual
    from genus2_siegel_fundamental_domain import (
        gottschling_min_margin,
        in_gottschling_domain,
    )
    from liouville_genus2 import parse_complex
except ImportError:  # pragma: no cover
    from plumbing.bolza_torus_plumbing_reach import (
        enumerate_symplectic_words,
        transform_omega,
    )
    from plumbing.genus2_hybrid_period_map import period_max_residual
    from plumbing.genus2_siegel_fundamental_domain import (
        gottschling_min_margin,
        in_gottschling_domain,
    )
    from plumbing.liouville_genus2 import parse_complex


J4 = np.block(
    [
        [np.zeros((2, 2), dtype=np.int64), np.eye(2, dtype=np.int64)],
        [-np.eye(2, dtype=np.int64), np.zeros((2, 2), dtype=np.int64)],
    ]
)


def symmetric_omega(omega11: complex, omega12: complex, omega22: complex) -> np.ndarray:
    return np.asarray(
        [[omega11, omega12], [omega12, omega22]], dtype=np.complex128
    )


def omega_from_row(row: dict[str, str], prefix: str = "omega") -> np.ndarray:
    return symmetric_omega(
        parse_complex(row[f"{prefix}11"]),
        parse_complex(row[f"{prefix}12"]),
        parse_complex(row[f"{prefix}22"]),
    )


def symplectic_inverse(matrix: Sequence[Sequence[int]]) -> np.ndarray:
    """Return the exact integer inverse of an ``Sp(4,Z)`` matrix."""

    value = np.asarray(matrix, dtype=np.int64)
    if value.shape != (4, 4):
        raise ValueError(f"symplectic matrix must have shape (4,4), got {value.shape}")
    if not np.array_equal(value.T @ J4 @ value, J4):
        raise ValueError("matrix is not symplectic")
    inverse = -J4 @ value.T @ J4
    if not np.array_equal(inverse @ value, np.eye(4, dtype=np.int64)):
        raise ValueError("exact symplectic inverse failed")
    return inverse


def b_shift_matrix(branch: Sequence[Sequence[int]]) -> np.ndarray:
    """Return the symplectic matrix acting as ``Omega -> Omega + branch``."""

    value = np.asarray(branch, dtype=np.int64)
    if value.shape != (2, 2) or not np.array_equal(value, value.T):
        raise ValueError("B-period branch must be a symmetric 2x2 integer matrix")
    matrix = np.eye(4, dtype=np.int64)
    matrix[:2, 2:] = value
    if not np.array_equal(matrix.T @ J4 @ matrix, J4):
        raise ValueError("constructed B-period shift is not symplectic")
    return matrix


@lru_cache(maxsize=None)
def words_and_matrices(depth: int) -> tuple[tuple[str, np.ndarray], ...]:
    return tuple(enumerate_symplectic_words(int(depth)))


def _matrix_fields(prefix: str, matrix: np.ndarray) -> dict[str, int]:
    return {
        f"{prefix}_{row}{column}": int(matrix[row, column])
        for row in range(4)
        for column in range(4)
    }


def _complex_string(value: complex) -> str:
    return f"{value.real:+.17e}{value.imag:+.17e}j"


def reduce_table_row(
    row: dict[str, str],
    *,
    correction_depth: int = 3,
    domain_tolerance: float = 2.0e-9,
    transform_tolerance: float = 2.0e-9,
    accept_first_certified: bool = False,
) -> dict[str, object]:
    """Return a certified fundamental representative for one table row."""

    row_id = str(row["row_id"])
    raw = omega_from_row(row, "omega")
    atlas_target = omega_from_row(row, "atlas_target_omega")
    search_depth = int(row["atlas_search_depth"])
    marking_index = int(row["atlas_marking_matrix_index"])
    markings = words_and_matrices(search_depth)
    if not 0 <= marking_index < len(markings):
        raise ValueError(
            f"{row_id}: atlas marking index {marking_index} is outside depth {search_depth}"
        )
    marking_word, marking = markings[marking_index]
    if str(row["atlas_marking_word"]) != marking_word:
        raise ValueError(f"{row_id}: atlas marking word/index mismatch")

    expected_local = transform_omega(marking, atlas_target)
    expected_local = 0.5 * (expected_local + expected_local.T)
    branch = np.rint((raw - expected_local).real).astype(np.int64)
    branch = np.rint(0.5 * (branch + branch.T)).astype(np.int64)

    # raw -> raw - branch -> inverse atlas marking -> near-fundamental matrix.
    pre_reduction = symplectic_inverse(marking) @ b_shift_matrix(-branch)
    near_fundamental = transform_omega(pre_reduction, raw)
    near_fundamental = 0.5 * (near_fundamental + near_fundamental.T)

    near_margin = float(gottschling_min_margin(near_fundamental))
    candidates = words_and_matrices(int(correction_depth))
    best: tuple[float, int, str, np.ndarray, np.ndarray] | None = None
    if near_margin >= -float(domain_tolerance):
        correction_word, correction = candidates[0]
        best = (near_margin, 0, correction_word, correction, near_fundamental)
    for correction_index, (correction_word, correction) in (
        () if best is not None else enumerate(candidates)
    ):
        candidate = transform_omega(correction, near_fundamental)
        candidate = 0.5 * (candidate + candidate.T)
        margin = float(gottschling_min_margin(candidate))
        if margin < -float(domain_tolerance):
            continue
        if accept_first_certified:
            best = (margin, correction_index, correction_word, correction, candidate)
            break
        # Prefer the most interior certified representative.  The index is a
        # deterministic breadth-first tie breaker on boundary orbits.
        key = (margin, -correction_index)
        if best is None or key > (best[0], -best[1]):
            best = (margin, correction_index, correction_word, correction, candidate)

    if best is None:
        return {
            "row_id": row_id,
            "fd_status": "failed",
            "fd_failure": (
                f"no Gottschling representative through correction depth "
                f"{correction_depth}; near margin={near_margin:.3e}"
            ),
            "fd_correction_depth_limit": int(correction_depth),
            "fd_near_margin": near_margin,
        }

    margin, correction_index, correction_word, correction, omega_fund = best
    total = np.asarray(correction, dtype=np.int64) @ pre_reduction
    transformed = transform_omega(total, raw)
    transformed = 0.5 * (transformed + transformed.T)
    residual = float(np.max(np.abs(transformed - omega_fund)))
    symplectic_error = int(np.max(np.abs(total.T @ J4 @ total - J4)))
    positive_floor = float(np.min(np.linalg.eigvalsh(omega_fund.imag)))
    certified = bool(
        residual <= float(transform_tolerance)
        and symplectic_error == 0
        and positive_floor > 0.0
        and in_gottschling_domain(omega_fund, tolerance=domain_tolerance)
    )
    return {
        "row_id": row_id,
        "fd_status": "ok" if certified else "failed",
        "fd_failure": "" if certified else "post-reduction certificate failed",
        "fd_omega11": _complex_string(complex(omega_fund[0, 0])),
        "fd_omega12": _complex_string(complex(omega_fund[0, 1])),
        "fd_omega22": _complex_string(complex(omega_fund[1, 1])),
        "fd_domain_margin": margin,
        "fd_im_omega_min_eigenvalue": positive_floor,
        "fd_raw_to_fund_residual": residual,
        "fd_symplectic_error": symplectic_error,
        "fd_branch11": int(branch[0, 0]),
        "fd_branch12": int(branch[0, 1]),
        "fd_branch22": int(branch[1, 1]),
        "fd_atlas_marking_word": marking_word,
        "fd_atlas_marking_index": marking_index,
        "fd_atlas_search_depth": search_depth,
        "fd_correction_word": correction_word,
        "fd_correction_index": correction_index,
        "fd_correction_depth_limit": int(correction_depth),
        "fd_near_margin": near_margin,
        **_matrix_fields("fd_sp4", total),
    }


def reduce_table_row_adaptive(
    row: dict[str, str],
    *,
    correction_depths: Sequence[int] = (3, 5, 6, 7),
    domain_tolerance: float = 2.0e-9,
    transform_tolerance: float = 2.0e-9,
    accept_first_certified: bool = False,
) -> dict[str, object]:
    """Raise the local wall-search depth only for unresolved rows."""

    attempted: list[int] = []
    result: dict[str, object] | None = None
    for depth in correction_depths:
        attempted.append(int(depth))
        result = reduce_table_row(
            row,
            correction_depth=int(depth),
            domain_tolerance=domain_tolerance,
            transform_tolerance=transform_tolerance,
            accept_first_certified=accept_first_certified,
        )
        if result["fd_status"] == "ok":
            break
    assert result is not None
    result["fd_attempted_depths"] = ",".join(str(depth) for depth in attempted)
    return result


def run(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", type=Path)
    parser.add_argument("--row-count", type=int, default=20)
    parser.add_argument("--row-offset", type=int, default=0)
    parser.add_argument("--correction-depth", type=int, default=3)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)

    selected: list[dict[str, str]] = []
    with args.table.open(newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            if index < args.row_offset:
                continue
            if len(selected) >= args.row_count:
                break
            selected.append(row)
    results = [
        reduce_table_row(row, correction_depth=args.correction_depth) for row in selected
    ]
    successful = [row for row in results if row["fd_status"] == "ok"]
    summary = {
        "selected_count": len(results),
        "successful_count": len(successful),
        "failed_count": len(results) - len(successful),
        "maximum_transform_residual": max(
            (float(row["fd_raw_to_fund_residual"]) for row in successful), default=math.inf
        ),
        "minimum_domain_margin": min(
            (float(row["fd_domain_margin"]) for row in successful), default=-math.inf
        ),
        "results": results,
    }
    print(json.dumps(summary, indent=2))
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    run()
