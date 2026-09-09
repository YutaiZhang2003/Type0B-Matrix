#!/usr/bin/env python3
"""Checks for the table-anchored genus-two fundamental reduction."""

from __future__ import annotations

import numpy as np

try:
    from bolza_torus_plumbing_reach import enumerate_symplectic_words, transform_omega
    from genus2_siegel_fundamental_domain import in_gottschling_domain, sample_invariant_domain
    from genus2_table_fundamental_reduction import (
        J4,
        b_shift_matrix,
        omega_from_row,
        reduce_table_row,
        reduce_table_row_adaptive,
        symplectic_inverse,
    )
except ImportError:  # pragma: no cover
    from plumbing.bolza_torus_plumbing_reach import (
        enumerate_symplectic_words,
        transform_omega,
    )
    from plumbing.genus2_siegel_fundamental_domain import (
        in_gottschling_domain,
        sample_invariant_domain,
    )
    from plumbing.genus2_table_fundamental_reduction import (
        J4,
        b_shift_matrix,
        omega_from_row,
        reduce_table_row,
        reduce_table_row_adaptive,
        symplectic_inverse,
    )


def _format(value: complex) -> str:
    return f"{value.real:+.17e}{value.imag:+.17e}j"


def _row(target: np.ndarray, raw: np.ndarray, word: str, index: int, depth: int) -> dict[str, str]:
    return {
        "row_id": "synthetic",
        "omega11": _format(raw[0, 0]),
        "omega12": _format(raw[0, 1]),
        "omega22": _format(raw[1, 1]),
        "atlas_target_omega11": _format(target[0, 0]),
        "atlas_target_omega12": _format(target[0, 1]),
        "atlas_target_omega22": _format(target[1, 1]),
        "atlas_marking_word": word,
        "atlas_marking_matrix_index": str(index),
        "atlas_search_depth": str(depth),
    }


def run() -> None:
    target = sample_invariant_domain(1, seed=20260719).omega[0]
    words = enumerate_symplectic_words(2)
    index = 37
    word, marking = words[index]
    inverse = symplectic_inverse(marking)
    assert np.array_equal(inverse @ marking, np.eye(4, dtype=np.int64))
    assert np.array_equal(marking.T @ J4 @ marking, J4)

    branch = np.asarray([[2, -1], [-1, 3]], dtype=np.int64)
    shift = b_shift_matrix(branch)
    raw = transform_omega(shift @ marking, target)
    result = reduce_table_row(_row(target, raw, word, index, 2), correction_depth=2)
    assert result["fd_status"] == "ok", result
    reduced = omega_from_row(result, "fd_omega")
    assert in_gottschling_domain(reduced, tolerance=2.0e-9)
    assert np.max(np.abs(reduced - target)) < 2.0e-10
    assert int(result["fd_branch11"]) == 2
    assert int(result["fd_branch12"]) == -1
    assert int(result["fd_branch22"]) == 3
    assert float(result["fd_raw_to_fund_residual"]) < 1.0e-12
    assert int(result["fd_symplectic_error"]) == 0
    adaptive = reduce_table_row_adaptive(_row(target, raw, word, index, 2))
    assert adaptive["fd_status"] == "ok"
    assert adaptive["fd_attempted_depths"] == "3"
    print("table fundamental-reduction checks passed")


if __name__ == "__main__":
    run()
