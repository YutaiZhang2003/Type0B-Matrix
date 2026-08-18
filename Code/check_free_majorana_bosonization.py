#!/usr/bin/env python3
"""Check bosonization of the human-note auxiliary Majorana block.

The human note orders the theta-trinion slots as ``(infinity, one, zero)``
and uses the quadratic sewing sign in its Eq. (6.16), with no extra linear
edge sign.  Let ``T`` multiply the coefficient at edge-parity ``p`` by that
same quadratic sign.  The bosonization identity for one chiral Majorana is

    (T F_F(q, eta))^2
        = theta[00|beta(eta)](0|Omega_Theta) * F_H(q),

where ``F_H`` is the Casimir-stripped rank-one Heisenberg block in the same
plumbing trivialization.  The square root is the branch with constant term 1.
Equivalently,

    F_F = T sqrt(theta * F_H).

This script compares that formula against the direct Fock/Pfaffian definition
of Eq. (6.16) in two ways.  It first uses independently generated period
matrices and Heisenberg products stored in the genus-two reference summary.
It then recomputes both quantities at a well-conditioned plumbing point.
No relative normalization or characteristic is fitted.
"""

from __future__ import annotations

import argparse
import cmath
import itertools
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
PYTHON_DIRECTORY = Path(__file__).resolve().parent / "python"
sys.path.insert(0, str(PYTHON_DIRECTORY))

from free_boson_plumbing import (  # noqa: E402
    riemann_theta_constant_genus2,
    theta_free_boson_product,
)
from plumbing_algorithms import (  # noqa: E402
    schottky_theta_period_matrix_cross_ratio,
)
from check_free_majorana_resummation import (  # noqa: E402
    direct_human_coefficients,
    human_orientation_sign,
)


DEFAULT_SUMMARY = (
    REPOSITORY
    / "Data Set"
    / "ns_genus2_fivepoint_r20_24_n8_12_axis_summary.json"
)


def human_characteristic(
    eta: Sequence[int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    r"""Return the characteristic in the stored theta period marking.

    Human slots are ``(infinity, one, zero)``.  After the Klein transform
    removes the quadratic orientation sign, the geometric lifts are

        (xi_zero, xi_one, xi_infinity) = (eta_3, eta_2, eta_1).

    In the stored theta period marking this gives

        beta_1 = 1[eta_1 eta_3 = -1],
        beta_2 = 1[eta_1 eta_2 = -1].
    """

    if len(eta) != 3 or any(int(value) not in (-1, 1) for value in eta):
        raise ValueError("eta must contain three signs in (infinity,one,zero) order")
    eta_infinity, eta_one, eta_zero = (int(value) for value in eta)
    beta = (
        int(eta_infinity * eta_zero < 0),
        int(eta_infinity * eta_one < 0),
    )
    return (0, 0), beta


def _parse_complex(value: str | float | complex) -> complex:
    return complex(value)


def _klein_transformed_direct_value(
    coefficients: dict[tuple[int, int, int], object],
    q_geometry: Sequence[complex],
    eta_human: Sequence[int],
) -> complex:
    r"""Evaluate ``\mathcal T F_F`` from the direct human coefficients."""

    q_zero, q_one, q_infinity = (complex(value) for value in q_geometry)
    q_human = (q_infinity, q_one, q_zero)
    x_human = tuple(
        int(sign) * cmath.sqrt(q)
        for sign, q in zip(eta_human, q_human)
    )
    return complex(
        sum(
            complex(coefficient * human_orientation_sign(levels))
            * x_human[0] ** levels[0]
            * x_human[1] ** levels[1]
            * x_human[2] ** levels[2]
            for levels, coefficient in coefficients.items()
        )
    )


def _unit_branch_square_root(value: complex) -> complex:
    root = cmath.sqrt(complex(value))
    return root if abs(root - 1.0) <= abs(root + 1.0) else -root


def _comparison_rows(
    *,
    label: str,
    coefficients: dict[tuple[int, int, int], object],
    q_geometry: Sequence[complex],
    omega: np.ndarray,
    heisenberg: complex,
) -> list[tuple[str, tuple[int, int, int], tuple[int, int], float]]:
    rows = []
    for eta_human in itertools.product((-1, 1), repeat=3):
        characteristic = human_characteristic(eta_human)
        theta = riemann_theta_constant_genus2(
            omega,
            characteristic,
            tol=1.0e-15,
        )
        bosonized = _unit_branch_square_root(theta * heisenberg)
        direct = _klein_transformed_direct_value(
            coefficients,
            q_geometry,
            eta_human,
        )
        relative_error = abs(direct / bosonized - 1.0)
        rows.append((label, eta_human, characteristic[1], relative_error))
    return rows


def run(
    summary: Path,
    cutoff: int,
    tolerance: float,
    fresh_word_length: int,
) -> None:
    source = json.loads(summary.read_text())
    coefficients = direct_human_coefficients(cutoff)

    rows: list[tuple[str, tuple[int, int, int], tuple[int, int], float]] = []
    for point in source["config"]["points"]:
        point_id = str(point["id"])
        q_geometry = tuple(
            _parse_complex(value) for value in point["q_values"]["theta"]
        )
        omega = np.asarray(
            [
                [_parse_complex(value) for value in omega_row]
                for omega_row in point["omega"]["theta"]
            ],
            dtype=np.complex128,
        )
        diagnostics = source["free_superfield"][point_id]["theta"]
        heisenberg = cmath.exp(
            complex(
                diagnostics["scalar_chiral_log_real"],
                diagnostics["scalar_chiral_log_imag"],
            )
        )

        rows.extend(
            _comparison_rows(
                label=point_id,
                coefficients=coefficients,
                q_geometry=q_geometry,
                omega=omega,
                heisenberg=heisenberg,
            )
        )

    worst = max(rows, key=lambda row: row[3])
    print(
        "Checked T(human Eq. (6.16)) against "
        "sqrt(theta[00|beta] * F_H)."
    )
    print(f"  total twice-level cutoff: {cutoff}")
    print(f"  reference points: {len(source['config']['points'])}")
    print(f"  eta assignments per point: 8")
    print(f"  comparisons: {len(rows)}")
    print(
        "  worst relative error: "
        f"{worst[3]:.6e} at {worst[0]}, eta={worst[1]}, beta={worst[2]}"
    )
    if worst[3] > tolerance:
        raise AssertionError(
            f"reference bosonization check exceeded tolerance {tolerance:.3e}"
        )

    fresh_q = (0.01 + 0.0j, 0.015 + 0.0j, 0.02 + 0.0j)
    fresh_omega = schottky_theta_period_matrix_cross_ratio(
        *fresh_q,
        max_word_len=fresh_word_length,
    )
    fresh_product = theta_free_boson_product(
        *fresh_q,
        max_word_length=fresh_word_length,
        max_mode=100,
        tolerance=1.0e-16,
    )
    fresh_rows = _comparison_rows(
        label="fresh-stable-point",
        coefficients=coefficients,
        q_geometry=fresh_q,
        omega=fresh_omega,
        heisenberg=cmath.exp(fresh_product.chiral_log_product),
    )
    fresh_worst = max(fresh_rows, key=lambda row: row[3])
    print(
        "  fresh recomputation worst relative error: "
        f"{fresh_worst[3]:.6e} at eta={fresh_worst[1]}, "
        f"beta={fresh_worst[2]}"
    )
    print(f"  fresh Schottky word length: {fresh_word_length}")
    if fresh_worst[3] > tolerance:
        raise AssertionError(
            f"fresh bosonization check exceeded tolerance {tolerance:.3e}"
        )
    print(
        "PASS: the corrected Klein-transformed bosonization identity agrees "
        "with the direct human sewing series at every tested sign assignment."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--cutoff", type=int, default=32)
    parser.add_argument("--tolerance", type=float, default=1.0e-10)
    parser.add_argument("--fresh-word-length", type=int, default=8)
    arguments = parser.parse_args()
    if arguments.cutoff < 0:
        parser.error("--cutoff must be non-negative")
    if arguments.tolerance <= 0:
        parser.error("--tolerance must be positive")
    if arguments.fresh_word_length < 1:
        parser.error("--fresh-word-length must be positive")
    run(
        arguments.summary,
        arguments.cutoff,
        arguments.tolerance,
        arguments.fresh_word_length,
    )
