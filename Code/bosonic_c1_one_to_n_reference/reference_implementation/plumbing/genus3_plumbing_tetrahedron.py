#!/usr/bin/env python3
"""Marked genus-three tetrahedral plumbing period map and inverse.

This module implements one explicit genus-three plumbing chart.  Four
three-punctured spheres are joined according to the tetrahedral graph ``K4``.
The six plumbing parameters are ordered as

    (q01, q02, q03, q12, q13, q23).

The star ``(01, 02, 03)`` is the spanning tree.  The three non-tree edges
``(12, 13, 23)`` define oriented fundamental cycles and hence a marked rank-3
Schottky group.  The forward map uses the exact plumbing Mobius maps and the
finite-word Schottky cross-ratio formula.  The inverse is a local numerical
inverse in this fixed chart and marking; it is not a global ``Sp(6,Z)`` atlas.

The mathematical Schottky relation is an infinite series.  A result is called
certified here only relative to the requested finite-word stability, period
residual, symmetry, and positivity tolerances.  No rigorous analytic bound on
the omitted word tail is asserted.
"""

from __future__ import annotations

import argparse
import cmath
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:
    from genus3_plumbing_channels import (
        genus3_channel_leading_period_slope_matrix,
    )
    from plumbing_algorithms import (
        GeneratorData,
        IDENTITY,
        INF,
        Mobius,
        mobius_fixed_points,
        plumbing_transition,
        schottky_period_matrix_cross_ratio,
    )
except ImportError:  # pragma: no cover - supports package-style execution
    from plumbing.genus3_plumbing_channels import (
        genus3_channel_leading_period_slope_matrix,
    )
    from plumbing.plumbing_algorithms import (
        GeneratorData,
        IDENTITY,
        INF,
        Mobius,
        mobius_fixed_points,
        plumbing_transition,
        schottky_period_matrix_cross_ratio,
    )


TWO_PI_I = 2.0j * math.pi

TETRAHEDRON_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (1, 3),
    (2, 3),
)

# Each row is an oriented fundamental cycle in the edge order above:
# 0 -> 1 -> 2 -> 0, 0 -> 1 -> 3 -> 0, and 0 -> 2 -> 3 -> 0.
TETRAHEDRON_CYCLE_MATRIX = np.asarray(
    [
        [1, -1, 0, 1, 0, 0],
        [1, 0, -1, 0, 1, 0],
        [0, 1, -1, 0, 0, 1],
    ],
    dtype=int,
)

# The chosen local coordinates at 0, 1, and infinity introduce signs in the
# leading Schottky multipliers and cross ratios.  Their logarithms give this
# constant half-twist matrix, defined modulo symmetric integral B-shifts.  We
# choose the +1/2 representative.  Omitting it gives the correct imaginary
# degeneration lengths but the wrong real-period branch for the inverse seed.
TETRAHEDRON_HALF_TWIST = np.asarray(
    [
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
        [0.5, 0.5, 0.0],
    ],
    dtype=np.complex128,
)

TETRAHEDRON_LOOPS: tuple[tuple[int, ...], ...] = (
    (0, 1, 2, 0),
    (0, 1, 3, 0),
    (0, 2, 3, 0),
)

# Puncture assignments on each three-punctured sphere.  These choices define
# the plumbing conformal frame and must be kept fixed when evaluating blocks.
TETRAHEDRON_PUNCTURES: tuple[dict[int, str], ...] = (
    {1: "zero", 2: "one", 3: "infty"},
    {0: "zero", 2: "one", 3: "infty"},
    {0: "zero", 1: "one", 3: "infty"},
    {0: "zero", 1: "one", 2: "infty"},
)

SYMMETRIC_INDEX_ORDER: tuple[tuple[int, int], ...] = (
    (0, 0),
    (1, 1),
    (2, 2),
    (0, 1),
    (0, 2),
    (1, 2),
)


@dataclass(frozen=True)
class Genus3ForwardResult:
    q_values: tuple[complex, ...]
    word_length: int
    omega: np.ndarray
    omega_raw: np.ndarray
    omega_leading: np.ndarray
    correction: np.ndarray
    symmetry_error: float
    minimum_imaginary_eigenvalue: float
    generator_multipliers: tuple[complex, ...]


@dataclass(frozen=True)
class Genus3InverseResult:
    success: bool
    certified: bool
    message: str
    evaluations: int
    q_values: tuple[complex, ...]
    tau_values: tuple[complex, ...]
    target_omega: np.ndarray
    target_branch: np.ndarray
    reconstructed_omega: np.ndarray
    high_word_omega: np.ndarray
    period_max_residual: float
    word_stability: float
    symmetry_error: float
    minimum_imaginary_eigenvalue: float
    max_q_abs: float
    inverse_word_length: int
    validation_word_length: int


def _validate_q_values(q_values: Sequence[complex]) -> tuple[complex, ...]:
    q_tuple = tuple(complex(value) for value in q_values)
    if len(q_tuple) != len(TETRAHEDRON_EDGES):
        raise ValueError("tetrahedral genus-three plumbing expects six q values")
    if any(not (math.isfinite(value.real) and math.isfinite(value.imag)) for value in q_tuple):
        raise ValueError("all tetrahedral plumbing parameters must be finite")
    if any(not 0.0 < abs(value) < 1.0 for value in q_tuple):
        raise ValueError("all tetrahedral plumbing parameters must satisfy 0 < |q_e| < 1")
    return q_tuple


def tetrahedron_transition_maps(q_values: Sequence[complex]) -> dict[tuple[int, int], Mobius]:
    """Return both orientations of all six exact plumbing transition maps."""

    q_tuple = _validate_q_values(q_values)
    q_by_edge = dict(zip(TETRAHEDRON_EDGES, q_tuple))
    transitions: dict[tuple[int, int], Mobius] = {}
    for left, right in TETRAHEDRON_EDGES:
        forward = plumbing_transition(
            TETRAHEDRON_PUNCTURES[left][right],
            TETRAHEDRON_PUNCTURES[right][left],
            q_by_edge[(left, right)],
        )
        transitions[(left, right)] = forward
        transitions[(right, left)] = forward.inv()
    return transitions


def compose_transition_path(
    transitions: dict[tuple[int, int], Mobius],
    vertices: Sequence[int],
) -> Mobius:
    """Compose oriented edge transitions along a closed graph path."""

    if len(vertices) < 2 or vertices[0] != vertices[-1]:
        raise ValueError("a Schottky transition path must be a nonempty closed loop")
    result = IDENTITY
    for source, target in zip(vertices[:-1], vertices[1:]):
        try:
            transition = transitions[(int(source), int(target))]
        except KeyError as exc:
            raise ValueError(f"vertices {source} and {target} are not joined in K4") from exc
        # If the path applies T1 first and then T2, the total map is T2 o T1.
        result = transition.compose(result)
    return result


def _fixed_point_multiplier(transform: Mobius, point: complex | None) -> complex:
    if point is INF:
        if transform.c != 0 or transform.a == 0:
            raise ValueError("infinity was returned as a non-fixed or degenerate point")
        # In the local coordinate u=1/z, gamma has derivative d/a at u=0.
        return transform.d / transform.a
    return transform.deriv(point)


def _generator_preserving_cycle_orientation(transform: Mobius) -> GeneratorData:
    """Orient fixed points while leaving the graph-cycle map itself unchanged."""

    first, second = mobius_fixed_points(transform)
    first_multiplier = _fixed_point_multiplier(transform, first)
    second_multiplier = _fixed_point_multiplier(transform, second)
    if abs(first_multiplier) <= abs(second_multiplier):
        attracting, repelling, multiplier = first, second, first_multiplier
    else:
        attracting, repelling, multiplier = second, first, second_multiplier
    if not (math.isfinite(multiplier.real) and math.isfinite(multiplier.imag)):
        raise ValueError("nonfinite tetrahedral Schottky multiplier")
    if not 0.0 < abs(multiplier) < 1.0:
        raise ValueError(
            "tetrahedral loop did not define a loxodromic generator with an "
            f"attracting multiplier inside the unit disk: |k|={abs(multiplier):.6g}"
        )
    return GeneratorData(
        gamma=transform,
        attracting=attracting,
        repelling=repelling,
        multiplier=multiplier,
    )


def generators_for_tetrahedron(q_values: Sequence[complex]) -> list[GeneratorData]:
    """Construct the three marked rank-3 Schottky generators from six q values."""

    transitions = tetrahedron_transition_maps(q_values)
    return [
        _generator_preserving_cycle_orientation(compose_transition_path(transitions, loop))
        for loop in TETRAHEDRON_LOOPS
    ]


def genus3_symmetric_period_vector(omega: Sequence[Sequence[complex]]) -> np.ndarray:
    matrix = np.asarray(omega, dtype=np.complex128)
    if matrix.shape != (3, 3):
        raise ValueError(f"expected a 3x3 period matrix, got shape {matrix.shape}")
    return np.asarray([matrix[i, j] for i, j in SYMMETRIC_INDEX_ORDER], dtype=np.complex128)


def genus3_real_period_vector(omega: Sequence[Sequence[complex]]) -> np.ndarray:
    vector = genus3_symmetric_period_vector(omega)
    return np.concatenate((vector.real, vector.imag))


def tetrahedron_leading_period_matrix(q_values: Sequence[complex]) -> np.ndarray:
    """Return the tropical period matrix, including the chart half twists."""

    q_tuple = _validate_q_values(q_values)
    edge_taus = np.asarray([cmath.log(value) / TWO_PI_I for value in q_tuple], dtype=np.complex128)
    cycles = TETRAHEDRON_CYCLE_MATRIX.astype(np.complex128)
    return cycles @ np.diag(edge_taus) @ cycles.T + TETRAHEDRON_HALF_TWIST


def _leading_period_linear_map() -> np.ndarray:
    """Return the shared all-channel slope specialized to the K4 marking."""

    return np.asarray(
        genus3_channel_leading_period_slope_matrix("tetrahedron"),
        dtype=float,
    )


def tetrahedron_leading_taus_from_omega(
    omega: Sequence[Sequence[complex]],
) -> tuple[complex, ...]:
    """Invert the six-dimensional tropical period map in this fixed marking."""

    reduced = np.asarray(omega, dtype=np.complex128) - TETRAHEDRON_HALF_TWIST
    edge_taus = np.linalg.solve(
        _leading_period_linear_map(),
        genus3_symmetric_period_vector(reduced),
    )
    return tuple(complex(value) for value in edge_taus)


def tetrahedron_leading_q_from_omega(
    omega: Sequence[Sequence[complex]],
) -> tuple[complex, ...]:
    return tuple(cmath.exp(TWO_PI_I * tau) for tau in tetrahedron_leading_taus_from_omega(omega))


def _symmetrize_period_branches(omega: np.ndarray) -> tuple[np.ndarray, float]:
    """Align integral B-shift branches before imposing exact symmetry."""

    raw = np.asarray(omega, dtype=np.complex128)
    if raw.shape != (3, 3):
        raise ValueError("a genus-three period matrix must have shape (3,3)")
    result = raw.copy()
    symmetry_error = 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            upper = complex(raw[i, j])
            lower = complex(raw[j, i])
            branch = int(round((upper - lower).real))
            lower_aligned = lower + branch
            symmetry_error = max(symmetry_error, abs(upper - lower_aligned))
            value = 0.5 * (upper + lower_aligned)
            result[i, j] = value
            result[j, i] = value
    return result, float(symmetry_error)


def period_difference_mod_symmetric_integer(
    left: Sequence[Sequence[complex]],
    right: Sequence[Sequence[complex]],
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``left-right`` after the nearest symmetric integral B shift."""

    difference = np.asarray(left, dtype=np.complex128) - np.asarray(right, dtype=np.complex128)
    branch = np.rint(difference.real).astype(int)
    branch = np.rint(0.5 * (branch + branch.T)).astype(int)
    return difference - branch, branch


def tetrahedron_schottky_forward(
    q_values: Sequence[complex],
    *,
    word_length: int = 3,
) -> Genus3ForwardResult:
    """Evaluate the marked finite-word genus-three Schottky period map."""

    q_tuple = _validate_q_values(q_values)
    if int(word_length) < 0:
        raise ValueError("word_length must be nonnegative")
    generators = generators_for_tetrahedron(q_tuple)
    omega_raw = schottky_period_matrix_cross_ratio(generators, max_word_len=int(word_length))
    omega, symmetry_error = _symmetrize_period_branches(omega_raw)
    if not np.all(np.isfinite(omega)):
        raise FloatingPointError("tetrahedral Schottky period matrix is nonfinite")
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(omega.imag)))
    leading = tetrahedron_leading_period_matrix(q_tuple)
    correction, _ = period_difference_mod_symmetric_integer(omega, leading)
    return Genus3ForwardResult(
        q_values=q_tuple,
        word_length=int(word_length),
        omega=omega,
        omega_raw=np.asarray(omega_raw, dtype=np.complex128),
        omega_leading=leading,
        correction=correction,
        symmetry_error=symmetry_error,
        minimum_imaginary_eigenvalue=minimum_eigenvalue,
        generator_multipliers=tuple(complex(generator.multiplier) for generator in generators),
    )


def _pack_taus(tau_values: Sequence[complex]) -> np.ndarray:
    return np.asarray(
        [component for value in tau_values for component in (complex(value).real, complex(value).imag)],
        dtype=float,
    )


def _unpack_taus(x: Sequence[float]) -> tuple[complex, ...]:
    values = tuple(float(value) for value in x)
    if len(values) != 12:
        raise ValueError("the genus-three inverse expects twelve real variables")
    return tuple(complex(values[index], values[index + 1]) for index in range(0, 12, 2))


def _q_from_taus(tau_values: Sequence[complex]) -> tuple[complex, ...]:
    return tuple(cmath.exp(TWO_PI_I * complex(value)) for value in tau_values)


def invert_tetrahedron_period_matrix(
    target_omega: Sequence[Sequence[complex]],
    *,
    word_length: int = 3,
    validation_word_step: int = 1,
    max_nfev: int = 300,
    period_tolerance: float = 1.0e-8,
    stability_tolerance: float = 1.0e-8,
    symmetry_tolerance: float = 1.0e-8,
    min_tau_imag: float = 1.0e-8,
    max_tau_imag: float = 50.0,
) -> Genus3InverseResult:
    """Locally invert and certify ``Omega -> q -> Omega`` in the fixed chart."""

    from scipy.optimize import least_squares

    target = np.asarray(target_omega, dtype=np.complex128)
    if target.shape != (3, 3):
        raise ValueError(f"expected a 3x3 target period matrix, got shape {target.shape}")
    if not np.all(np.isfinite(target)):
        raise ValueError("target period matrix must be finite")
    if not math.isfinite(float(symmetry_tolerance)) or float(symmetry_tolerance) < 0.0:
        raise ValueError("symmetry_tolerance must be finite and nonnegative")
    target_symmetry_error = float(np.max(np.abs(target - target.T)))
    if target_symmetry_error > float(symmetry_tolerance):
        raise ValueError(
            f"target period matrix symmetry defect {target_symmetry_error:.6e} "
            f"exceeds tolerance {float(symmetry_tolerance):.6e}"
        )
    target = 0.5 * (target + target.T)
    if float(np.min(np.linalg.eigvalsh(target.imag))) <= 0.0:
        raise ValueError("target period matrix must have positive-definite imaginary part")
    if int(validation_word_step) < 1:
        raise ValueError("validation_word_step must be positive")

    seed_taus = list(tetrahedron_leading_taus_from_omega(target))
    for index, tau in enumerate(seed_taus):
        seed_taus[index] = complex(tau.real - round(tau.real), tau.imag)
    if any(not min_tau_imag < tau.imag < max_tau_imag for tau in seed_taus):
        raise ValueError(
            "the leading tetrahedral inverse is outside the allowed plumbing strip; "
            "a different symplectic marking or plumbing chart is required"
        )
    # The individual logarithms in the Schottky formula use principal
    # branches.  A finite-difference probe can therefore change Omega by an
    # integral symmetric B-shift even when the underlying marked surface moves
    # smoothly.  Align every trial point dynamically rather than freezing the
    # branch at the initial seed.

    x0 = _pack_taus(seed_taus)
    lower = np.asarray(
        [component for _ in range(6) for component in (-2.0, float(min_tau_imag))],
        dtype=float,
    )
    upper = np.asarray(
        [component for _ in range(6) for component in (2.0, float(max_tau_imag))],
        dtype=float,
    )
    x0 = np.minimum(np.maximum(x0, lower), upper)

    def residual(x: np.ndarray) -> np.ndarray:
        tau_values = _unpack_taus(x)
        q_values = _q_from_taus(tau_values)
        try:
            forward = tetrahedron_schottky_forward(q_values, word_length=word_length)
        except Exception:
            return 1.0e6 * np.ones(12, dtype=float)
        difference, _ = period_difference_mod_symmetric_integer(forward.omega, target)
        return genus3_real_period_vector(difference)

    optimizer = least_squares(
        residual,
        x0,
        bounds=(lower, upper),
        max_nfev=int(max_nfev),
        diff_step=1.0e-5,
        x_scale="jac",
        xtol=1.0e-11,
        ftol=1.0e-11,
        gtol=1.0e-11,
    )

    tau_values = _unpack_taus(optimizer.x)
    q_values = _q_from_taus(tau_values)
    low = tetrahedron_schottky_forward(q_values, word_length=word_length)
    validation_word_length = int(word_length) + int(validation_word_step)
    high = tetrahedron_schottky_forward(q_values, word_length=validation_word_length)
    high_difference, target_branch = period_difference_mod_symmetric_integer(high.omega, target)
    low_high_difference, _ = period_difference_mod_symmetric_integer(high.omega, low.omega)
    period_residual = float(np.max(np.abs(high_difference)))
    word_stability = float(np.max(np.abs(low_high_difference)))
    symmetry_error = max(low.symmetry_error, high.symmetry_error)
    minimum_eigenvalue = min(
        low.minimum_imaginary_eigenvalue,
        high.minimum_imaginary_eigenvalue,
    )
    certified = bool(
        optimizer.success
        and period_residual <= float(period_tolerance)
        and word_stability <= float(stability_tolerance)
        and symmetry_error <= float(symmetry_tolerance)
        and minimum_eigenvalue > 0.0
    )
    return Genus3InverseResult(
        success=bool(optimizer.success),
        certified=certified,
        message=str(optimizer.message),
        evaluations=int(optimizer.nfev),
        q_values=q_values,
        tau_values=tau_values,
        target_omega=target,
        target_branch=target_branch,
        reconstructed_omega=low.omega,
        high_word_omega=high.omega,
        period_max_residual=period_residual,
        word_stability=word_stability,
        symmetry_error=float(symmetry_error),
        minimum_imaginary_eigenvalue=float(minimum_eigenvalue),
        max_q_abs=max(float(abs(value)) for value in q_values),
        inverse_word_length=int(word_length),
        validation_word_length=validation_word_length,
    )


def _parse_q_list(value: str) -> tuple[complex, ...]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != 6:
        raise argparse.ArgumentTypeError("--q-values requires six comma-separated complex numbers")
    try:
        return tuple(complex(part.replace("i", "j")) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _complex_payload(value: complex) -> dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag)}


def _matrix_payload(matrix: np.ndarray) -> list[list[dict[str, float]]]:
    return [[_complex_payload(complex(value)) for value in row] for row in np.asarray(matrix)]


def _result_payload(
    q_truth: Sequence[complex],
    target: Genus3ForwardResult,
    inverse: Genus3InverseResult,
) -> dict[str, object]:
    scalar_fields = {
        key: value
        for key, value in asdict(inverse).items()
        if key
        not in {
            "q_values",
            "tau_values",
            "target_omega",
            "target_branch",
            "reconstructed_omega",
            "high_word_omega",
        }
    }
    return {
        "chart": "genus3-tetrahedron-k4-fixed-marking",
        "edge_order": [list(edge) for edge in TETRAHEDRON_EDGES],
        "q_truth": [_complex_payload(complex(value)) for value in q_truth],
        "target_word_length": int(target.word_length),
        "target_omega": _matrix_payload(target.omega),
        "inverse": {
            **scalar_fields,
            "q_values": [_complex_payload(value) for value in inverse.q_values],
            "tau_values": [_complex_payload(value) for value in inverse.tau_values],
            "target_branch": np.asarray(inverse.target_branch, dtype=int).tolist(),
            "reconstructed_omega": _matrix_payload(inverse.reconstructed_omega),
            "high_word_omega": _matrix_payload(inverse.high_word_omega),
        },
    }


def _print_matrix(label: str, matrix: np.ndarray) -> None:
    print(label)
    for row in np.asarray(matrix):
        print("  " + "  ".join(f"{value.real:+.12e}{value.imag:+.12e}j" for value in row))


def run(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Validate the fixed-marking genus-three tetrahedral map Omega -> q -> Omega."
    )
    parser.add_argument(
        "--q-values",
        type=_parse_q_list,
        default=_parse_q_list("0.012+0.001j,0.010-0.001j,0.014+0j,0.011+0.001j,0.009-0.001j,0.013+0j"),
    )
    parser.add_argument("--target-word-length", type=int, default=4)
    parser.add_argument("--inverse-word-length", type=int, default=3)
    parser.add_argument("--validation-word-step", type=int, default=1)
    parser.add_argument("--max-nfev", type=int, default=300)
    parser.add_argument("--period-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--stability-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)

    q_truth = _validate_q_values(args.q_values)
    target = tetrahedron_schottky_forward(q_truth, word_length=args.target_word_length)
    inverse = invert_tetrahedron_period_matrix(
        target.omega,
        word_length=args.inverse_word_length,
        validation_word_step=args.validation_word_step,
        max_nfev=args.max_nfev,
        period_tolerance=args.period_tolerance,
        stability_tolerance=args.stability_tolerance,
    )

    print("Genus-three tetrahedral plumbing round trip")
    print("  edge order: " + ", ".join(str(edge) for edge in TETRAHEDRON_EDGES))
    print("  q truth:     " + "  ".join(f"{value.real:+.8e}{value.imag:+.8e}j" for value in q_truth))
    print("  q recovered: " + "  ".join(f"{value.real:+.8e}{value.imag:+.8e}j" for value in inverse.q_values))
    _print_matrix("  target Omega:", target.omega)
    _print_matrix("  reconstructed high-word Omega:", inverse.high_word_omega)
    print(f"  optimizer success: {inverse.success}")
    print(f"  certified: {inverse.certified}")
    print(f"  evaluations: {inverse.evaluations}")
    print(f"  max period residual: {inverse.period_max_residual:.6e}")
    print(f"  word stability: {inverse.word_stability:.6e}")
    print(f"  symmetry error: {inverse.symmetry_error:.6e}")
    print(f"  min eig Im(Omega): {inverse.minimum_imaginary_eigenvalue:.6e}")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(_result_payload(q_truth, target, inverse), indent=2) + "\n")
        print(f"  wrote {args.output_json}")

    if not inverse.certified:
        raise SystemExit(1)


if __name__ == "__main__":
    run()
