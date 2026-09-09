#!/usr/bin/env python3
"""Audit the absolute free-boson normalization in a separating long tube.

The plumbing scalar and the canonical scalar are compared before any fitted
normalization is applied.  The canonical reference is the normalized
Arakelov determinant, whose separating degeneration is fixed by two standard
genus-one scalar partition functions.  This makes the target-space constant
mode convention and the handle-momentum Gaussian separately visible.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from free_boson_plumbing import (
        dedekind_eta_abs_from_q,
        glasses_free_boson_product,
        igusa_chi10_log_abs_genus2,
        noncompact_scalar_loop_momentum_factor,
        tau_imag_from_q,
    )
    from liouville_genus2 import format_complex, parse_complex
    from plumbing_algorithms import plumbing_genus2_period_matrix
except ImportError:  # pragma: no cover - supports package-style execution
    from plumbing.free_boson_plumbing import (
        dedekind_eta_abs_from_q,
        glasses_free_boson_product,
        igusa_chi10_log_abs_genus2,
        noncompact_scalar_loop_momentum_factor,
        tau_imag_from_q,
    )
    from plumbing.liouville_genus2 import format_complex, parse_complex
    from plumbing.plumbing_algorithms import plumbing_genus2_period_matrix


DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent / "results" / "free_boson_long_tube_normalization"
)


def _complex_record(value: complex) -> dict[str, float]:
    value = complex(value)
    return {"real": float(value.real), "imag": float(value.imag)}


def genus1_canonical_scalar_from_q(q_value: complex, *, eta_max_mode: int = 300) -> float:
    r"""Return ``1/(sqrt(Im tau) |eta(tau)|^2)`` for ``q=e^(2 pi i tau)``."""

    eta_abs = dedekind_eta_abs_from_q(q_value, max_mode=eta_max_mode)
    return float(tau_imag_from_q(q_value) ** -0.5 * eta_abs**-2)


def evaluate_long_tube_point(
    q_left: complex,
    q_right: complex,
    q_bridge: complex,
    *,
    period_algorithm: str = "collocation",
    collocation_basis_order: int = 60,
    collocation_samples: int = 256,
    period_word_length: int = 10,
    period_b_order: int = 900,
    max_word_length: int = 10,
    max_mode: int = 100,
    product_tolerance: float = 1.0e-14,
    theta_nmax: int | None = None,
    theta_tolerance: float = 1.0e-14,
) -> dict[str, object]:
    r"""Evaluate one absolute-normalization point in the glasses channel.

    The reduced partition functions divide the connected target zero mode by
    ``V_X/(2 pi)``.  The factor ``det(Im Omega)^(-1/2)`` is retained because it
    is the Gaussian integral over the two handle momenta.
    """

    q_left = complex(q_left)
    q_right = complex(q_right)
    q_bridge = complex(q_bridge)
    if min(abs(q_left), abs(q_right), abs(q_bridge)) <= 0.0:
        raise ValueError("all plumbing parameters must be nonzero")
    if max(abs(q_left), abs(q_right), abs(q_bridge)) >= 1.0:
        raise ValueError("all plumbing parameters must satisfy |q|<1")

    omega, period_method = plumbing_genus2_period_matrix(
        "glasses",
        q_left,
        q_right,
        q_bridge,
        algorithm=period_algorithm,
        schottky_word_len=period_word_length,
        schottky_b_order=period_b_order,
        collocation_basis_order=collocation_basis_order,
        collocation_samples=collocation_samples,
    )
    omega = np.asarray(omega, dtype=np.complex128)
    det_im_omega = float(np.linalg.det(omega.imag))

    oscillator = glasses_free_boson_product(
        q_left,
        q_right,
        q_bridge,
        max_word_length=max_word_length,
        max_mode=max_mode,
        tolerance=product_tolerance,
    )
    loop_momentum_gaussian = noncompact_scalar_loop_momentum_factor(omega)
    constant_target_mode_per_normalized_volume = 1.0
    plumbing_scalar = (
        constant_target_mode_per_normalized_volume
        * loop_momentum_gaussian
        * oscillator.nonchiral_value
    )

    eta_left = dedekind_eta_abs_from_q(q_left, max_mode=max_mode)
    eta_right = dedekind_eta_abs_from_q(q_right, max_mode=max_mode)
    genus1_left = genus1_canonical_scalar_from_q(q_left, eta_max_mode=max_mode)
    genus1_right = genus1_canonical_scalar_from_q(q_right, eta_max_mode=max_mode)

    # Fay's separating coordinate obeys Omega_12 = 2 pi i t + O(t^3).
    fay_t_abs = abs(omega[0, 1] / (2.0j * math.pi))
    arakelov_t_norm = 4.0 * math.pi**2 * fay_t_abs * (eta_left * eta_right) ** 2
    arakelov_factorized = (
        arakelov_t_norm ** (-1.0 / 6.0) * genus1_left * genus1_right
    )

    # Vandermeulen's normalized genus-two scalar uses chi10=prod_even theta^2.
    # Phi -> |eta_left eta_right|^2 in the separating limit.  Keeping exact
    # Omega and chi10 while using this leading Phi makes the residual O(t^2).
    log_abs_chi10 = igusa_chi10_log_abs_genus2(
        omega,
        nmax=theta_nmax,
        tol=theta_tolerance,
        normalization="product",
    )
    phi_leading = (eta_left * eta_right) ** 2
    log_arakelov_modular_leading = (
        math.log(2.0)
        - log_abs_chi10 / 12.0
        - math.log(phi_leading) / 6.0
        - 0.5 * math.log(det_im_omega)
    )
    arakelov_modular_leading = math.exp(log_arakelov_modular_leading)

    # Raw torus sewing omits the Casimir term:
    # Z_1^pl=|q|^(1/12) Z_1^canonical.  Together with the direct Arakelov
    # tube anomaly this predicts the complete plumbing/canonical ratio.
    predicted_plumbing_over_arakelov = (
        abs(q_left * q_right) ** (1.0 / 12.0) * arakelov_t_norm ** (1.0 / 6.0)
    )
    measured_plumbing_over_arakelov = plumbing_scalar / arakelov_modular_leading
    normalization_constant = (
        measured_plumbing_over_arakelov / predicted_plumbing_over_arakelov
    )

    measured_factorized_ratio = plumbing_scalar / arakelov_factorized
    factorized_normalization_constant = (
        measured_factorized_ratio / predicted_plumbing_over_arakelov
    )
    oscillator_only_normalization_constant = (
        (oscillator.nonchiral_value / arakelov_modular_leading)
        / predicted_plumbing_over_arakelov
    )

    chi10_asymptotic_log_abs = (
        12.0 * math.log(2.0)
        + 2.0 * math.log(abs(q_bridge))
        + 24.0 * math.log(eta_left * eta_right)
    )
    q_bridge_from_period_abs = 4.0 * math.pi**2 * fay_t_abs

    return {
        "q_left": _complex_record(q_left),
        "q_right": _complex_record(q_right),
        "q_bridge": _complex_record(q_bridge),
        "q_bridge_abs": float(abs(q_bridge)),
        "period_method": period_method,
        "omega": [
            [_complex_record(omega[i, j]) for j in range(2)] for i in range(2)
        ],
        "det_im_omega": det_im_omega,
        "fay_t_abs": float(fay_t_abs),
        "q_bridge_from_period_abs": float(q_bridge_from_period_abs),
        "q_bridge_period_ratio": float(q_bridge_from_period_abs / abs(q_bridge)),
        "eta_left_abs": float(eta_left),
        "eta_right_abs": float(eta_right),
        "genus1_left_canonical": float(genus1_left),
        "genus1_right_canonical": float(genus1_right),
        "constant_target_mode_per_V_over_2pi": constant_target_mode_per_normalized_volume,
        "loop_momentum_gaussian": float(loop_momentum_gaussian),
        "plumbing_oscillator": float(oscillator.nonchiral_value),
        "plumbing_scalar_reduced": float(plumbing_scalar),
        "primitive_count": int(oscillator.primitive_count),
        "omitted_chiral_tail_estimate": float(oscillator.omitted_chiral_tail_estimate),
        "arakelov_t_norm": float(arakelov_t_norm),
        "arakelov_scalar_factorized": float(arakelov_factorized),
        "arakelov_phi_leading": float(phi_leading),
        "arakelov_scalar_modular_phi_leading": float(arakelov_modular_leading),
        "chi10_log_abs_exact": float(log_abs_chi10),
        "chi10_exact_over_asymptotic_abs": float(
            math.exp(log_abs_chi10 - chi10_asymptotic_log_abs)
        ),
        "predicted_plumbing_over_arakelov": float(
            predicted_plumbing_over_arakelov
        ),
        "measured_plumbing_over_arakelov": float(measured_plumbing_over_arakelov),
        "normalization_constant": float(normalization_constant),
        "normalization_constant_minus_one": float(normalization_constant - 1.0),
        "factorized_normalization_constant": float(factorized_normalization_constant),
        "oscillator_only_normalization_constant": float(
            oscillator_only_normalization_constant
        ),
        "mixed_plumbing_ordinary_volume_vs_canonical_V_over_2pi": float(
            normalization_constant / (2.0 * math.pi)
        ),
        "mixed_plumbing_V_over_2pi_vs_canonical_ordinary_volume": float(
            normalization_constant * 2.0 * math.pi
        ),
    }


def _fit_boundary_constant(rows: list[dict[str, object]]) -> tuple[float, float]:
    x = np.asarray([float(row["q_bridge_abs"]) ** 2 for row in rows], dtype=np.float64)
    y = np.asarray([float(row["normalization_constant"]) for row in rows], dtype=np.float64)
    if len(rows) < 2:
        return float(y[0]), 0.0
    slope, intercept = np.polyfit(x, y, 1)
    return float(intercept), float(slope)


def _write_results(output_dir: Path, payload: dict[str, object]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "normalization_audit.json"
    csv_path = output_dir / "normalization_scan.csv"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    rows = payload["points"]
    assert isinstance(rows, list)
    fields = (
        "q_bridge_abs",
        "period_method",
        "q_bridge_period_ratio",
        "chi10_exact_over_asymptotic_abs",
        "loop_momentum_gaussian",
        "plumbing_oscillator",
        "plumbing_scalar_reduced",
        "arakelov_scalar_modular_phi_leading",
        "predicted_plumbing_over_arakelov",
        "measured_plumbing_over_arakelov",
        "normalization_constant",
        "normalization_constant_minus_one",
        "factorized_normalization_constant",
        "oscillator_only_normalization_constant",
        "mixed_plumbing_ordinary_volume_vs_canonical_V_over_2pi",
        "mixed_plumbing_V_over_2pi_vs_canonical_ordinary_volume",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})
    return json_path, csv_path


def run(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the absolute free-boson plumbing normalization in a separating long tube."
        )
    )
    parser.add_argument("--q-left", type=parse_complex, default=0.08 + 0.0j)
    parser.add_argument("--q-right", type=parse_complex, default=0.11 + 0.0j)
    parser.add_argument(
        "--q-bridge",
        type=parse_complex,
        nargs="+",
        default=(1.0e-2 + 0.0j, 1.0e-3 + 0.0j, 1.0e-4 + 0.0j, 1.0e-5 + 0.0j),
    )
    parser.add_argument(
        "--period-algorithm",
        choices=("collocation", "schottky", "auto"),
        default="collocation",
    )
    parser.add_argument("--collocation-basis-order", type=int, default=60)
    parser.add_argument("--collocation-samples", type=int, default=256)
    parser.add_argument("--period-word-length", type=int, default=10)
    parser.add_argument("--period-b-order", type=int, default=900)
    parser.add_argument("--max-word-length", type=int, default=10)
    parser.add_argument("--max-mode", type=int, default=100)
    parser.add_argument("--theta-nmax", type=int)
    parser.add_argument("--theta-tolerance", type=float, default=1.0e-14)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    q_left = complex(args.q_left)
    q_right = complex(args.q_right)
    rows = [
        evaluate_long_tube_point(
            q_left,
            q_right,
            complex(q_bridge),
            period_algorithm=args.period_algorithm,
            collocation_basis_order=args.collocation_basis_order,
            collocation_samples=args.collocation_samples,
            period_word_length=args.period_word_length,
            period_b_order=args.period_b_order,
            max_word_length=args.max_word_length,
            max_mode=args.max_mode,
            theta_nmax=args.theta_nmax,
            theta_tolerance=args.theta_tolerance,
        )
        for q_bridge in args.q_bridge
    ]
    boundary_constant, boundary_slope = _fit_boundary_constant(rows)

    payload: dict[str, object] = {
        "scope": "Absolute c=1 free-boson plumbing/canonical normalization at a separating node.",
        "canonical_metric": "Arakelov",
        "target_zero_mode_convention": (
            "The full connected factor is V_X/(2*pi); reported partitions divide by this factor."
        ),
        "handle_momentum_measure": (
            "<p|p'>=delta(p-p'), completeness dp, giving det(Im Omega)^(-1/2)."
        ),
        "weyl_prediction": (
            "Z_pl/Z_Arakelov -> |q_left*q_right|^(1/12) * ||t||^(1/6), "
            "||t||=4*pi^2*|t|*|eta_left*eta_right|^2."
        ),
        "canonical_formula": (
            "Z_2^Ar=2*|prod_even theta_delta^2|^(-1/12)*Phi^(-1/6)/sqrt(det Im Omega)."
        ),
        "phi_approximation": "Phi=|eta_left*eta_right|^2+O(t^2).",
        "q_left": _complex_record(q_left),
        "q_right": _complex_record(q_right),
        "boundary_extrapolated_normalization_constant": boundary_constant,
        "boundary_extrapolated_minus_one": boundary_constant - 1.0,
        "linear_slope_in_abs_q_bridge_squared": boundary_slope,
        "points": rows,
        "references": {
            "canonical_scalar": "https://arxiv.org/abs/1902.02420",
            "plumbing_free_boson": "https://arxiv.org/abs/0712.0628",
            "determinant_sewing": "https://doi.org/10.1063/1.529239",
        },
    }

    print("Free-boson absolute long-tube normalization audit")
    print(f"  q_left  = {format_complex(q_left)}")
    print(f"  q_right = {format_complex(q_right)}")
    print("  target zero mode = V_X/(2*pi), divided out on both sides")
    if args.period_algorithm == "collocation":
        print("  period map = normalized holomorphic one-forms")
    else:
        print(f"  period map = {args.period_algorithm}")
    print()
    print(
        "|q_bridge|   4*pi^2|t|/|q_B|   chi10 exact/asym   "
        "extracted constant   constant-1"
    )
    for row in rows:
        print(
            f"{float(row['q_bridge_abs']):11.3e}   "
            f"{float(row['q_bridge_period_ratio']):17.12f}   "
            f"{float(row['chi10_exact_over_asymptotic_abs']):17.12f}   "
            f"{float(row['normalization_constant']):18.12f}   "
            f"{float(row['normalization_constant_minus_one']):+.3e}"
        )
    print()
    print(f"  boundary extrapolation = {boundary_constant:.16e}")
    print(f"  extrapolation minus one = {boundary_constant - 1.0:+.3e}")
    print("  ordinary-volume mismatch controls differ by exactly 2*pi")

    if not args.no_write:
        json_path, csv_path = _write_results(args.output_dir, payload)
        print(f"  JSON = {json_path}")
        print(f"  CSV  = {csv_path}")


if __name__ == "__main__":
    run()
