#!/usr/bin/env python3
"""Formula-blind equal-energy scan of the genuine Spin(23) SSVV amplitude.

This driver deliberately imports only the numerical amplitude engine.  It does
not import or evaluate any proposed closed form.  The comparison with a
candidate formula is a separate post-processing step performed only after this
JSON output has been completed and frozen.

The leg convention is

    S(omega0) -> S(omega1) V(omega2) V(omega3),
    omega1 = omega2 = omega3 = omega,  omega0 = 3 omega,
    Re(omega) = 1/9.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np


CODE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = CODE_DIR.parent
DEFAULT_OUTPUT = (
    WORKSPACE_DIR
    / "Data Set"
    / "high_accuracy"
    / "spin23_ssvv_equal_c_recursion_elliptic_threshold.json"
)

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from spin23_singlet_amplitudes import evaluate_singlet_amplitudes  # noqa: E402
import ns_elliptic_conversion as elliptic_conversion


PROFILES: dict[str, dict[str, Any]] = {
    "precision_infinite_q7": {
        "q_order": 7,
        "lower_q_order": 6,
        "p_nodes": 96,
        "p_max": 0.0,
        "p_cut": 0.0,
        "momentum_scheme": "infinite_gauss",
        "infinite_gauss_scale": 0.5,
        "epsilon0": 0.30,
        "epsilon1": 0.24,
        "theta_orders": (48, 48, 192),
        "radial_order": 72,
        "disk_total_order": 32,
        "crossed_disk_total_order": 32,
        "block_backend": "c_recursion",
        "series_parameter": "sewing",
        "recursion_cancellation_limit": 1.0e6,
        "recursion_digits": 70,
        "recursion_reference_p_max": 0.18,
    },
    "precision_infinite_q8": {
        "q_order": 8,
        "lower_q_order": 7,
        "p_nodes": 96,
        "p_max": 0.0,
        "p_cut": 0.0,
        "momentum_scheme": "infinite_gauss",
        "infinite_gauss_scale": 0.5,
        "epsilon0": 0.30,
        "epsilon1": 0.24,
        "theta_orders": (48, 48, 192),
        "radial_order": 72,
        "disk_total_order": 32,
        "crossed_disk_total_order": 32,
        "block_backend": "c_recursion",
        "series_parameter": "sewing",
        "recursion_cancellation_limit": 1.0e6,
        "recursion_digits": 70,
        "recursion_reference_p_max": 0.18,
    },
    "precision_infinite128_q7": {
        "q_order": 7,
        "lower_q_order": 6,
        "p_nodes": 128,
        "p_max": 0.0,
        "p_cut": 0.0,
        "momentum_scheme": "infinite_gauss",
        "infinite_gauss_scale": 0.5,
        "epsilon0": 0.30,
        "epsilon1": 0.24,
        "theta_orders": (48, 48, 192),
        "radial_order": 72,
        "disk_total_order": 32,
        "crossed_disk_total_order": 32,
        "block_backend": "c_recursion",
        "series_parameter": "sewing",
        "recursion_cancellation_limit": 1.0e6,
        "recursion_digits": 70,
        "recursion_reference_p_max": 0.18,
    },
    "precision_infinite_dense_q7": {
        "q_order": 7,
        "lower_q_order": 6,
        "p_nodes": 96,
        "p_max": 0.0,
        "p_cut": 0.0,
        "momentum_scheme": "infinite_gauss",
        "infinite_gauss_scale": 0.5,
        "epsilon0": 0.30,
        "epsilon1": 0.24,
        "theta_orders": (64, 64, 256),
        "radial_order": 96,
        "disk_total_order": 36,
        "crossed_disk_total_order": 36,
        "block_backend": "c_recursion",
        "series_parameter": "sewing",
        "recursion_cancellation_limit": 1.0e6,
        "recursion_digits": 70,
        "recursion_reference_p_max": 0.18,
    },
    "production_infinite_q7": {
        "q_order": 7,
        "lower_q_order": 6,
        "p_nodes": 96,
        "p_max": 0.0,
        "p_cut": 0.0,
        "momentum_scheme": "infinite_gauss",
        "infinite_gauss_scale": 0.5,
        "epsilon0": 0.08,
        "epsilon1": 0.06,
        "theta_orders": (15, 15, 60),
        "radial_order": 24,
        "disk_total_order": 20,
        "crossed_disk_total_order": 20,
        "block_backend": "c_recursion",
        "series_parameter": "sewing",
        "recursion_cancellation_limit": 1.0e6,
        "recursion_digits": 70,
        "recursion_reference_p_max": 0.18,
    },
    "infinite128_q7_control": {
        "q_order": 7,
        "lower_q_order": 6,
        "p_nodes": 128,
        "p_max": 0.0,
        "p_cut": 0.0,
        "momentum_scheme": "infinite_gauss",
        "infinite_gauss_scale": 1.0,
        "epsilon0": 0.08,
        "epsilon1": 0.06,
        "theta_orders": (15, 15, 60),
        "radial_order": 24,
        "disk_total_order": 20,
        "crossed_disk_total_order": 20,
        "block_backend": "c_recursion",
        "series_parameter": "sewing",
        "recursion_cancellation_limit": 1.0e6,
        "recursion_digits": 70,
        "recursion_reference_p_max": 0.18,
    },
    "infinite96_scale05_q7_control": {
        "q_order": 7,
        "lower_q_order": 6,
        "p_nodes": 96,
        "p_max": 0.0,
        "p_cut": 0.0,
        "momentum_scheme": "infinite_gauss",
        "infinite_gauss_scale": 0.5,
        "epsilon0": 0.08,
        "epsilon1": 0.06,
        "theta_orders": (15, 15, 60),
        "radial_order": 24,
        "disk_total_order": 20,
        "crossed_disk_total_order": 20,
        "block_backend": "c_recursion",
        "series_parameter": "sewing",
        "recursion_cancellation_limit": 1.0e6,
        "recursion_digits": 70,
        "recursion_reference_p_max": 0.18,
    },
    "infinite128_scale05_q7_control": {
        "q_order": 7,
        "lower_q_order": 6,
        "p_nodes": 128,
        "p_max": 0.0,
        "p_cut": 0.0,
        "momentum_scheme": "infinite_gauss",
        "infinite_gauss_scale": 0.5,
        "epsilon0": 0.08,
        "epsilon1": 0.06,
        "theta_orders": (15, 15, 60),
        "radial_order": 24,
        "disk_total_order": 20,
        "crossed_disk_total_order": 20,
        "block_backend": "c_recursion",
        "series_parameter": "sewing",
        "recursion_cancellation_limit": 1.0e6,
        "recursion_digits": 70,
        "recursion_reference_p_max": 0.18,
    },
    "infinite_q8_control": {
        "q_order": 8,
        "lower_q_order": 7,
        "p_nodes": 96,
        "p_max": 0.0,
        "p_cut": 0.0,
        "momentum_scheme": "infinite_gauss",
        "infinite_gauss_scale": 0.5,
        "epsilon0": 0.08,
        "epsilon1": 0.06,
        "theta_orders": (15, 15, 60),
        "radial_order": 24,
        "disk_total_order": 20,
        "crossed_disk_total_order": 20,
        "block_backend": "c_recursion",
        "series_parameter": "sewing",
        "recursion_cancellation_limit": 1.0e6,
        "recursion_digits": 70,
        "recursion_reference_p_max": 0.18,
    },
    "production_q7": {
        "q_order": 7,
        "lower_q_order": 6,
        "p_nodes": "segmented",
        "p_max": 2.5,
        "p_cut": 0.03,
        "epsilon0": 0.08,
        "epsilon1": 0.06,
        "theta_orders": (15, 15, 60),
        "radial_order": 24,
        "disk_total_order": 20,
        "crossed_disk_total_order": 20,
        "block_backend": "c_recursion",
        "series_parameter": "sewing",
        "recursion_cancellation_limit": 1.0e6,
        "recursion_digits": 70,
        "recursion_reference_p_max": 0.18,
    },
    "q8_control": {
        "q_order": 8,
        "lower_q_order": 7,
        "p_nodes": "segmented",
        "p_max": 2.5,
        "p_cut": 0.03,
        "epsilon0": 0.08,
        "epsilon1": 0.06,
        "theta_orders": (15, 15, 60),
        "radial_order": 24,
        "disk_total_order": 20,
        "crossed_disk_total_order": 20,
        "block_backend": "c_recursion",
        "series_parameter": "sewing",
        "recursion_cancellation_limit": 1.0e6,
        "recursion_digits": 70,
        "recursion_reference_p_max": 0.18,
    },
    "momentum_dense_q7": {
        "q_order": 7,
        "lower_q_order": 6,
        "p_nodes": [32, 36, 48, 36, 32, 28, 28],
        "p_max": 2.5,
        "p_cut": 0.03,
        "epsilon0": 0.08,
        "epsilon1": 0.06,
        "theta_orders": (15, 15, 60),
        "radial_order": 24,
        "disk_total_order": 20,
        "crossed_disk_total_order": 20,
        "block_backend": "c_recursion",
        "series_parameter": "sewing",
        "recursion_cancellation_limit": 1.0e6,
        "recursion_digits": 70,
        "recursion_reference_p_max": 0.18,
    },
    "moduli_dense_q7": {
        "q_order": 7,
        "lower_q_order": 6,
        "p_nodes": "segmented",
        "p_max": 2.5,
        "p_cut": 0.03,
        "epsilon0": 0.08,
        "epsilon1": 0.06,
        "theta_orders": (30, 30, 120),
        "radial_order": 48,
        "disk_total_order": 20,
        "crossed_disk_total_order": 20,
        "block_backend": "c_recursion",
        "series_parameter": "sewing",
        "recursion_cancellation_limit": 1.0e6,
        "recursion_digits": 70,
        "recursion_reference_p_max": 0.18,
    },
    "pmax4_q7": {
        "q_order": 7,
        "lower_q_order": 6,
        "p_nodes": "segmented",
        "p_max": 4.0,
        "p_cut": 0.03,
        "epsilon0": 0.08,
        "epsilon1": 0.06,
        "theta_orders": (15, 15, 60),
        "radial_order": 24,
        "disk_total_order": 20,
        "crossed_disk_total_order": 20,
        "block_backend": "c_recursion",
        "series_parameter": "sewing",
        "recursion_cancellation_limit": 1.0e6,
        "recursion_digits": 70,
        "recursion_reference_p_max": 0.18,
    },
}

# Preserve explicit historical quadrature controls for reproducibility.
for _profile in PROFILES.values():
    _profile.setdefault("momentum_scheme", "cutoff")
PROFILES["threshold_sewing_q7"] = {
    **PROFILES["production_infinite_q7"],
    "momentum_scheme": "threshold_weighted",
    "p_nodes": [16, 96, 32],
    "momentum_threshold_options": {"endpoint": 0.18, "tail": 3.2, "beta": 2.0, "a": 1.0, "s": 0.0},
}
PROFILES["threshold_elliptic_q7"] = {
    **PROFILES["threshold_sewing_q7"], "series_parameter": "elliptic_nome",
}

DEFAULT_Y = tuple(float(value) for value in np.linspace(1.0 / 6.0, 7.0 / 30.0, 11))


def complex_pair(value: complex) -> list[float]:
    value = complex(value)
    return [float(value.real), float(value.imag)]


def relative_change(current: complex, previous: complex) -> float:
    return float(abs(current - previous) / max(abs(current), np.finfo(float).tiny))


def cancellation_ratio(pieces: Mapping[str, complex]) -> float:
    values = [complex(value) for value in pieces.values()]
    total = math.fsum(value.real for value in values) + 1j * math.fsum(
        value.imag for value in values
    )
    return float(sum(abs(value) for value in values) / max(abs(total), np.finfo(float).tiny))


def evaluate_y(y: float, settings: Mapping[str, Any]) -> dict[str, Any]:
    omega = 1.0 / 9.0 + 1.0j * float(y)
    started = time.perf_counter()
    evaluation = evaluate_singlet_amplitudes(
        (omega, omega, omega, 3.0 * omega), **dict(settings)
    )
    elapsed = time.perf_counter() - started
    current = complex(evaluation.values.ssvv_raw)
    lower = (
        None
        if evaluation.lower_order_values is None
        else complex(evaluation.lower_order_values.ssvv_raw)
    )
    pieces = {
        name: complex(value)
        for name, value in evaluation.values.pieces["ssvv"].items()
    }
    return {
        "im_omega": float(y),
        "omega": complex_pair(omega),
        "omega0": complex_pair(3.0 * omega),
        "ssvv_raw": complex_pair(current),
        "ssvv_unit_descendants": complex_pair(
            evaluation.values.ssvv_unit_descendants
        ),
        "lower_q_ssvv_raw": None if lower is None else complex_pair(lower),
        "adjacent_q_absolute_change": (
            None if lower is None else float(abs(current - lower))
        ),
        "adjacent_q_relative_change": (
            None if lower is None else relative_change(current, lower)
        ),
        "piece_cancellation_ratio": cancellation_ratio(pieces),
        "pieces": {name: complex_pair(value) for name, value in pieces.items()},
        "momentum_nodes": evaluation.momentum_nodes,
        "momentum_scheme": evaluation.momentum_quadrature["scheme"],
        "momentum_quadrature": evaluation.momentum_quadrature,
        "block_backend": evaluation.block_backend,
        **elliptic_conversion.representation_metadata(evaluation.series_parameter, evaluation.q_order),
        "maximum_recursion_cancellation": evaluation.maximum_recursion_cancellation,
        "high_precision_recursion_node_count": evaluation.high_precision_recursion_node_count,
        "maximum_gram_condition": evaluation.maximum_gram_condition,
        "maximum_equilibrated_gram_condition": (
            evaluation.maximum_equilibrated_gram_condition
        ),
        "high_precision_gram_solve_count": evaluation.high_precision_gram_solve_count,
        "template_seconds": evaluation.template_seconds,
        "build_seconds": evaluation.build_seconds,
        "integration_seconds": evaluation.integration_seconds,
        "runtime_seconds": elapsed,
    }


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", nargs="+", default=["threshold_elliptic_q7"])
    parser.add_argument("--y", nargs="+", type=float, default=list(DEFAULT_Y))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    unknown = [name for name in args.profiles if name not in PROFILES]
    if unknown:
        parser.error(f"unknown profiles: {', '.join(unknown)}")
    if any(not (0.0 < y < 0.25) for y in args.y):
        parser.error("this direct-contour scan requires 0 < Im(omega) < 1/4")

    if args.output.exists() and not args.force:
        report = json.loads(args.output.read_text(encoding="utf-8"))
    else:
        report = {"profiles": {}}
    report.update(
        {
            "status": "running",
            "calculation_stage": "formula_blind_numerical_evaluation",
            "proposal_imported_or_evaluated": False,
            "process": "S(omega0) -> S(omega1) V(omega2) V(omega3)",
            "normalization": "raw reduced coefficient of delta_ab",
            "convention": (
                "omega1=omega2=omega3=omega=1/9+i*y; omega0=3*omega"
            ),
            "direct_contour_domain": "0 < Im(omega) < 1/4",
            "structure_constant_backend": (
                "exp(log C), with Upsilon_1 arguments reduced by "
                "Upsilon_1(x+1)=gamma(x) Upsilon_1(x)"
            ),
            "infinite_momentum_map": (
                "profile-recorded rule: endpoint/bulk/infinite-tail weighted quadrature "
                "by default; historical logarithmic maps retained as independent controls"
            ),
        }
    )

    for profile_name in args.profiles:
        settings = PROFILES[profile_name]
        existing = dict(report["profiles"].get(profile_name, {}))
        if existing.get("rows") and json.dumps(existing.get("settings"), sort_keys=True) != json.dumps(settings, sort_keys=True):
            parser.error("numerical settings changed; choose a fresh output file")
        if any(row.get("momentum_scheme") != settings["momentum_scheme"] for row in existing.get("rows", [])):
            parser.error("output uses another or unrecorded momentum rule; choose a fresh output file")
        if any(row.get("block_backend") != settings["block_backend"] or row.get("series_parameter") != settings["series_parameter"] for row in existing.get("rows", [])):
            parser.error("output contains another block backend or series parameter; choose a fresh output file")
        if settings["series_parameter"] == "elliptic_nome" and any(row.get("numerical_algorithm") != elliptic_conversion.ALGORITHM_VERSION for row in existing.get("rows", [])):
            parser.error("output uses an older elliptic/OPE implementation; choose a fresh output file")
        rows_by_y = {
            float(row["im_omega"]): row for row in existing.get("rows", [])
        }
        print(f"running {profile_name}", flush=True)
        for y in args.y:
            row = evaluate_y(float(y), settings)
            rows_by_y[float(y)] = row
            rows = [rows_by_y[key] for key in sorted(rows_by_y)]
            existing.update({"settings": settings, "rows": rows})
            report["profiles"][profile_name] = existing
            write_report(args.output, report)
            print(
                f"  y={y:.9f} M={complex(*row['ssvv_raw'])} "
                f"dq={row['adjacent_q_relative_change']:.3e} "
                f"cancel={row['piece_cancellation_ratio']:.3e} "
                f"seconds={row['runtime_seconds']:.1f}",
                flush=True,
            )

    report["status"] = "complete"
    write_report(args.output, report)
    print(args.output)


if __name__ == "__main__":
    main()
