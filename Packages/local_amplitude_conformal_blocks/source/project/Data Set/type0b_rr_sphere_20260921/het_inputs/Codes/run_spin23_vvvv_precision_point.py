#!/usr/bin/env python3
"""Run resumable high-accuracy checks of the four-vector amplitude ansatz.

The default kinematics are the asymmetric validation point

    omega0 = 1/3 + 0.6 i,
    (omega1,omega2,omega3) = (0.2,0.3,0.5) omega0.

Each completed profile is written immediately, so a long precision run can be
resumed without recomputing finished profiles.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


CODE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = CODE_DIR.parent
FIT_DIR = CODE_DIR / "heterotic_so23_1to3_vvvv_fit_bundle"
SETTINGS_PATH = CODE_DIR / "spin23_vvvv_stable_settings.json"
DEFAULT_OUTPUT = (
    WORKSPACE_DIR
    / "Data Set"
    / "high_accuracy"
    / "spin23_vvvv_asymmetric_precision.json"
)

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if str(FIT_DIR) not in sys.path:
    sys.path.insert(0, str(FIT_DIR))

from heterotic_so23_1to3_fast import pair_channel_ansatz  # noqa: E402
from spin23_vvvv_scan_evaluator import evaluate_point  # noqa: E402


CUSTOM_PROFILES: dict[str, dict[str, Any]] = {
    "precision_q8_small_patch": {
        "mp_dps": 90,
        "q_order": 8,
        "p_max": 6.0,
        "p_nodes": [24, 27, 36, 27, 24, 21, 21],
        "p_cut": 0.03,
        "z_radial_nodes": 48,
        "z_angular_nodes": 120,
        "ope_radius": 0.06,
        "crossed_ope_radius": 0.05,
        "disk_total_order": 24,
        "lens_radial_nodes": 40,
        "lens_angular_nodes": 128,
        "lens_power": 3.0,
        "reference_p_max": 0.18,
        "cancellation_limit": 1.0e7,
        "estimate_q_error": True,
    },
    "precision_q8_large_patch": {
        "mp_dps": 90,
        "q_order": 8,
        "p_max": 6.0,
        "p_nodes": [24, 27, 36, 27, 24, 21, 21],
        "p_cut": 0.03,
        "z_radial_nodes": 48,
        "z_angular_nodes": 120,
        "ope_radius": 0.10,
        "crossed_ope_radius": 0.08,
        "disk_total_order": 24,
        "lens_radial_nodes": 40,
        "lens_angular_nodes": 128,
        "lens_power": 3.0,
        "reference_p_max": 0.18,
        "cancellation_limit": 1.0e7,
        "estimate_q_error": True,
    },
    "precision_q9": {
        "mp_dps": 90,
        "q_order": 9,
        "p_max": 6.0,
        "p_nodes": [24, 27, 36, 27, 24, 21, 21],
        "p_cut": 0.03,
        "z_radial_nodes": 48,
        "z_angular_nodes": 120,
        # Match the validated large-patch q=8 profile so that the q=9/q=8
        # difference isolates recursion order rather than quadrature settings.
        "ope_radius": 0.10,
        "crossed_ope_radius": 0.08,
        "disk_total_order": 24,
        "lens_radial_nodes": 40,
        "lens_angular_nodes": 128,
        "lens_power": 3.0,
        "reference_p_max": 0.18,
        "cancellation_limit": 1.0e7,
        "estimate_q_error": True,
    },
    "precision_q8_refined_quadrature": {
        "mp_dps": 90,
        "q_order": 8,
        "p_max": 6.0,
        "p_nodes": [32, 36, 48, 36, 32, 28, 28],
        "p_cut": 0.03,
        "z_radial_nodes": 64,
        "z_angular_nodes": 160,
        "ope_radius": 0.10,
        "crossed_ope_radius": 0.08,
        "disk_total_order": 32,
        "lens_radial_nodes": 52,
        "lens_angular_nodes": 160,
        "lens_power": 3.0,
        "reference_p_max": 0.18,
        "cancellation_limit": 1.0e7,
        # q=9 already supplies the recursion-order control.  Avoid duplicating
        # an expensive q=7 evaluation in this pure quadrature check.
        "estimate_q_error": False,
    },
    "precision_q8_worldsheet_refined": {
        "mp_dps": 90,
        "q_order": 8,
        # Keep the validated baseline momentum grid while refining only the
        # worldsheet quadratures.  Comparing this profile with the fully
        # refined one isolates momentum-quadrature sensitivity.
        "p_max": 6.0,
        "p_nodes": [24, 27, 36, 27, 24, 21, 21],
        "p_cut": 0.03,
        "z_radial_nodes": 64,
        "z_angular_nodes": 160,
        "ope_radius": 0.10,
        "crossed_ope_radius": 0.08,
        "disk_total_order": 32,
        "lens_radial_nodes": 52,
        "lens_angular_nodes": 160,
        "lens_power": 3.0,
        "reference_p_max": 0.18,
        "cancellation_limit": 1.0e7,
        "estimate_q_error": False,
    },
    "precision_q8_worldsheet_ultra": {
        "mp_dps": 90,
        "q_order": 8,
        "p_max": 6.0,
        "p_nodes": [24, 27, 36, 27, 24, 21, 21],
        "p_cut": 0.03,
        "z_radial_nodes": 80,
        "z_angular_nodes": 200,
        "ope_radius": 0.10,
        "crossed_ope_radius": 0.08,
        "disk_total_order": 40,
        "lens_radial_nodes": 64,
        "lens_angular_nodes": 200,
        "lens_power": 3.0,
        "reference_p_max": 0.18,
        "cancellation_limit": 1.0e7,
        "estimate_q_error": False,
    },
    "precision_q9_worldsheet_ultra": {
        "mp_dps": 90,
        "q_order": 9,
        "p_max": 6.0,
        "p_nodes": [24, 27, 36, 27, 24, 21, 21],
        "p_cut": 0.03,
        "z_radial_nodes": 80,
        "z_angular_nodes": 200,
        "ope_radius": 0.10,
        "crossed_ope_radius": 0.08,
        "disk_total_order": 40,
        "lens_radial_nodes": 64,
        "lens_angular_nodes": 200,
        "lens_power": 3.0,
        "reference_p_max": 0.18,
        "cancellation_limit": 1.0e7,
        "estimate_q_error": False,
    },
}


def complex_pair(value: complex) -> list[float]:
    value = complex(value)
    return [float(value.real), float(value.imag)]


def jsonable(value: Any) -> Any:
    if isinstance(value, complex):
        return complex_pair(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(report), indent=2) + "\n", encoding="utf-8")


def profile_values(report: dict[str, Any], name: str) -> np.ndarray:
    """Return one saved profile's three complex amplitudes."""
    return np.asarray(
        [complex(*pair) for pair in report["profiles"][name]["numerical"]],
        dtype=complex,
    )


def relative_profile_shift(
    report: dict[str, Any], first: str, second: str
) -> list[float]:
    """Symmetric channelwise relative change between two saved profiles."""
    a = profile_values(report, first)
    b = profile_values(report, second)
    scale = np.maximum(np.maximum(np.abs(a), np.abs(b)), np.finfo(float).tiny)
    return [float(value) for value in np.abs(a - b) / scale]


def add_precision_summary(report: dict[str, Any]) -> None:
    """Add reproducible comparisons and a conservative control envelope."""
    comparison_profiles = {
        "patch_radius_q8_48x120": (
            "precision_q8_small_patch",
            "precision_q8_large_patch",
        ),
        "worldsheet_q8_48x120_to_64x160": (
            "precision_q8_large_patch",
            "precision_q8_worldsheet_refined",
        ),
        "worldsheet_q8_64x160_to_80x200": (
            "precision_q8_worldsheet_refined",
            "precision_q8_worldsheet_ultra",
        ),
        "momentum_nodes_q8_at_64x160": (
            "precision_q8_worldsheet_refined",
            "precision_q8_refined_quadrature",
        ),
        "recursion_q8_to_q9_at_80x200": (
            "precision_q8_worldsheet_ultra",
            "precision_q9_worldsheet_ultra",
        ),
    }
    comparisons: dict[str, Any] = {}
    for label, (first, second) in comparison_profiles.items():
        if first not in report["profiles"] or second not in report["profiles"]:
            continue
        shift = relative_profile_shift(report, first, second)
        comparisons[label] = {
            "first_profile": first,
            "second_profile": second,
            "channelwise_relative_shift": shift,
            "maximum_relative_shift": max(shift),
        }
    report["comparisons"] = comparisons

    best_name = "precision_q9_worldsheet_ultra"
    required_controls = (
        "patch_radius_q8_48x120",
        "worldsheet_q8_64x160_to_80x200",
        "momentum_nodes_q8_at_64x160",
        "recursion_q8_to_q9_at_80x200",
    )
    if best_name not in report["profiles"] or not all(
        name in comparisons for name in required_controls
    ):
        report.pop("precision_conclusion", None)
        return

    residual = np.asarray(
        report["profiles"][best_name]["relative_ansatz_residual"], dtype=float
    )
    controls = np.asarray(
        [comparisons[name]["channelwise_relative_shift"] for name in required_controls],
        dtype=float,
    )
    envelope = np.max(controls, axis=0)
    report["precision_conclusion"] = {
        "best_profile": best_name,
        "best_relative_ansatz_residual": [float(value) for value in residual],
        "maximum_best_relative_ansatz_residual": float(np.max(residual)),
        "conservative_observed_control_envelope": [
            float(value) for value in envelope
        ],
        "maximum_conservative_observed_control_envelope": float(np.max(envelope)),
        "ansatz_residual_over_control_envelope": [
            float(value) for value in residual / envelope
        ],
        "passes_within_observed_control_envelope": bool(np.all(residual <= envelope)),
        "interpretation": (
            "The ansatz passes at this asymmetric kinematic point within the "
            "largest observed channelwise variation from the final worldsheet, "
            "recursion-order, momentum-node, and patch-radius controls. This is "
            "a numerical validation at one point, not an analytic proof."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["production", "high", "q_plus2"],
        help="Stable-settings or custom precision profile names.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    profiles = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    profiles.update(CUSTOM_PROFILES)
    unknown = [name for name in args.profiles if name not in profiles]
    if unknown:
        parser.error(f"unknown profiles: {', '.join(unknown)}")

    omega0 = complex(1.0 / 3.0, 0.6)
    outgoing = [ratio * omega0 for ratio in (0.2, 0.3, 0.5)]
    energies = [*outgoing, omega0]
    candidate_abc = np.asarray(pair_channel_ansatz(energies), dtype=complex)
    candidate_m = candidate_abc[::-1]

    if args.output.exists() and not args.force:
        report = json.loads(args.output.read_text(encoding="utf-8"))
    else:
        report = {
            "status": "running",
            "kinematics": {
                "energy_ordering": "[omega1,omega2,omega3,omega0]",
                "omega0": complex_pair(omega0),
                "outgoing_ratios": [0.2, 0.3, 0.5],
                "energies": [complex_pair(value) for value in energies],
            },
            "ansatz": {
                "description": "pair_channel_ansatz; result recorded in scan order (M1,M2,M3)=(C,B,A)",
                "values": [complex_pair(value) for value in candidate_m],
            },
            "profiles": {},
        }

    for name in args.profiles:
        if name in report["profiles"] and not args.force:
            print(f"skipping completed profile {name}", flush=True)
            continue
        settings = profiles[name]
        print(f"running profile {name}", flush=True)
        started = time.time()
        result = evaluate_point(*outgoing, settings)
        elapsed = time.time() - started
        numerical = np.asarray([result[key] for key in ("M1", "M2", "M3")])
        relative = np.abs(numerical - candidate_m) / np.maximum(
            np.abs(candidate_m), np.finfo(float).tiny
        )
        adjacent = np.asarray(
            [result[f"estimated_rel_error_M{index}"] for index in range(1, 4)]
        )
        finite_adjacent = np.isfinite(adjacent)
        maximum_adjacent = (
            float(np.max(adjacent[finite_adjacent]))
            if np.any(finite_adjacent)
            else None
        )
        report["profiles"][name] = {
            "settings": settings,
            "runtime_seconds": elapsed,
            "numerical": [complex_pair(value) for value in numerical],
            "relative_ansatz_residual": [float(value) for value in relative],
            "maximum_relative_ansatz_residual": float(np.max(relative)),
            "adjacent_q_relative_change": [
                float(value) if np.isfinite(value) else None for value in adjacent
            ],
            "maximum_adjacent_q_relative_change": maximum_adjacent,
            "residual_over_adjacent_q_change": [
                (
                    float(relative[index] / max(adjacent[index], np.finfo(float).tiny))
                    if finite_adjacent[index]
                    else None
                )
                for index in range(3)
            ],
            "backend_diagnostics": {
                key: jsonable(value)
                for key, value in result.items()
                if key.startswith("block_")
            },
        }
        write_report(args.output, report)
        print(
            f"  max ansatz residual {np.max(relative):.6e}; "
            + (
                f"max adjacent-q change {maximum_adjacent:.6e}; "
                if maximum_adjacent is not None
                else "adjacent-q change not estimated; "
            )
            + f"runtime {elapsed:.1f}s",
            flush=True,
        )

    add_precision_summary(report)
    report["status"] = "complete"
    write_report(args.output, report)
    print(args.output)


if __name__ == "__main__":
    main()
