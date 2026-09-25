#!/usr/bin/env python3
"""Probe the equal-outgoing VVVV amplitude toward the proposed omega=i/2 pole.

The equal-energy convention is

    omega1 = omega2 = omega3 = omega,   omega0 = 3 omega,

and the proposed coefficient is

    M(omega) = -3 pi omega^4 / (1 + 2 i omega).

The current numerical backend keeps the internal Liouville momentum contour on
the real axis.  Consequently, rows with pole_margin = 1 - 4 Im(omega) <= 0 are
explicitly marked as being outside the documented no-contour-crossing domain.
Finite values in that region are diagnostics of the raw prescription, not
automatically values of the physical analytic continuation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import mpmath as mp
import numpy as np


CODE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = CODE_DIR.parent
FIT_DIR = CODE_DIR / "heterotic_so23_1to3_vvvv_fit_bundle"
DEFAULT_OUTPUT = (
    WORKSPACE_DIR
    / "Data Set"
    / "high_accuracy"
    / "spin23_vvvv_equal_pole_c_recursion_elliptic_raw_contour.json"
)

if str(FIT_DIR) not in sys.path:
    sys.path.insert(0, str(FIT_DIR))

import heterotic_so23_1to3_fast as fast  # noqa: E402


equal_outgoing_coefficient_regularized_fast = (
    fast.equal_outgoing_coefficient_regularized_fast
)


PROFILES: dict[str, dict[str, Any]] = {
    "q5_standard": {
        "q_order": 5,
        "p_nodes": "segmented",
        "p_max": 4.0,
        "epsilon0": 0.08,
        "epsilon1": 0.06,
        "theta_orders": (24, 24, 60),
        "radial_order": 24,
        "disk_total_order": 14,
        "lens_radial_order": 20,
        "lens_angular_order": 56,
        "lens_power": 3.0,
    },
    "q6_dense": {
        "q_order": 6,
        "p_nodes": [20, 23, 30, 23, 20, 18, 18],
        "p_max": 6.0,
        "epsilon0": 0.08,
        "epsilon1": 0.06,
        "theta_orders": (28, 28, 72),
        "radial_order": 28,
        "disk_total_order": 18,
        "lens_radial_order": 24,
        "lens_angular_order": 72,
        "lens_power": 3.0,
    },
    "q6_near_pole": {
        "q_order": 6,
        "p_nodes": [72, 36, 30, 24, 20, 18, 18],
        "p_max": 6.0,
        "p_cut": 0.0002,
        "reference_p_max": 0.18,
        "cancellation_limit": 1.0e7,
        "epsilon0": 0.08,
        "epsilon1": 0.06,
        "theta_orders": (32, 32, 80),
        "radial_order": 32,
        "disk_total_order": 20,
        "lens_radial_order": 28,
        "lens_angular_order": 88,
        "lens_power": 3.0,
    },
    "q7_near_pole_refined": {
        "q_order": 7,
        "p_nodes": [96, 44, 36, 28, 22, 20, 20],
        "p_max": 6.0,
        "p_cut": 0.0001,
        "reference_p_max": 0.18,
        "cancellation_limit": 1.0e7,
        "epsilon0": 0.08,
        "epsilon1": 0.06,
        "theta_orders": (40, 40, 96),
        "radial_order": 40,
        "disk_total_order": 24,
        "lens_radial_order": 34,
        "lens_angular_order": 104,
        "lens_power": 3.0,
    },
}

DEFAULT_Y_VALUES = (
    0.20,
    0.23,
    0.24,
    0.245,
    0.249,
    0.251,
    0.255,
    0.26,
    0.28,
    0.30,
    0.35,
    0.40,
    0.45,
    0.475,
    0.49,
    0.495,
    0.497,
    0.498,
    0.499,
    0.4995,
    0.4998,
)


def complex_pair(value: complex) -> list[float]:
    value = complex(value)
    return [float(value.real), float(value.imag)]


def proposed_amplitude(omega: complex) -> complex:
    return -3.0 * math.pi * omega**4 / (1.0 + 2.0j * omega)


def proposed_residue() -> complex:
    return 3.0j * math.pi / 32.0


def equal_fixed_p_moduli_integrand(
    omega: complex, momentum: complex, settings: dict[str, Any]
) -> complex:
    """Return the equal-energy A-channel integrand before the dP/pi measure."""
    q_order = int(settings["q_order"])
    energies = [complex(omega)] * 3 + [3.0 * complex(omega)]
    h1, h2, h3, h4 = map(fast.h_of_p, energies)
    hp = fast.h_of_p(momentum)
    max_level2 = 2 * q_order + 1

    s_primary_computer = fast.ref.NSBlockComputer(
        h4, h3, h2, h1, False, False, max_level2=max_level2
    )
    s_starstar_computer = fast.ref.NSBlockComputer(
        h4, h3, h2, h1, True, True, max_level2=max_level2
    )
    s_primary = fast.ref.ResummedNSBlock(s_primary_computer, hp, q_order)
    s_starstar = fast.ref.ResummedNSBlock(s_starstar_computer, hp, q_order)
    even_structure = complex(
        fast.ref.c_even(omega, omega, momentum)
        * fast.ref.c_even(omega, 3.0 * omega, momentum)
    )
    odd_structure = complex(
        fast.ref.c_odd(omega, omega, momentum)
        * fast.ref.c_odd(omega, 3.0 * omega, momentum)
    )
    s_kernel = fast.PKernel(
        momentum,
        1.0,
        s_primary,
        s_starstar,
        even_structure,
        odd_structure,
        "reference",
        None,
    )

    # The equal-outgoing t-channel has the same external weights, but retain
    # the explicit routed construction used by build_t_channel_data.
    t_primary_computer = fast.ref.NSBlockComputer(
        h4, h1, h2, h3, False, False, max_level2=max_level2
    )
    t_starstar_computer = fast.ref.NSBlockComputer(
        h4, h1, h2, h3, True, True, max_level2=max_level2
    )
    t_primary_series = fast._build_direct_block_series(
        t_primary_computer, hp, q_order
    )
    t_starstar_series = fast._build_direct_block_series(
        t_starstar_computer, hp, q_order
    )
    t_kernel = fast.TKernel(
        momentum,
        1.0,
        t_primary_computer,
        t_starstar_computer,
        even_structure,
        odd_structure,
        hp,
        t_primary_series,
        t_starstar_series,
        "reference",
        None,
    )
    t_kernel.resummed_primary = fast.ref.ResummedNSBlock(t_primary_computer,hp,q_order)
    t_kernel.resummed_adjacent = t_kernel.resummed_primary.adjacent_component(
        fast.ref.ResummedNSBlock(t_starstar_computer,hp,q_order))

    s_annulus = fast.full_plane_annulus(
        [s_kernel],
        [s_kernel],
        energies,
        settings["epsilon0"],
        settings["epsilon1"],
        settings["theta_orders"],
        settings["radial_order"],
        [t_kernel], [t_kernel], q_order,
    )
    s_disk = fast.full_analytic_s_disk(
        [s_kernel],
        [s_kernel],
        energies,
        settings["epsilon0"],
        q_order,
        settings["disk_total_order"],
    )
    t_lens = fast.full_lens_integral(
        [t_kernel],
        [t_kernel],
        energies,
        settings["epsilon1"],
        q_order,
        settings["lens_radial_order"],
        settings["lens_angular_order"],
        settings["lens_power"],
    )
    return complex((s_annulus + s_disk + t_lens)[0])


def crossed_pole_residue_probe(
    y: float,
    settings: dict[str, Any],
    offsets: tuple[float, ...] = (2.0e-4, 1.0e-4, 5.0e-5, 2.5e-5),
) -> dict[str, Any]:
    """Numerically extract the first crossed internal-P pole residue."""
    if y <= 0.25:
        raise ValueError("the first crossed pole exists on this branch only for y>1/4")
    omega = 1.0j * float(y)
    pole = 1.0j * (4.0 * y - 1.0)
    estimates = []
    for offset in offsets:
        plus = equal_fixed_p_moduli_integrand(omega, pole + offset, settings)
        minus = equal_fixed_p_moduli_integrand(omega, pole - offset, settings)
        estimate = 0.5 * (offset * plus + (-offset) * minus)
        estimates.append(
            {
                "offset": offset,
                "plus_integrand": complex_pair(plus),
                "minus_integrand": complex_pair(minus),
                "residue": complex_pair(estimate),
            }
        )
    return {
        "y": float(y),
        "omega": complex_pair(omega),
        "crossed_internal_pole": complex_pair(pole),
        "residue_estimates": estimates,
    }


def add_extrapolated_contour_result(
    residue_row: dict[str, Any],
    raw_amplitude: complex,
    prediction: complex,
) -> None:
    """Extrapolate the symmetric pole-residue probe linearly in offset^2.

    For an analytic simple-pole remainder, the symmetric estimator has only
    even powers of the real offset.  A quadratic fit in offset^2 through the
    stored probes removes the O(offset^2) and O(offset^4) terms.  Four probe
    offsets are retained so this is an overdetermined fit rather than an exact
    interpolation through three samples.
    """
    estimates = residue_row["residue_estimates"]
    if len(estimates) < 3:
        return
    offsets2 = np.asarray([float(item["offset"]) ** 2 for item in estimates])
    residues = np.asarray([complex(*item["residue"]) for item in estimates])
    degree = min(2, len(estimates) - 1)
    residue = complex(
        np.polyfit(offsets2, residues.real, degree)[-1],
        np.polyfit(offsets2, residues.imag, degree)[-1],
    )
    correction = -2.0j * residue
    corrected = raw_amplitude + correction
    omega = complex(*residue_row["omega"])
    exact_residue = proposed_residue()
    residue_estimator = (omega - 0.5j) * corrected
    residue_row["zero_offset_extrapolation"] = {
        "model": "quadratic fit in offset^2 through all stored probes",
        "internal_p_residue": complex_pair(residue),
        "contour_correction": complex_pair(correction),
        "contour_corrected_amplitude": complex_pair(corrected),
        "relative_corrected_formula_difference": float(
            abs(corrected - prediction)
            / max(abs(prediction), np.finfo(float).tiny)
        ),
        "omega_pole_residue_estimator": complex_pair(residue_estimator),
        "relative_to_exact_proposed_residue": float(
            abs(residue_estimator - exact_residue) / abs(exact_residue)
        ),
    }


def add_residue_fit_summaries(report: dict[str, Any]) -> None:
    """Fit the corrected finite-distance residue estimators to the pole."""
    summaries: dict[str, Any] = {}
    exact = proposed_residue()
    for profile_name, profile in report.get("profiles", {}).items():
        points = []
        for row in profile.get("crossed_pole_residue_probes", []):
            extrapolation = row.get("zero_offset_extrapolation")
            if extrapolation is None or float(row["y"]) < 0.475:
                continue
            omega = complex(*row["omega"])
            corrected = complex(*extrapolation["contour_corrected_amplitude"])
            points.append(
                (
                    float(0.5 - row["y"]),
                    (omega - 0.5j) * corrected,
                    float(row["y"]),
                )
            )
        if len(points) < 3:
            continue
        points.sort(reverse=True)
        fits = []
        for exclude_closest in (0, 1, 2):
            selected = points[: len(points) - exclude_closest]
            if len(selected) < 3:
                continue
            x = np.asarray([item[0] for item in selected])
            values = np.asarray([item[1] for item in selected])
            for degree in (2, 3):
                if len(selected) <= degree:
                    continue
                estimate = complex(
                    np.polyfit(x, values.real, degree)[-1],
                    np.polyfit(x, values.imag, degree)[-1],
                )
                fits.append(
                    {
                        "degree_in_delta": degree,
                        "delta_definition": "delta=0.5-Im(omega)",
                        "y_min": min(item[2] for item in selected),
                        "y_max": max(item[2] for item in selected),
                        "number_of_points": len(selected),
                        "pole_residue_extrapolation": complex_pair(estimate),
                        "relative_to_exact_proposed_residue": float(
                            abs(estimate - exact) / abs(exact)
                        ),
                    }
                )
        summaries[profile_name] = {
            "finite_distance_estimators": [
                {
                    "y": item[2],
                    "delta": item[0],
                    "residue_estimator": complex_pair(item[1]),
                }
                for item in points
            ],
            "fits": fits,
        }
    report["omega_pole_residue_fit_summaries"] = summaries


def evaluate_row(y: float, settings: dict[str, Any]) -> dict[str, Any]:
    omega = 1.0j * float(y)
    started = time.time()
    try:
        numerical = complex(
            equal_outgoing_coefficient_regularized_fast(3.0 * omega, **settings)
        )
        status = "ok"
        diagnostic = None
    except Exception as exc:  # retain failures as part of the pole diagnostic
        numerical = complex(np.nan, np.nan)
        status = "failed"
        diagnostic = f"{type(exc).__name__}: {exc}"
    elapsed = time.time() - started

    prediction = proposed_amplitude(omega)
    exact_residue = proposed_residue()
    numerical_residue = (omega - 0.5j) * numerical
    predicted_residue = (omega - 0.5j) * prediction
    missing_residue = (omega - 0.5j) * (prediction - numerical)
    finite = np.isfinite(numerical.real) and np.isfinite(numerical.imag)
    return {
        "y": float(y),
        "omega": complex_pair(omega),
        "omega0": complex_pair(3.0 * omega),
        "pole_margin": float(1.0 - 4.0 * y),
        "inside_documented_real_p_contour_domain": bool(1.0 - 4.0 * y > 0.0),
        "status": status,
        "diagnostic": diagnostic,
        "runtime_seconds": elapsed,
        "numerical": complex_pair(numerical),
        "prediction": complex_pair(prediction),
        "relative_amplitude_difference": (
            float(abs(numerical - prediction) / max(abs(prediction), np.finfo(float).tiny))
            if finite
            else None
        ),
        "numerical_residue_estimator": complex_pair(numerical_residue),
        "prediction_residue_estimator": complex_pair(predicted_residue),
        "missing_contour_residue_estimator": complex_pair(missing_residue),
        "exact_proposed_residue": complex_pair(exact_residue),
        "relative_numerical_residue_error": (
            float(abs(numerical_residue - exact_residue) / abs(exact_residue))
            if finite
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", nargs="+", default=["q5_standard"])
    parser.add_argument("--y", nargs="+", type=float, default=list(DEFAULT_Y_VALUES))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--contour-residue-y",
        nargs="+",
        type=float,
        default=[],
        help="extract the first crossed internal-P pole residue at these y>1/4 values",
    )
    parser.add_argument(
        "--skip-amplitudes",
        action="store_true",
        help="retain saved amplitude rows and run only requested contour-residue probes",
    )
    args = parser.parse_args()

    unknown = [name for name in args.profiles if name not in PROFILES]
    if unknown:
        parser.error(f"unknown profiles: {', '.join(unknown)}")
    if any(y <= 0.0 or y >= 0.5 for y in args.y):
        parser.error("all y values must satisfy 0 < y < 0.5")
    if any(y <= 0.25 or y >= 0.5 for y in args.contour_residue_y):
        parser.error("contour-residue y values must satisfy 1/4 < y < 1/2")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.force:
        report = json.loads(args.output.read_text(encoding="utf-8"))
        if report.get("numerical_algorithm") != fast.ref.elliptic_conversion.ALGORITHM_VERSION:
            parser.error("output uses another numerical algorithm; choose a fresh report")
    else:
        report = {"profiles": {}}
    report.update(
        {
            "status": "running",
            "series_parameter": "elliptic_nome",
            "numerical_algorithm": fast.ref.elliptic_conversion.ALGORITHM_VERSION,
            "q_definition": "qhat=exp(-pi*K(1-z)/K(z))",
            "local_ope_policy": "back_expansion_of_truncated_nome",
            "accuracy_status": "requires_independent_convergence_checks",
            "convention": "omega1=omega2=omega3=omega=i*y; omega0=3*omega",
            "proposed_formula": "M(omega)=-3*pi*omega^4/(1+2*i*omega)",
            "proposed_pole": complex_pair(0.5j),
            "proposed_residue_in_omega": complex_pair(proposed_residue()),
            "current_driver_mpmath_decimal_precision": 70,
            "contour_warning": (
                "The implemented internal-P contour is undeformed. Rows with "
                "pole_margin=1-4*y <= 0 are outside its documented no-crossing "
                "domain and cannot by themselves test the physical continuation."
            ),
        }
    )

    # The c-recursion coefficients develop severe intermediate cancellation
    # for q>=7 and P close to zero on this analytically continued slice.
    # Binary64 or mpmath's default precision can therefore generate a false
    # endpoint blow-up even though the final block is finite.
    mp.mp.dps = 70

    for profile_name in args.profiles:
        settings = PROFILES[profile_name]
        if not args.skip_amplitudes:
            existing_profile = report["profiles"].get(profile_name, {})
            rows_by_y = {
                float(row["y"]): row for row in existing_profile.get("rows", [])
            }
            print(f"running {profile_name}", flush=True)
            for y in args.y:
                row = evaluate_row(y, settings)
                rows_by_y[float(y)] = row
                rows = [rows_by_y[key] for key in sorted(rows_by_y)]
                print(
                    f"  y={y:.6f} margin={row['pole_margin']:+.3f} "
                    f"amplitude_rel={row['relative_amplitude_difference']} "
                    f"residue_rel={row['relative_numerical_residue_error']}",
                    flush=True,
                )
                profile_payload = dict(existing_profile)
                profile_payload.update(
                    {
                        "settings": settings,
                        "rows": rows,
                        "amplitude_mpmath_decimal_precision": mp.mp.dps,
                    }
                )
                report["profiles"][profile_name] = profile_payload
                args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        elif profile_name not in report["profiles"]:
            report["profiles"][profile_name] = {"settings": settings, "rows": []}

        if args.contour_residue_y:
            mp.mp.dps = 70
            existing_residue_rows = report["profiles"][profile_name].get(
                "crossed_pole_residue_probes", []
            )
            residue_rows_by_y = {
                float(row["y"]): row for row in existing_residue_rows
            }
            print(f"extracting crossed-pole residues for {profile_name}", flush=True)
            for y in args.contour_residue_y:
                residue_row = crossed_pole_residue_probe(y, settings)
                saved_rows = report["profiles"][profile_name].get("rows", [])
                saved = next(
                    (row for row in saved_rows if abs(float(row["y"]) - y) < 1.0e-14),
                    None,
                )
                if saved is None:
                    saved = evaluate_row(y, settings)
                raw_amplitude = complex(*saved["numerical"])
                prediction = proposed_amplitude(1.0j * y)
                for estimate in residue_row["residue_estimates"]:
                    p_residue = complex(*estimate["residue"])
                    # The half-line dP/pi representation contains the two
                    # reflection-related poles of the full P contour.  Keeping
                    # the original analytic branch after their exchange adds
                    # -2 i times the upper-half-plane residue.
                    correction = -2.0j * p_residue
                    corrected = raw_amplitude + correction
                    estimate["contour_correction"] = complex_pair(correction)
                    estimate["contour_corrected_amplitude"] = complex_pair(corrected)
                    estimate["relative_corrected_formula_difference"] = float(
                        abs(corrected - prediction)
                        / max(abs(prediction), np.finfo(float).tiny)
                    )
                residue_row["raw_undeformed_amplitude"] = complex_pair(raw_amplitude)
                residue_row["prediction"] = complex_pair(prediction)
                add_extrapolated_contour_result(
                    residue_row, raw_amplitude, prediction
                )
                residue_rows_by_y[float(y)] = residue_row
                residue_rows = [
                    residue_rows_by_y[key] for key in sorted(residue_rows_by_y)
                ]
                last = complex(*residue_row["residue_estimates"][-1]["residue"])
                corrected_error = residue_row["residue_estimates"][-1][
                    "relative_corrected_formula_difference"
                ]
                print(
                    f"  y={y:.6f} Res_P={last} corrected_rel={corrected_error:.3e}",
                    flush=True,
                )
            report["profiles"][profile_name]["crossed_pole_residue_probes"] = (
                residue_rows
            )
            report["profiles"][profile_name][
                "residue_probe_mpmath_decimal_precision"
            ] = mp.mp.dps
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    # Also upgrade older saved rows when this script is rerun, without
    # requiring the expensive moduli integrals to be recomputed.
    for profile in report.get("profiles", {}).values():
        for residue_row in profile.get("crossed_pole_residue_probes", []):
            add_extrapolated_contour_result(
                residue_row,
                complex(*residue_row["raw_undeformed_amplitude"]),
                complex(*residue_row["prediction"]),
            )
    # Provenance migration for rows generated by earlier revisions of this
    # probe, before precision was written into each profile.  The old q5/q6
    # amplitude scans used mpmath's 15-digit default; all contour probes used
    # 70 digits.  q6_near_pole was regenerated at 70 digits in the present
    # review, and the q7 profile has always required 70 digits.
    legacy_precision = {
        "q5_standard": (15, 70),
        "q6_dense": (15, 70),
        "q6_near_pole": (70, 70),
        "q7_near_pole_refined": (70, 70),
    }
    for name, profile in report.get("profiles", {}).items():
        if name in legacy_precision:
            amplitude_dps, residue_dps = legacy_precision[name]
            profile.setdefault("amplitude_mpmath_decimal_precision", amplitude_dps)
            if profile.get("crossed_pole_residue_probes"):
                profile.setdefault(
                    "residue_probe_mpmath_decimal_precision", residue_dps
                )
    add_residue_fit_summaries(report)
    report["status"] = "complete"
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
