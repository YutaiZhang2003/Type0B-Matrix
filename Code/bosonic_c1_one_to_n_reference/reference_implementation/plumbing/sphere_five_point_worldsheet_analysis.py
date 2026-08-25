#!/usr/bin/env python3
"""Quarantined convergent-ray fitting experiment.

Numerically fitting on an imaginary-energy ray does not define an unambiguous
continuation to the physical domain.  The old helpers remain readable only to
audit earlier artifacts; production uses the direct i-epsilon finite part in
``sphere_five_point_physical_scan.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def quadratic_fit(points: list[dict[str, float]]) -> dict[str, object]:
    t_values = np.asarray([point["t"] for point in points], dtype=float)
    values = np.asarray([point["Q_real"] for point in points], dtype=float)
    design = np.column_stack((np.ones_like(t_values), t_values, t_values**2))
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    residuals = values - design @ coefficients
    return {
        "coefficients_in_t": [float(value) for value in coefficients],
        "coefficients_in_complex_omega": [
            complex(coefficients[0]),
            complex(-1.0j * coefficients[1]),
            complex(-coefficients[2]),
        ],
        "maximum_absolute_residual": float(np.max(np.abs(residuals))),
        "rms_residual": float(np.sqrt(np.mean(residuals**2))),
    }


def complex_coefficient_payload(values: list[complex]) -> list[dict[str, float]]:
    return [{"real": value.real, "imag": value.imag} for value in values]


def reduced_continuation(omega: np.ndarray, coefficients_in_t: list[float]) -> np.ndarray:
    constant, linear_t, quadratic_t = coefficients_in_t
    return constant - 1.0j * linear_t * omega - quadratic_t * omega**2


def xi_amplitude(omega: np.ndarray, coefficients_in_t: list[float]) -> np.ndarray:
    """Return ``mu^3 A_tree = 4 i omega^5 Q(omega)``."""

    return 4.0j * omega**5 * reduced_continuation(omega, coefficients_in_t)


def analyze(input_path: Path, output_directory: Path) -> dict[str, object]:
    source_bytes = input_path.read_bytes()
    source = json.loads(source_bytes)
    fits: dict[str, dict[str, object]] = {}
    for order, payload in source["dense_scans"].items():
        # Use the common five-point interval.  The extra level-six t=0.14
        # point is retained as a low-energy validation point, not allowed to
        # bias comparisons across block orders.
        common_points = [point for point in payload["points"] if point["t"] >= 0.18]
        fit = quadratic_fit(common_points)
        fit["coefficients_in_complex_omega"] = complex_coefficient_payload(
            fit["coefficients_in_complex_omega"]
        )
        fits[order] = fit

    omega = np.linspace(0.0, 0.60, 121)
    amplitudes = {
        order: xi_amplitude(omega, fit["coefficients_in_t"])
        for order, fit in fits.items()
    }
    central = amplitudes["8"]
    real_stack = np.vstack([value.real for value in amplitudes.values()])
    imag_stack = np.vstack([value.imag for value in amplitudes.values()])

    output_directory.mkdir(parents=True, exist_ok=True)
    curve_payload = []
    for index, energy in enumerate(omega):
        curve_payload.append(
            {
                "omega": float(energy),
                "amplitude_real": float(central.real[index]),
                "amplitude_imag": float(central.imag[index]),
                "block_envelope_real_min": float(np.min(real_stack[:, index])),
                "block_envelope_real_max": float(np.max(real_stack[:, index])),
                "block_envelope_imag_min": float(np.min(imag_stack[:, index])),
                "block_envelope_imag_max": float(np.max(imag_stack[:, index])),
            }
        )
    curve_path = output_directory / "worldsheet_curve.json"
    curve_path.write_text(json.dumps(curve_payload, indent=2) + "\n")

    result = {
        "status": "worldsheet_frozen_before_matrix_model_comparison",
        "source": str(input_path.resolve()),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "fit_variable": "Q(i t)=a+b t+c t^2; Q(omega)=a-i b omega-c omega^2",
        "normalization": "mu^3 A_tree(omega)=4 i omega^5 Q(omega)",
        "fits_by_block_order": fits,
        "central_block_order": 8,
        "systematic_band": "pointwise envelope of block orders 4, 6, and 8",
        "curve_table": str(curve_path.resolve()),
    }
    result_path = output_directory / "worldsheet_fit_frozen.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    raise RuntimeError(
        "the convergent-ray fit is quarantined; run "
        "sphere_five_point_physical_scan.py for the direct i-epsilon finite part"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parent
        / "results"
        / "sphere_five_point_1to4"
        / "worldsheet_convergent_scan.json",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path(__file__).parent / "results" / "sphere_five_point_1to4",
    )
    arguments = parser.parse_args()
    print(json.dumps(analyze(arguments.input, arguments.output_directory), indent=2))


if __name__ == "__main__":
    main()
