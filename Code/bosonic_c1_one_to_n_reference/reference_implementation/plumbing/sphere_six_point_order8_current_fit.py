#!/usr/bin/env python3
"""Construct and fit the current paired order-8 sphere 1->5 result.

This target-free postprocessor uses the completed 30-point blind Cannon data.
At each t it adds the paired order-6-to-8 shift, measured on common Sobol
points, to the full-statistics order-6 production mean.  No matrix-model
coefficient or target curve is present in this file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


STATUS = "sphere_1to5_order8_30point_target_free_fit"
EXPECTED_SOURCE_SHA256 = "e460d5de342a61641da662d3f08bef81e2e1759e7220ae8dc365676850f152d7"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def design(t: np.ndarray) -> np.ndarray:
    return np.column_stack((np.ones_like(t), t, t**2, t**3))


def weighted_fit(t: np.ndarray, q: np.ndarray, errors: np.ndarray, label: str) -> dict[str, Any]:
    matrix = design(t)
    weights = 1.0 / errors**2
    normal = matrix.T @ (weights[:, None] * matrix)
    covariance = np.linalg.inv(normal)
    coefficients = covariance @ (matrix.T @ (weights * q))
    residuals = q - matrix @ coefficients
    return {
        "weighting": label,
        "coefficients_in_t": [float(value) for value in coefficients],
        "coefficient_standard_errors": [
            float(value) for value in np.sqrt(np.diag(covariance))
        ],
        "coefficient_covariance": [
            [float(value) for value in row] for row in covariance
        ],
        "chi_squared": float(np.sum((residuals / errors) ** 2)),
        "degrees_of_freedom": len(t) - 4,
        "maximum_absolute_residual": float(np.max(np.abs(residuals))),
        "rms_residual": float(np.sqrt(np.mean(residuals**2))),
    }


def unweighted_fit(t: np.ndarray, q: np.ndarray) -> dict[str, Any]:
    matrix = design(t)
    coefficients, *_ = np.linalg.lstsq(matrix, q, rcond=None)
    residuals = q - matrix @ coefficients
    return {
        "weighting": "unweighted sensitivity",
        "coefficients_in_t": [float(value) for value in coefficients],
        "maximum_absolute_residual": float(np.max(np.abs(residuals))),
        "rms_residual": float(np.sqrt(np.mean(residuals**2))),
    }


def build_fit(source_path: Path, output_path: Path) -> dict[str, Any]:
    actual_hash = sha256(source_path)
    if actual_hash != EXPECTED_SOURCE_SHA256:
        raise ValueError("the completed 30-point blind source table changed")
    source = json.loads(source_path.read_text())
    if source.get("status") != "worldsheet_only_complete_unvalidated":
        raise ValueError("expected the completed blind 30-point table")
    if source.get("target_formula_available") is not False:
        raise ValueError("target information is present in the worldsheet source")
    raw_points = source["points"]
    if len(raw_points) != 30:
        raise ValueError("expected thirty worldsheet points")

    points: list[dict[str, float]] = []
    for item in raw_points:
        t = float(item["t"])
        if not 0.0 < t < 1.0 / 3.0:
            raise ValueError("a point lies outside the residue-free chamber")
        q6 = float(item["Q5_worldsheet"]["real"])
        q6_error = float(item["Q5_qmc_standard_error"]["real"])
        block = item["paired_systematics"]["block_order"]
        shift = float(block["paired_shift_Q5"]["real"])
        shift_error = float(block["paired_standard_error_Q5"]["real"])
        order8 = q6 + shift
        statistical_error = float(np.hypot(q6_error, shift_error))
        points.append(
            {
                "t": t,
                "Q5_order6_full_statistics": q6,
                "Q5_order6_qmc_standard_error": q6_error,
                "paired_Q5_order8_minus_order6": shift,
                "paired_shift_standard_error": shift_error,
                "Q5_order8_estimate": order8,
                "Q5_order8_propagated_statistical_error": statistical_error,
                "available_numerical_envelope_proxy_Q5": float(
                    item["stability_envelope_Q5"]
                ),
                "amplitude_imaginary_order8_estimate": -5.0 * t**6 * order8,
            }
        )

    t = np.asarray([point["t"] for point in points], dtype=float)
    if list(t) != sorted(t) or len(set(t)) != len(t):
        raise ValueError("t values must be unique and increasing")
    q8 = np.asarray([point["Q5_order8_estimate"] for point in points], dtype=float)
    statistical = np.asarray(
        [point["Q5_order8_propagated_statistical_error"] for point in points],
        dtype=float,
    )
    envelope = np.asarray(
        [point["available_numerical_envelope_proxy_Q5"] for point in points],
        dtype=float,
    )
    result: dict[str, Any] = {
        "status": STATUS,
        "program": Path(__file__).name,
        "source_worldsheet_scan": str(source_path),
        "source_worldsheet_scan_sha256": actual_hash,
        "target_information_used": False,
        "kinematic_domain": "omega=i*t with 0<t<1/3",
        "point_count": len(points),
        "order8_estimator": (
            "full-statistics order-6 mean plus the mean paired order-8-minus-order-6 "
            "shift evaluated on identical Sobol points"
        ),
        "sampling": {
            "order6_production": "14 scrambled Sobol replicates of 2^15 points",
            "paired_order6_to_order8": "6 scrambled Sobol replicates of 2^11 points",
        },
        "uncertainty_interpretation": {
            "propagated_statistical_error": (
                "quadrature sum of production-mean QMC error and paired-shift QMC error"
            ),
            "available_numerical_envelope_proxy": (
                "maximum stored order-6 campaign stability component; used as a "
                "conservative comparison proxy, not an order-8-to-higher certificate"
            ),
            "formal_order8_to_higher_convergence_certificate": False,
        },
        "fit_ansatz": "Q_5(i*t)=a+b*t+c*t^2+d*t^3",
        "primary_fit": weighted_fit(
            t,
            q8,
            statistical,
            "diagonal propagated QMC errors of the paired order-8 estimator",
        ),
        "available_envelope_weighted_sensitivity_fit": weighted_fit(
            t,
            q8,
            envelope,
            "diagonal available numerical-envelope proxy",
        ),
        "unweighted_sensitivity_fit": unweighted_fit(t, q8),
        "points": points,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    base = Path(__file__).parent / "results" / "sphere_six_point_1to5"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=base / "cannon_blind30_3h_v2" / "assembled" / "worldsheet_scan_unfrozen.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=base / "order8_30point_current" / "order8_target_free_fit.json",
    )
    arguments = parser.parse_args()
    result = build_fit(arguments.source, arguments.output)
    print(json.dumps(result["primary_fit"], indent=2, sort_keys=True))
    print(arguments.output)


if __name__ == "__main__":
    main()
