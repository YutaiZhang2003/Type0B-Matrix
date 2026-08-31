#!/usr/bin/env python3
"""Compare a preliminary Type-0B all-tachyon result at fixed complex energies.

This postprocessor never imports the worldsheet integrator or changes its data.
Matrix predictions support 1->2 through 1->5; worldsheet input currently supports
only the five-point (1->4) c-recursion cluster summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


SUMMARY_SCHEMA = "type0b-ns-fivepoint-order8-c-recursion-summary-v5"
SOURCE = "https://arxiv.org/pdf/2201.05621"


def encoded(value: complex) -> dict:
    value = complex(value)
    if not (math.isfinite(value.real) and math.isfinite(value.imag)):
        raise ValueError("nonfinite complex value")
    return {"real": value.real, "imag": value.imag}


def decoded(value: dict) -> complex:
    result = complex(value["real"], value["imag"])
    encoded(result)
    return result


def matrix_coefficients(outgoing: tuple[complex, ...]) -> dict:
    """Coefficients of delta(E) mu_F^(-(n-1)), in BRY worldsheet energies.

    BRY (2.12)-(2.13): the c=1 energy is twice the worldsheet energy,
    including a factor 1/2 from delta(E). Perturbatively T=(R+L)/2 and
    the two decoupled, identical matrix sectors imply A_T=2^(-n) A_R.
    No equality of worldsheet NS/R diagrams is assumed.
    """
    n = len(outgoing)
    if not 2 <= n <= 5:
        raise ValueError("implemented matrix predictions require 2 <= n <= 5")
    outgoing = tuple(complex(value) for value in outgoing)
    for value in outgoing:
        encoded(value)
    incoming = sum(outgoing)
    tachyon = 1j * incoming * math.prod(outgoing)
    for j in range(1, n - 1):
        tachyon *= j + 2j * incoming
    return {
        "process": f"1->{n}",
        "external_point_count": n + 1,
        "incoming_energy": encoded(incoming),
        "outgoing_energies": [encoded(value) for value in outgoing],
        "stripped_factor": f"delta(omega_in-sum omega_a) * mu_F^(-{n-1})",
        "all_tachyon": encoded(tachyon),
        "all_right_mode": encoded(2**n * tachyon),
        "formula_all_tachyon": "i Omega prod_a(omega_a) prod_{j=1}^{n-2}(j+2i Omega)",
        "matrix_basis_relation": "A_T = 2^(-n) A_R; T=(R+L)/2",
        "source": SOURCE,
    }


def configured_energies(config: dict) -> tuple[complex, ...]:
    physics = config["physics"]
    energies = tuple(float(v) for v in physics["real_outgoing_energies"])
    weights = tuple(float(v) for v in physics["epsilon_weights"])
    epsilon = float(physics["epsilon"])
    if len(weights) != len(energies):
        raise ValueError("epsilon weights and outgoing energies differ in length")
    if any(not math.isfinite(v) or v <= 0 for v in (*energies, *weights, epsilon)):
        raise ValueError("this protocol requires positive finite energies, weights and epsilon")
    return tuple(e + 1j * epsilon * w for e, w in zip(energies, weights))


def fivepoint_worldsheet(raw_integral: complex) -> complex:
    """Literal all-NS coefficient: (i/64) gs^5 C_S2 I5 = i I5/(pi^2 mu^3)."""
    return 1j * raw_integral / math.pi**2


def compare_summary(summary: dict, config: dict, config_sha256: str) -> dict:
    if summary.get("schema") != SUMMARY_SCHEMA:
        raise ValueError("only the five-point c-recursion summary is supported")
    if summary.get("config_sha256") != config_sha256:
        raise ValueError("summary/config SHA-256 mismatch")
    if summary.get("matrix_model_used") is not False:
        raise ValueError("expected independent worldsheet input")
    if summary.get("task_count") != config["array"]["task_count"]:
        raise ValueError("summary task count differs from production config")
    outgoing = configured_energies(config)
    if len(outgoing) != 4:
        raise ValueError("the worldsheet converter supports 1->4, not 1->5")
    prediction = matrix_coefficients(outgoing)
    matrix = decoded(prediction["all_tachyon"])
    rows = summary["radius_summaries"]
    radii = [float(row["collar_radius"]) for row in rows]
    expected = [float(v) for v in config["subtraction"]["collar_radii"]]
    if not expected or sorted(radii) != sorted(expected) or len(set(radii)) != len(radii):
        raise ValueError("summary must contain every configured collar radius exactly once")
    comparisons = []
    for row in rows:
        if row.get("block_backend") != "c":
            raise ValueError("expected c-recursion data")
        integral = decoded(row["integral_mean"])
        worldsheet = fivepoint_worldsheet(integral)
        residual = worldsheet - matrix
        errors = [float(row[f"standard_error_{part}"]) for part in ("real", "imag")]
        if any(not math.isfinite(v) or v < 0 for v in errors):
            raise ValueError("standard errors must be finite and nonnegative")
        comparisons.append({
            "collar_radius": row["collar_radius"],
            "raw_integral": encoded(integral),
            "worldsheet_all_tachyon": encoded(worldsheet),
            # Multiplication by i rotates the two component uncertainties.
            "worldsheet_qmc_standard_error_real": errors[1] / math.pi**2,
            "worldsheet_qmc_standard_error_imag": errors[0] / math.pi**2,
            "matrix_all_tachyon": encoded(matrix),
            "residual": encoded(residual),
            "relative_complex_discrepancy": abs(residual) / abs(matrix) if matrix else None,
            "replicate_count": row["replicate_count"],
            "face_collar_certificates_passed": row.get("face_collar_certificates_passed"),
        })
    return {
        "schema": "type0b-fixed-complex-frequency-comparison-v1",
        "status": "preliminary_fixed_complex_frequency_comparison",
        "worldsheet_source_status": summary.get("status"),
        "prediction": prediction,
        "frequency_epsilon": config["physics"]["epsilon"],
        "epsilon_extrapolation_required": False,
        "epsilon_extrapolation_performed": False,
        "normalization": {
            "g_s": "4/(pi mu_F)", "C_S2": "pi/g_s^2",
            "literal_fivepoint": "(i/64) g_s^5 C_S2 delta(E) I5",
            "stripped_worldsheet_all_tachyon": "i I5 / pi^2",
            "worldsheet_NS_R_diagram_equality_assumed": False,
        },
        "comparisons": comparisons,
        "collar_stability_differences_raw_integral": summary.get("collar_stability_differences", []),
        "convergence_certified": False,
        "error_scope": "QMC sampling only; block, momentum and collar systematics are not bounded here.",
        "continuation_caveat": "Uses the worldsheet run's continuation prescription; no contour or residue changes are made here.",
        "radius_selection": "all configured radii; no selection or fit to the matrix prediction",
        "matrix_model_used_in_postprocessing": True,
        "matrix_model_used_in_worldsheet_input": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--summary", type=Path, help="Omit to write only the matrix prediction.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = [args.config] + ([args.summary] if args.summary else [])
    if args.output.resolve() in [path.resolve() for path in inputs]:
        parser.error("output must not overwrite either input")
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    if args.summary:
        summary_bytes = args.summary.read_bytes()
        result = compare_summary(json.loads(summary_bytes), config, config_hash)
        result["summary_source"] = str(args.summary.resolve())
        result["summary_sha256"] = hashlib.sha256(summary_bytes).hexdigest()
    else:
        result = {
            "schema": "type0b-fixed-complex-frequency-prediction-v1",
            "status": "matrix_prediction_only_no_worldsheet_result",
            "prediction": matrix_coefficients(configured_energies(config)),
            "frequency_epsilon": config["physics"]["epsilon"],
            "epsilon_extrapolation_required": False,
        }
    result["config_source"] = str(args.config.resolve())
    result["config_sha256"] = config_hash
    result["comparison_code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"{result['status']}: {args.output}")


if __name__ == "__main__":
    main()
