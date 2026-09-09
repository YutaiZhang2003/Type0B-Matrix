#!/usr/bin/env python3
"""Apply BRY normalization to the wall-one scan and compare with the MQM."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    HERE
    / "results"
    / "type0b_sphere_four_point_wall_one_ten_point_scan_positive_sheet_m30.json"
)
DEFAULT_OUTPUT = (
    HERE
    / "results"
    / "type0b_sphere_four_point_wall_one_ten_point_matrix_comparison_positive_sheet_m30.json"
)


def _complex(payload: dict[str, float]) -> complex:
    return complex(float(payload["real"]), float(payload["imag"]))


def _json_complex(value: complex) -> dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag)}


def bry_worldsheet_coefficient(reduced_moduli_integral: complex) -> complex:
    r"""Return the coefficient of ``delta(energy) mu_F^-2`` in BRY units."""

    return 8.0j * complex(reduced_moduli_integral) / math.pi


def matrix_model_coefficient(
    incoming_energy: complex,
    outgoing_energies: tuple[complex, complex, complex],
) -> complex:
    r"""BRY tree-level right-side ``1->3`` matrix-model coefficient."""

    omega1, omega2, omega3 = outgoing_energies
    omega = complex(incoming_energy)
    return 8.0j * omega * omega1 * omega2 * omega3 * (1.0 + 2.0j * omega)


def _chi_square_survival_even(value: float, degrees_of_freedom: int) -> float:
    """Exact chi-square survival function for a positive even number of dof."""

    if degrees_of_freedom <= 0 or degrees_of_freedom % 2:
        raise ValueError("degrees_of_freedom must be positive and even")
    half = 0.5 * float(value)
    return math.exp(-half) * sum(
        half**order / math.factorial(order)
        for order in range(degrees_of_freedom // 2)
    )


def compare_scan(
    payload: dict[str, Any],
    *,
    maximum_relative_standard_error: float = 0.15,
) -> dict[str, Any]:
    tolerance = float(maximum_relative_standard_error)
    if tolerance <= 0.0 or not math.isfinite(tolerance):
        raise ValueError("maximum_relative_standard_error must be positive")
    comparisons: list[dict[str, Any]] = []
    chi_square = 0.0
    for point in payload["points"]:
        if point["status"] not in ("integrated", "integrated-low-precision"):
            raise ValueError(f"point {point['index']} was not integrated")
        amplitude = point["amplitude"]
        reduced = _complex(amplitude["mean"])
        outgoing = tuple(
            _complex(value) for value in amplitude["outgoing_energies"]
        )
        if len(outgoing) != 3:
            raise ValueError("a four-point amplitude needs three outgoing energies")
        incoming = _complex(amplitude["incoming_energy"])
        conservation_error = abs(incoming - sum(outgoing))
        if conservation_error > 1.0e-12:
            raise ValueError("the scan point violates energy conservation")

        worldsheet = bry_worldsheet_coefficient(reduced)
        matrix = matrix_model_coefficient(incoming, outgoing)  # type: ignore[arg-type]
        residual = worldsheet - matrix
        # Multiplication by 8i/pi exchanges the real and imaginary errors.
        error_real = 8.0 * float(amplitude["standard_error_imag"]) / math.pi
        error_imag = 8.0 * float(amplitude["standard_error_real"]) / math.pi
        if error_real <= 0.0 or error_imag <= 0.0:
            raise ValueError("the comparison requires positive component errors")
        point_chi_square = (
            (residual.real / error_real) ** 2
            + (residual.imag / error_imag) ** 2
        )
        relative_standard_error = math.hypot(error_real, error_imag) / max(
            abs(worldsheet), 1.0e-300
        )
        chi_square += point_chi_square
        comparisons.append(
            {
                "index": int(point["index"]),
                "x": float(point["x"]),
                "t": float(point["t"]),
                "incoming_energy": _json_complex(incoming),
                "outgoing_energies": [_json_complex(value) for value in outgoing],
                "reduced_moduli_integral": _json_complex(reduced),
                "worldsheet_coefficient": _json_complex(worldsheet),
                "worldsheet_standard_error_real": error_real,
                "worldsheet_standard_error_imag": error_imag,
                "matrix_model_coefficient": _json_complex(matrix),
                "residual": _json_complex(residual),
                "relative_complex_discrepancy": abs(residual) / abs(matrix),
                "chi_square_contribution": point_chi_square,
                "relative_standard_error": relative_standard_error,
                "precision_passed": relative_standard_error <= tolerance,
                "crossing_relative_spread": float(
                    point["crossing_audit"]["relative_spread"]
                ),
            }
        )

    degrees_of_freedom = 2 * len(comparisons)
    precision_passed = all(point["precision_passed"] for point in comparisons)
    return {
        "description": (
            "BRY-normalized wall-one worldsheet scan compared with the "
            "tree-level Type-0B matrix-model 1->3 coefficient"
        ),
        "stripped_factor": "delta(omega-sum omega_i) * mu_F^(-2)",
        "dictionary": {
            "omega_matrix_model": "2 * omega_worldsheet",
            "g_s": "4 / (pi * mu_F)",
            "C_S2": "pi / g_s^2",
            "additional_leg_pole_factor": False,
        },
        "worldsheet_formula": "A_WS = (8 i / pi) M",
        "matrix_model_formula": (
            "A_MQM = 8 i omega omega1 omega2 omega3 (1 + 2 i omega)"
        ),
        "source": str(DEFAULT_INPUT),
        "continuation_sheet": payload.get(
            "continuation_origin",
            "legacy scan beginning at negative real Liouville momenta",
        ),
        "maximum_relative_standard_error": tolerance,
        "precision_gate_passed": precision_passed,
        "status": (
            "comparison-ready" if precision_passed else "unconverged-moduli-scan"
        ),
        "points": comparisons,
        "aggregate": {
            "chi_square": chi_square,
            "degrees_of_freedom": degrees_of_freedom,
            "chi_square_per_degree_of_freedom": chi_square / degrees_of_freedom,
            "nominal_p_value": _chi_square_survival_even(
                chi_square, degrees_of_freedom
            ),
            "valid_for_inference": precision_passed,
            "caveat": (
                "The nominal chi-square is not valid when the target-blind "
                "precision gate fails.  Even after that gate passes, it treats "
                "the real/imaginary randomized-QMC errors at different points "
                "as independent Gaussian errors."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--maximum-relative-standard-error", type=float, default=0.15
    )
    args = parser.parse_args()
    source = json.loads(args.input.read_text())
    result = compare_scan(
        source,
        maximum_relative_standard_error=args.maximum_relative_standard_error,
    )
    result["source"] = str(args.input.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for point in result["points"]:
        ws = _complex(point["worldsheet_coefficient"])
        mm = _complex(point["matrix_model_coefficient"])
        print(
            f"x={point['x']:.3f}, t={point['t']:.3f}: "
            f"A_WS={ws.real:+.6f}{ws.imag:+.6f}i, "
            f"A_MQM={mm.real:+.6f}{mm.imag:+.6f}i"
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
