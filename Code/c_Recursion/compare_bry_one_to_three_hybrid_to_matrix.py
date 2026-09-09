#!/usr/bin/env python3
"""Compare an unfrozen target-blind hybrid BRY run with the Type-0B MQM.

This is deliberately a post-run operation.  It verifies that the input JSON
contains no prior matrix-model data, records its SHA-256 digest, and never
promotes the worldsheet result to a frozen result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DEFAULT_WORLDSHEET = (
    RESULTS / "bry_one_to_three_hybrid_worldsheet_q12_production.json"
)
DEFAULT_OUTPUT = (
    RESULTS / "bry_one_to_three_hybrid_postrun_matrix_comparison.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def complex_value(pair: dict[str, float]) -> complex:
    return complex(pair["real"], pair["imag"])


def complex_pair(value: complex) -> dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worldsheet", type=Path, default=DEFAULT_WORLDSHEET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    worldsheet = json.loads(args.worldsheet.read_text())

    if worldsheet.get("status") != "worldsheet_only_unfrozen":
        raise ValueError("input is not an unfrozen worldsheet-only result")
    if worldsheet.get("comparison_performed") is not False:
        raise ValueError("input was contaminated by a prior comparison")
    if worldsheet.get("matrix_model_data_included") is not False:
        raise ValueError("input contains matrix-model data")

    omega = complex_value(worldsheet["kinematics"]["incoming_energy"])
    omega_out = complex_value(
        worldsheet["kinematics"]["each_outgoing_energy"]
    )
    worldsheet_reduced = complex_value(
        worldsheet["reduced_moduli_integral"]
    )
    worldsheet_amplitude = complex_value(
        worldsheet["worldsheet_amplitude_coefficient"]
    )

    matrix_reduced = math.pi * omega * omega_out**3 * (1.0 + 2j * omega)
    matrix_amplitude = 8j * omega * omega_out**3 * (1.0 + 2j * omega)
    expected_worldsheet_amplitude = 8j * worldsheet_reduced / math.pi
    normalization_residual = worldsheet_amplitude - expected_worldsheet_amplitude
    difference_reduced = worldsheet_reduced - matrix_reduced
    difference_amplitude = worldsheet_amplitude - matrix_amplitude
    relative_difference = abs(difference_reduced) / abs(matrix_reduced)

    payload = {
        "status": "preliminary_postrun_matrix_comparison",
        "worldsheet_was_frozen_before_comparison": False,
        "comparison_is_target_blind": True,
        "comparison_caveat": (
            "The source run is target-blind but remains unfrozen; the numerical "
            "difference must not be quoted as a final accuracy claim until "
            "production-level convergence checks pass."
        ),
        "worldsheet_source": {
            "path": str(args.worldsheet),
            "sha256": sha256(args.worldsheet),
            "status": worldsheet["status"],
            "settings": worldsheet["settings"],
        },
        "kinematics": worldsheet["kinematics"],
        "normalization": {
            "reduced_to_amplitude": "A = (8 i / pi) M",
            "worldsheet_normalization_residual": complex_pair(
                normalization_residual
            ),
        },
        "worldsheet": {
            "reduced_moduli_integral": complex_pair(worldsheet_reduced),
            "amplitude_coefficient": complex_pair(worldsheet_amplitude),
        },
        "matrix_model": {
            "formula_reduced": (
                "M_MQM = pi omega omega_1 omega_2 omega_3 "
                "(1 + 2 i omega)"
            ),
            "reduced_moduli_prediction": complex_pair(matrix_reduced),
            "amplitude_coefficient_prediction": complex_pair(matrix_amplitude),
        },
        "comparison": {
            "difference_reduced": complex_pair(difference_reduced),
            "absolute_difference_reduced": abs(difference_reduced),
            "difference_amplitude": complex_pair(difference_amplitude),
            "absolute_difference_amplitude": abs(difference_amplitude),
            "relative_complex_difference": relative_difference,
            "relative_complex_difference_percent": 100.0 * relative_difference,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"input sha256={payload['worldsheet_source']['sha256']}")
    print(f"relative difference={100.0 * relative_difference:.9f}%")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
