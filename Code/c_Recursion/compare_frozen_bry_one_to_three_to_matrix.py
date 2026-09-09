#!/usr/bin/env python3
"""Post-freeze comparison of the BRY worldsheet result with Type-0B MQM."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DEFAULT_FROZEN = RESULTS / "bry_one_to_three_worldsheet_frozen.json"
DEFAULT_MANIFEST = RESULTS / "bry_one_to_three_worldsheet_freeze_manifest.json"
DEFAULT_OUTPUT = RESULTS / "bry_one_to_three_postfreeze_matrix_comparison.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def complex_value(pair) -> complex:
    return complex(pair["real"], pair["imag"])


def complex_pair(value: complex) -> dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    frozen_hash = sha256(args.frozen)
    if manifest.get("status") != "worldsheet_freeze_complete":
        raise ValueError("worldsheet freeze manifest is not complete")
    if manifest.get("comparison_allowed") is not True:
        raise ValueError("freeze manifest does not allow comparison")
    if manifest.get("frozen_worldsheet_sha256") != frozen_hash:
        raise ValueError("frozen worldsheet hash does not match the manifest")

    frozen = json.loads(args.frozen.read_text())
    if frozen.get("status") != "worldsheet_result_frozen":
        raise ValueError("input is not a frozen worldsheet result")
    if frozen.get("comparison_performed") is not False:
        raise ValueError("frozen input was contaminated by a prior comparison")
    if frozen.get("matrix_model_data_included") is not False:
        raise ValueError("frozen input contains matrix-model data")
    if frozen["blind_convergence"].get("passed") is not True:
        raise ValueError("frozen input did not pass blind convergence")

    omega = complex_value(frozen["kinematics"]["incoming_energy"])
    omega_out = complex_value(frozen["kinematics"]["each_outgoing_energy"])
    worldsheet_reduced = complex_value(frozen["reduced_moduli_integral"])
    worldsheet_amplitude = complex_value(
        frozen["worldsheet_amplitude_coefficient"]
    )
    matrix_reduced = (
        math.pi
        * omega
        * omega_out**3
        * (1.0 + 2j * omega)
    )
    matrix_amplitude = 8j * omega * omega_out**3 * (1.0 + 2j * omega)
    relative_difference = abs(worldsheet_reduced - matrix_reduced) / abs(
        matrix_reduced
    )
    worldsheet_relative_bound = frozen["blind_convergence"][
        "relative_linear_bound"
    ]
    payload = {
        "status": "postfreeze_matrix_comparison_complete",
        "verified_frozen_worldsheet_sha256": frozen_hash,
        "freeze_manifest_sha256": sha256(args.manifest),
        "worldsheet_was_frozen_before_comparison": True,
        "kinematics": frozen["kinematics"],
        "worldsheet": {
            "reduced_moduli_integral": complex_pair(worldsheet_reduced),
            "amplitude_coefficient": complex_pair(worldsheet_amplitude),
            "relative_internal_linear_bound": worldsheet_relative_bound,
        },
        "matrix_model": {
            "reduced_moduli_prediction": complex_pair(matrix_reduced),
            "amplitude_coefficient_prediction": complex_pair(matrix_amplitude),
        },
        "comparison": {
            "difference_reduced": complex_pair(
                worldsheet_reduced - matrix_reduced
            ),
            "relative_complex_difference": relative_difference,
            "relative_complex_difference_percent": 100.0 * relative_difference,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"post-freeze relative difference={100.0 * relative_difference:.6f}%"
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
