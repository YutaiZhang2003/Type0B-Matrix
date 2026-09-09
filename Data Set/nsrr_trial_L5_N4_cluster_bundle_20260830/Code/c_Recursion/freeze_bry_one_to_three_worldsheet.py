#!/usr/bin/env python3
"""Validate and freeze the target-free BRY sphere-four-point worldsheet run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DEFAULT_PRIMARY = RESULTS / "bry_one_to_three_high_accuracy_primary.json"
DEFAULT_P_CONTROL = RESULTS / "bry_one_to_three_high_accuracy_p_order24.json"
DEFAULT_TAIL = RESULTS / "bry_one_to_three_high_accuracy_tail_5_6.json"
DEFAULT_FROZEN = RESULTS / "bry_one_to_three_worldsheet_frozen.json"
DEFAULT_MANIFEST = RESULTS / "bry_one_to_three_worldsheet_freeze_manifest.json"

AXIS_TOLERANCES = {
    "block_order": 7.5e-4,
    "cap_radius": 1.0e-3,
    "moduli_order": 2.0e-3,
    "momentum_order": 1.0e-4,
    "momentum_tail": 1.0e-8,
}
TOTAL_LINEAR_TOLERANCE = 3.0e-3


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def complex_value(pair) -> complex:
    return complex(pair["real"], pair["imag"])


def complex_pair(value: complex) -> dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag)}


def load_worldsheet(path: Path, expected_campaign: str):
    data = json.loads(path.read_text())
    if data.get("status") != "worldsheet_only_unfrozen":
        raise ValueError(f"{path} is not an unfrozen worldsheet result")
    if data.get("campaign") != expected_campaign:
        raise ValueError(f"{path} has the wrong campaign label")
    if data.get("comparison_performed") is not False:
        raise ValueError(f"{path} was already compared")
    if data.get("matrix_model_data_included") is not False:
        raise ValueError(f"{path} contains matrix-model data")
    if data.get("recursion_backend") != "h":
        raise ValueError(f"{path} is not an all-h worldsheet computation")
    return data


def variant_map(data):
    return {item["name"]: item for item in data["variants"]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, default=DEFAULT_PRIMARY)
    parser.add_argument("--p-control", type=Path, default=DEFAULT_P_CONTROL)
    parser.add_argument("--tail", type=Path, default=DEFAULT_TAIL)
    parser.add_argument("--frozen", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    primary = load_worldsheet(args.primary, "primary")
    p_control = load_worldsheet(args.p_control, "p-order-control")
    tail = load_worldsheet(args.tail, "tail-control")
    variants = variant_map(primary)
    required = {
        "candidate_q12_eps005_z20",
        "block_control_q10_eps005_z20",
        "cap_control_q12_eps0075_z20",
        "moduli_control_q12_eps005_z18",
    }
    if set(variants) != required:
        raise ValueError("primary campaign does not have the frozen control set")

    candidate_record = variants["candidate_q12_eps005_z20"]
    candidate = complex_value(candidate_record["reduced_moduli_integral"])
    scale = abs(candidate)
    if not math.isfinite(scale) or scale == 0.0:
        raise ValueError("candidate worldsheet value is not finite and nonzero")

    controls = {
        "block_order": complex_value(
            variants["block_control_q10_eps005_z20"]["reduced_moduli_integral"]
        ),
        "cap_radius": complex_value(
            variants["cap_control_q12_eps0075_z20"]["reduced_moduli_integral"]
        ),
        "moduli_order": complex_value(
            variants["moduli_control_q12_eps005_z18"]["reduced_moduli_integral"]
        ),
        "momentum_order": complex_value(
            p_control["variants"][0]["reduced_moduli_integral"]
        ),
    }
    tail_value = complex_value(tail["variants"][0]["reduced_moduli_integral"])
    relative_shifts = {
        name: abs(value - candidate) / scale for name, value in controls.items()
    }
    relative_shifts["momentum_tail"] = abs(tail_value) / scale
    passed_axes = {
        name: relative_shifts[name] <= tolerance
        for name, tolerance in AXIS_TOLERANCES.items()
    }
    total_linear = sum(relative_shifts.values())
    total_rss = math.sqrt(sum(value * value for value in relative_shifts.values()))
    passed = all(passed_axes.values()) and total_linear <= TOTAL_LINEAR_TOLERANCE
    if not passed:
        raise SystemExit("blind convergence gates failed; refusing to freeze")

    source_hashes = {
        "primary": sha256(args.primary),
        "p_order_control": sha256(args.p_control),
        "tail_control": sha256(args.tail),
    }
    frozen = {
        "status": "worldsheet_result_frozen",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "comparison_performed": False,
        "matrix_model_data_included": False,
        "source_worldsheet_sha256": source_hashes,
        "kinematics": primary["kinematics"],
        "recursion_backend": "h",
        "candidate_settings": {
            key: candidate_record[key]
            for key in (
                "q_order",
                "epsilon",
                "angular_order",
                "radial_order",
                "cap_angular_order",
                "cap_radial_order",
                "block_backend",
            )
        },
        "momentum_quadrature": primary["momentum_quadrature"],
        "precision": primary["precision"],
        "region_integrals": {
            key: candidate_record[key]
            for key in (
                "low_z_region_integral",
                "bulk_region_integral",
                "t_cap_region_integral",
            )
        },
        "reduced_moduli_integral": complex_pair(candidate),
        "worldsheet_amplitude_coefficient": candidate_record[
            "worldsheet_amplitude_coefficient"
        ],
        "blind_convergence": {
            "relative_axis_shifts": relative_shifts,
            "axis_tolerances": AXIS_TOLERANCES,
            "passed_axes": passed_axes,
            "relative_linear_bound": total_linear,
            "relative_rss_estimate": total_rss,
            "linear_tolerance": TOTAL_LINEAR_TOLERANCE,
            "absolute_linear_bound": total_linear * scale,
            "passed": True,
        },
    }
    args.frozen.parent.mkdir(parents=True, exist_ok=True)
    args.frozen.write_text(json.dumps(frozen, indent=2) + "\n")
    manifest = {
        "status": "worldsheet_freeze_complete",
        "frozen_worldsheet_file": str(args.frozen),
        "frozen_worldsheet_sha256": sha256(args.frozen),
        "source_worldsheet_sha256": source_hashes,
        "comparison_allowed": True,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"frozen M={candidate.real:+.14f}{candidate.imag:+.14f}i, "
        f"linear bound={total_linear:.6%}"
    )
    print(f"wrote {args.frozen}")
    print(f"wrote {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
