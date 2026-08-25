#!/usr/bin/env python3
"""Produce and freeze the twelve-point sphere ``1->4`` worldsheet extension.

This program is intentionally target-free: it imports no matrix-model formula
or coefficient.  It retains the numerical prescription of the existing
imaginary-ray campaign, assigns a disjoint Sobol seed block to every new
point, and checkpoints after each completed integral so a long run can be
resumed safely.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import date
from pathlib import Path
from typing import Any

from sphere_five_point_extended_scan import scan_point


STATUS = "sphere_1to4_worldsheet_extension_frozen_before_external_comparison"
BASE_SEED = 20277001
SEED_STRIDE = 8

# The historical table already resolves the exact zero with t=0.249 and
# t=0.251.  The new points interlace the remaining broad gaps, sample both
# sides of the first contour wall, and stay away from the second wall at 1/2.
DESIGN: tuple[tuple[float, int, int, str], ...] = (
    (0.19, 6, 10, "real"),
    (0.21, 6, 10, "real"),
    (0.23, 6, 10, "real"),
    (0.27, 6, 10, "real"),
    (0.29, 6, 10, "real"),
    (0.31, 6, 10, "real"),
    (0.33, 6, 10, "real"),
    (0.35, 8, 9, "continued"),
    (0.37, 8, 9, "continued"),
    (0.39, 8, 9, "continued"),
    (0.44, 8, 9, "continued"),
    (0.47, 8, 9, "continued"),
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _settings() -> dict[str, Any]:
    return {
        "momentum_order": 20,
        "momentum_maximum": 6.0,
        "momentum_panels": 1,
        "momentum_power": 1.25,
        "block_scheme": "c",
        "replicates": 4,
        "radial_power": 0.15,
        "base_seed": BASE_SEED,
        "per_point_seed_stride": SEED_STRIDE,
        "first_contour_wall": 0.4,
        "second_contour_wall": 0.5,
    }


def _empty_payload() -> dict[str, Any]:
    return {
        "status": "sphere_1to4_worldsheet_extension_in_progress",
        "calculation": "genus-zero equal-energy sphere 1->4 amplitude",
        "matrix_model_information_used": False,
        "kinematics": "omega=i*t; signed energies (4*omega,-omega,-omega,-omega,-omega)",
        "normalization": "Q_4=I_5/(16*pi^2*omega^5); mu^3*A=4*i*omega^5*Q_4",
        "moduli_prescription": "120-chart plumbing-coordinate mixture",
        "liouville_contour": (
            "real below t=0.35; continued implementation from t=0.35, "
            "with the first crossed residue active only for t>0.4"
        ),
        "settings": _settings(),
        "design": {
            "target_merged_point_count": 30,
            "historical_point_count": 18,
            "extension_point_count": len(DESIGN),
            "extension_t_values": [entry[0] for entry in DESIGN],
            "selection": (
                "interlace broad historical gaps across the validated ray; "
                "retain the existing t=0.249 and 0.251 zero probes; avoid "
                "t=0.25 and both contour walls"
            ),
        },
        "points": [],
    }


def _load_resume(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _empty_payload()
    payload = json.loads(path.read_text())
    if payload.get("matrix_model_information_used") is not False:
        raise RuntimeError("resume artifact does not certify target-free production")
    if payload.get("settings") != _settings():
        raise RuntimeError("resume artifact settings differ from the frozen design")
    if payload.get("design", {}).get("extension_t_values") != [x[0] for x in DESIGN]:
        raise RuntimeError("resume artifact has a different extension design")
    return payload


def _flat_record(
    raw: dict[str, Any],
    *,
    block_order: int,
    sobol_power: int,
    contour: str,
    seed: int,
) -> dict[str, Any]:
    t_value = float(raw["t"])
    q = raw["Q"]
    sigma = raw["Q_standard_error"]
    q_real = float(q["real"])
    q_imag = float(q["imag"])
    sigma_real = float(sigma["real"])
    if not all(math.isfinite(value) for value in (q_real, q_imag, sigma_real)):
        raise ArithmeticError("non-finite worldsheet output")
    if sigma_real <= 0.0:
        raise ArithmeticError("non-positive worldsheet standard error")
    if contour == "real":
        residue_status = "not applicable"
    elif t_value < 0.4:
        residue_status = "inactive below t=2/5"
    else:
        residue_status = "included"
    return {
        "t": t_value,
        "Q": q_real,
        "Q_imaginary_part": q_imag,
        "Q_standard_error": sigma_real,
        "raw_integral": raw["raw_integral"],
        "raw_standard_error": raw["raw_standard_error"],
        "replicate_estimates": raw["replicate_estimates"],
        "block_order": block_order,
        "momentum_order": 20,
        "sobol_power": sobol_power,
        "replicates": 4,
        "contour": contour,
        "residue_status": residue_status,
        "seed": seed,
    }


def run(output_path: Path, manifest_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _load_resume(output_path)
    completed = {round(float(point["t"]), 12): point for point in payload["points"]}
    for index, (t_value, block_order, sobol_power, contour) in enumerate(DESIGN):
        key = round(t_value, 12)
        if key in completed:
            print(f"skipping completed omega=i*{t_value:.6g}", flush=True)
            continue
        seed = BASE_SEED + SEED_STRIDE * index
        print(
            f"evaluating omega=i*{t_value:.6g}, block={block_order}, "
            f"sobol=2^{sobol_power}, seed={seed}",
            flush=True,
        )
        raw = scan_point(
            t_value,
            block_order=block_order,
            momentum_order=20,
            momentum_maximum=6.0,
            momentum_panels=1,
            momentum_power=1.25,
            block_scheme="c",
            liouville_contour=contour,
            sobol_power=sobol_power,
            replicates=4,
            radial_power=0.15,
            seed=seed,
        )
        completed[key] = _flat_record(
            raw,
            block_order=block_order,
            sobol_power=sobol_power,
            contour=contour,
            seed=seed,
        )
        payload["points"] = [completed[value] for value in sorted(completed)]
        write_json(output_path, payload)

    expected = [round(entry[0], 12) for entry in DESIGN]
    actual = [round(float(point["t"]), 12) for point in payload["points"]]
    if actual != expected:
        raise RuntimeError(f"extension is incomplete: found {actual}, expected {expected}")
    payload["status"] = STATUS
    payload["point_count"] = len(payload["points"])
    write_json(output_path, payload)
    manifest = {
        "status": STATUS,
        "artifact": str(output_path.resolve()),
        "sha256": sha256_file(output_path),
        "frozen_on": date.today().isoformat(),
        "point_count": len(payload["points"]),
        "t_values": [point["t"] for point in payload["points"]],
        "matrix_model_information_used": False,
    }
    write_json(manifest_path, manifest)
    return payload, manifest


def main() -> None:
    run_dir = (
        Path(__file__).parent
        / "results"
        / "sphere_five_point_1to4"
        / "blind30_20260824"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=run_dir / "worldsheet_extension_12point.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=run_dir / "worldsheet_extension_12point_frozen.json",
    )
    arguments = parser.parse_args()
    _, manifest = run(arguments.output, arguments.manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
