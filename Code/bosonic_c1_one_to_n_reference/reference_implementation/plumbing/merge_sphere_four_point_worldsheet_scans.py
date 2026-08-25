#!/usr/bin/env python3
"""Merge and freeze compatible worldsheet-only sphere ``1->3`` scans.

The merger deliberately contains no matrix-model formula.  Its only role is
to validate the old and extension scans, retain every distinct point, and
freeze the combined worldsheet artifact before fitting or comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import date
from pathlib import Path
from typing import Any


SOURCE_STATUS = "worldsheet_only_frozen_before_external_comparison"
MERGED_STATUS = "worldsheet_only_merged_and_frozen_before_external_comparison"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _verified_source(path: Path, manifest_path: Path) -> tuple[dict[str, Any], str]:
    payload = json.loads(path.read_text())
    manifest = json.loads(manifest_path.read_text())
    artifact_hash = sha256_file(path)
    if artifact_hash != manifest.get("sha256"):
        raise RuntimeError(f"source scan does not match its freeze manifest: {path}")
    if payload.get("status") != SOURCE_STATUS:
        raise RuntimeError(f"source scan is not a frozen worldsheet-only artifact: {path}")
    if payload.get("matrix_model_information_used") is not False:
        raise RuntimeError(f"source scan does not certify blind production: {path}")
    if int(manifest.get("point_count", -1)) != len(payload.get("points", [])):
        raise RuntimeError(f"source point count does not match its manifest: {path}")
    return payload, artifact_hash


def _numerical_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Return settings that must agree, excluding the randomization seed."""

    return {key: value for key, value in settings.items() if key != "base_seed"}


def merge_scans(
    base_path: Path,
    base_manifest_path: Path,
    extension_path: Path,
    extension_manifest_path: Path,
    output_path: Path,
    freeze_manifest_path: Path,
    *,
    expected_point_count: int = 30,
) -> tuple[dict[str, Any], dict[str, Any]]:
    base, base_hash = _verified_source(base_path, base_manifest_path)
    extension, extension_hash = _verified_source(extension_path, extension_manifest_path)

    compatible_metadata = (
        "calculation",
        "kinematics",
        "domain",
        "liouville_contour",
        "moduli_prescription",
        "normalization",
    )
    for key in compatible_metadata:
        if base.get(key) != extension.get(key):
            raise RuntimeError(f"base and extension disagree in {key}")
    if _numerical_settings(base["settings"]) != _numerical_settings(extension["settings"]):
        raise RuntimeError("base and extension numerical settings differ")
    base_seed = int(base["settings"]["base_seed"])
    extension_seed = int(extension["settings"]["base_seed"])
    expected_extension_seed = base_seed + 1009 * len(base["points"])
    if extension_seed != expected_extension_seed:
        raise RuntimeError(
            "extension seed stream is not the disjoint continuation of the base scan"
        )

    base_t = [float(point["t"]) for point in base["points"]]
    extension_t = [float(point["t"]) for point in extension["points"]]
    if base_t != sorted(base_t) or extension_t != sorted(extension_t):
        raise RuntimeError("source t values must be increasing")
    if len(extension_t) > len(base_t) - 1:
        raise RuntimeError("the interlaced extension has too many points")
    expected_extension_t = [
        0.5 * (base_t[index] + base_t[index + 1])
        for index in range(len(extension_t))
    ]
    if any(
        not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12)
        for actual, expected in zip(extension_t, expected_extension_t, strict=True)
    ):
        raise RuntimeError("extension points do not follow the declared interlaced design")

    points = [
        *({**point, "scan_cohort": "known_base"} for point in base["points"]),
        *(
            {**point, "scan_cohort": "new_extension"}
            for point in extension["points"]
        ),
    ]
    t_values = [float(point["t"]) for point in points]
    if len(points) != expected_point_count:
        raise RuntimeError(
            f"merged scan has {len(points)} points; expected {expected_point_count}"
        )
    if len(set(t_values)) != expected_point_count:
        raise RuntimeError("base and extension scans contain duplicate t values")
    first_wall = float(base["liouville_contour"]["first_residue_wall"])
    if any(not 0.0 < value < first_wall for value in t_values):
        raise RuntimeError("a merged point lies outside the residue-free chamber")
    points.sort(key=lambda point: float(point["t"]))

    merged: dict[str, Any] = {
        "status": MERGED_STATUS,
        "calculation": base["calculation"],
        "matrix_model_information_used": False,
        "kinematics": base["kinematics"],
        "domain": base["domain"],
        "liouville_contour": base["liouville_contour"],
        "moduli_prescription": base["moduli_prescription"],
        "normalization": base["normalization"],
        "settings": {
            **_numerical_settings(base["settings"]),
            "seed_streams": {
                "base": base_seed,
                "extension": extension_seed,
                "per_point_increment": 1009,
            },
        },
        "design": {
            "point_count": expected_point_count,
            "base_point_count": len(base["points"]),
            "extension_point_count": len(extension["points"]),
            "extension_rule": (
                "interlace the first fourteen adjacent base-grid intervals; "
                "omit the unused midpoint nearest the first residue wall"
            ),
            "t_values": [float(point["t"]) for point in points],
        },
        "provenance": {
            "base_artifact": str(base_path.resolve()),
            "base_manifest": str(base_manifest_path.resolve()),
            "base_sha256": base_hash,
            "extension_artifact": str(extension_path.resolve()),
            "extension_manifest": str(extension_manifest_path.resolve()),
            "extension_sha256": extension_hash,
            "matrix_model_information_used": False,
        },
        "points": points,
    }
    write_json(output_path, merged)
    manifest: dict[str, Any] = {
        "status": MERGED_STATUS,
        "artifact": str(output_path.resolve()),
        "sha256": sha256_file(output_path),
        "frozen_on": date.today().isoformat(),
        "point_count": expected_point_count,
        "t_values": merged["design"]["t_values"],
        "matrix_model_information_used": False,
    }
    write_json(freeze_manifest_path, manifest)
    return merged, manifest


def main() -> None:
    base_dir = Path(__file__).parent / "results" / "sphere_four_point_1to3"
    run_dir = base_dir / "blind30_20260824"
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=base_dir / "worldsheet_scan.json")
    parser.add_argument(
        "--base-manifest", type=Path, default=base_dir / "worldsheet_scan_frozen.json"
    )
    parser.add_argument(
        "--extension", type=Path, default=run_dir / "worldsheet_extension_14point.json"
    )
    parser.add_argument(
        "--extension-manifest",
        type=Path,
        default=run_dir / "worldsheet_extension_14point_frozen.json",
    )
    parser.add_argument("--output", type=Path, default=run_dir / "worldsheet_scan_30point.json")
    parser.add_argument(
        "--freeze-manifest",
        type=Path,
        default=run_dir / "worldsheet_scan_30point_frozen.json",
    )
    parser.add_argument("--expected-point-count", type=int, default=30)
    arguments = parser.parse_args()
    _, manifest = merge_scans(
        arguments.base,
        arguments.base_manifest,
        arguments.extension,
        arguments.extension_manifest,
        arguments.output,
        arguments.freeze_manifest,
        expected_point_count=arguments.expected_point_count,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
