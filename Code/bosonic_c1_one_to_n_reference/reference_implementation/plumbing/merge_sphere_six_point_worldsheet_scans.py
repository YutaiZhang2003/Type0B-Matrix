#!/usr/bin/env python3
"""Merge and freeze compatible worldsheet-only sphere 1->5 scans."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path


WORLDSHEET_STATUS = "worldsheet_only_no_matrix_model_imported"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    arguments = parser.parse_args()

    base = json.loads(arguments.base.read_text())
    extension = json.loads(arguments.extension.read_text())
    for name, payload in (("base", base), ("extension", extension)):
        if payload.get("status") != WORLDSHEET_STATUS:
            raise RuntimeError(f"{name} scan is not a completed worldsheet-only artifact")
        if not bool(payload["kinematic_domain"]["all_points_below_wall"]):
            raise RuntimeError(f"{name} scan leaves the residue-free chamber")
    if base["settings"] != extension["settings"]:
        raise RuntimeError("base and extension numerical settings differ")
    if base["normalization"] != extension["normalization"]:
        raise RuntimeError("base and extension normalization records differ")
    if base["kinematic_domain"] != extension["kinematic_domain"]:
        raise RuntimeError("base and extension kinematic chambers differ")

    points = [*base["points"], *extension["points"]]
    t_values = [float(point["t"]) for point in points]
    if len(points) != 16 or len(set(t_values)) != 16:
        raise RuntimeError("the merged scan must contain 16 distinct points")
    first_wall = float(base["kinematic_domain"]["first_residue_wall"])
    if any(not 0.0 < value < first_wall for value in t_values):
        raise RuntimeError("a merged point is outside 0<t<1/3")
    if any(
        sum(int(value) for value in point["block_fallback_counts"].values()) != 0
        for point in points
    ):
        raise RuntimeError("a merged point used a block fallback")
    points.sort(key=lambda point: float(point["t"]))

    merged = {
        "status": WORLDSHEET_STATUS,
        "normalization": base["normalization"],
        "kinematic_domain": base["kinematic_domain"],
        "settings": base["settings"],
        "provenance": {
            "base_artifact": str(arguments.base),
            "base_sha256": sha256_file(arguments.base),
            "base_point_count": len(base["points"]),
            "extension_artifact": str(arguments.extension),
            "extension_sha256": sha256_file(arguments.extension),
            "extension_point_count": len(extension["points"]),
            "matrix_model_information_used": False,
        },
        "points": points,
    }
    write_json(arguments.output, merged)
    manifest = {
        "artifact": arguments.output.name,
        "sha256": sha256_file(arguments.output),
        "frozen_on": date.today().isoformat(),
        "status": "worldsheet_only_frozen_before_matrix_model_comparison",
        "kinematic_chamber": "omega=i t with 0<t<1/3",
        "point_count": 16,
        "points": [float(point["t"]) for point in points],
        "settings": merged["settings"],
        "matrix_model_information_used": False,
    }
    write_json(arguments.freeze_manifest, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
