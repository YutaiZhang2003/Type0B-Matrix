#!/usr/bin/env python3
"""Merge direct low- and high-radius genus-two RQMC sweeps."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable, Sequence


def _load(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        rows = list(reader)
    if not fields or not rows:
        raise ValueError(f"empty radius table: {path}")
    return fields, rows


def merge_rows(
    low_rows: Sequence[dict[str, str]],
    high_rows: Sequence[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    """Return one reciprocal grid with a unique self-dual row and provenance."""

    if any(float(row["radius"]) > 1.0 + 1.0e-13 for row in low_rows):
        raise ValueError("the low-radius table contains R>1")
    if any(float(row["radius"]) < 1.0 - 1.0e-13 for row in high_rows):
        raise ValueError("the high-radius table contains R<1")
    low_one = [row for row in low_rows if math.isclose(float(row["radius"]), 1.0)]
    high_one = [row for row in high_rows if math.isclose(float(row["radius"]), 1.0)]
    if len(low_one) != 1 or len(high_one) != 1:
        raise ValueError("each input must contain exactly one self-dual row")
    for field in (
        "free_energy_over_gs_squared",
        "rqmc_scramble_standard_error",
        "normalized_worldsheet_shape",
        "normalized_worldsheet_shape_jackknife_se",
    ):
        if not math.isclose(
            float(low_one[0][field]),
            float(high_one[0][field]),
            rel_tol=2.0e-14,
            abs_tol=2.0e-14,
        ):
            raise ValueError(f"the two R=1 rows disagree in {field}")

    merged = [
        dict(row)
        for row in low_rows
        if float(row["radius"]) < 1.0 - 1.0e-13
    ] + [dict(row) for row in high_rows]
    merged.sort(key=lambda row: float(row["radius"]))
    if len(merged) != len(low_rows) + len(high_rows) - 1:
        raise AssertionError("the merged reciprocal grid has the wrong size")
    if len({float(row["radius"]) for row in merged}) != len(merged):
        raise AssertionError("the merged reciprocal grid contains duplicate radii")

    provenance: list[dict[str, object]] = []
    for row in merged:
        radius = float(row["radius"])
        provenance.append(
            {
                "radius": radius,
                "source": (
                    "direct-low-radius-compact-theta-reweight"
                    if radius < 1.0 - 1.0e-13
                    else (
                        "shared-self-dual-base"
                        if math.isclose(radius, 1.0, abs_tol=1.0e-13)
                        else "previous-direct-high-radius-sweep"
                    )
                ),
            }
        )
    return merged, provenance


def run(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("low_radius_csv", type=Path)
    parser.add_argument("high_radius_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)

    low_fields, low_rows = _load(args.low_radius_csv)
    high_fields, high_rows = _load(args.high_radius_csv)
    if low_fields != high_fields:
        raise ValueError("the low- and high-radius tables have different schemas")
    merged, provenance = merge_rows(low_rows, high_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.output_dir / "radius_sweep_R05_R2_39_direct.csv"
    provenance_path = args.output_dir / "radius_provenance.csv"
    with data_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=low_fields)
        writer.writeheader()
        writer.writerows(merged)
    with provenance_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("radius", "source"))
        writer.writeheader()
        writer.writerows(provenance)

    summary = {
        "scope": (
            "Direct R<1 compact-theta reweighting merged with the previous direct "
            "R>=1 genus-two free-energy sweep on the same 6,846-node RQMC sample."
        ),
        "low_radius_csv": str(args.low_radius_csv),
        "high_radius_csv": str(args.high_radius_csv),
        "low_radius_count_including_R1": len(low_rows),
        "high_radius_count_including_R1": len(high_rows),
        "combined_radius_count": len(merged),
        "radius_interval": [float(merged[0]["radius"]), float(merged[-1]["radius"])],
        "self_dual_row_policy": "retain the previous high-radius table's R=1 row after equality check",
        "data_csv": data_path.name,
        "provenance_csv": provenance_path.name,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"merged {len(low_rows)} low/self-dual and {len(high_rows)} high/self-dual "
        f"rows into {len(merged)} unique radii"
    )


if __name__ == "__main__":
    run()
