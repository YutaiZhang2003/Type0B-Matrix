#!/usr/bin/env python3
"""Extend an R>=1 genus-two radius table to R<1 by exact T-duality."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Sequence


NUMERIC_FIELDS = (
    "radius",
    "free_energy_over_gs_squared",
    "rqmc_scramble_standard_error",
    "volume_calibrated_free_energy_over_gs_squared",
    "volume_calibrated_scramble_standard_error",
    "normalized_worldsheet_shape",
    "normalized_worldsheet_shape_jackknife_se",
    "contribution_effective_sample_size",
    "largest_node_fraction",
)
SCALED_FIELDS = (
    "free_energy_over_gs_squared",
    "rqmc_scramble_standard_error",
    "volume_calibrated_free_energy_over_gs_squared",
    "volume_calibrated_scramble_standard_error",
    "normalized_worldsheet_shape",
    "normalized_worldsheet_shape_jackknife_se",
)


def _load_rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != NUMERIC_FIELDS:
            raise ValueError("the input radius table does not have the production schema")
        rows = [
            {key: float(value) for key, value in row.items()}
            for row in reader
        ]
    rows.sort(key=lambda row: row["radius"])
    if len(rows) < 2:
        raise ValueError("need at least R=1 and one R>1 point")
    if not math.isclose(rows[0]["radius"], 1.0, rel_tol=0.0, abs_tol=1.0e-13):
        raise ValueError("the source table must begin at the self-dual radius R=1")
    if any(row["radius"] < 1.0 - 1.0e-13 for row in rows):
        raise ValueError("the source table already contains R<1 points")
    if len({row["radius"] for row in rows}) != len(rows):
        raise ValueError("the source radii are not unique")
    return rows


def extend_rows(
    rows: Sequence[dict[str, float]],
) -> tuple[list[dict[str, float]], list[dict[str, object]]]:
    """Return the reciprocal grid and an explicit transformation ledger.

    In the target-space zero-mode convention used by the production integrand,

        F2(r) = F2(1/r) / r^2,  0 < r < 1.

    The same positive scale multiplies every scramble, so the absolute and
    paired-jackknife errors scale identically.  Contribution ESS and the
    largest-node fraction are invariant under this common rescaling.
    """

    output = [dict(row) for row in rows]
    provenance: list[dict[str, object]] = []
    for row in rows:
        radius = row["radius"]
        provenance.append(
            {
                "radius": radius,
                "source": "computed-self-dual" if math.isclose(radius, 1.0) else "computed-R>=1",
                "dual_partner_radius": 1.0 / radius,
                "multiplicative_scale_from_partner": 1.0,
            }
        )
        if math.isclose(radius, 1.0, rel_tol=0.0, abs_tol=1.0e-13):
            continue
        reciprocal = 1.0 / radius
        factor = 1.0 / (reciprocal * reciprocal)
        dual = dict(row)
        dual["radius"] = reciprocal
        for field in SCALED_FIELDS:
            dual[field] = row[field] * factor
        output.append(dual)
        provenance.append(
            {
                "radius": reciprocal,
                "source": "exact-T-duality",
                "dual_partner_radius": radius,
                "multiplicative_scale_from_partner": factor,
            }
        )

    output.sort(key=lambda row: row["radius"])
    provenance.sort(key=lambda row: float(row["radius"]))
    if len(output) != 2 * len(rows) - 1:
        raise AssertionError("reciprocal extension has the wrong number of rows")
    return output, provenance


def _write_numeric_csv(path: Path, rows: Sequence[dict[str, float]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=NUMERIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_provenance_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    fieldnames = (
        "radius",
        "source",
        "dual_partner_radius",
        "multiplicative_scale_from_partner",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _maximum_duality_residual(rows: Sequence[dict[str, float]]) -> float:
    residual = 0.0
    for row in rows:
        radius = row["radius"]
        inverse = min(rows, key=lambda candidate: abs(candidate["radius"] - 1.0 / radius))
        if not math.isclose(
            inverse["radius"],
            1.0 / radius,
            rel_tol=2.0e-13,
            abs_tol=2.0e-13,
        ):
            raise AssertionError(f"radius {radius} lacks its reciprocal partner")
        target = inverse["free_energy_over_gs_squared"] / (radius * radius)
        residual = max(
            residual,
            abs(row["free_energy_over_gs_squared"] / target - 1.0),
        )
    return residual


def run(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)

    source_rows = _load_rows(args.input_csv)
    rows, provenance = extend_rows(source_rows)
    residual = _maximum_duality_residual(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.output_dir / "radius_sweep_R05_R2_39.csv"
    provenance_path = args.output_dir / "radius_provenance.csv"
    summary_path = args.output_dir / "summary.json"
    _write_numeric_csv(data_path, rows)
    _write_provenance_csv(provenance_path, provenance)

    source_hash = hashlib.sha256(args.input_csv.read_bytes()).hexdigest()
    summary = {
        "scope": (
            "The verified R>=1 genus-two free-energy sweep extended to R<1 "
            "without new Monte Carlo evaluations."
        ),
        "input_csv": str(args.input_csv),
        "input_sha256": source_hash,
        "source_radius_count": len(source_rows),
        "derived_radius_count": len(rows) - len(source_rows),
        "combined_radius_count": len(rows),
        "radius_interval": [rows[0]["radius"], rows[-1]["radius"]],
        "t_duality_convention": "F2(r)/gs^2 = r^-2 F2(1/r)/gs^2 for 0<r<1",
        "uncertainty_rule": (
            "absolute RQMC and paired-shape uncertainties receive the same "
            "r^-2 factor; ESS and largest-node fraction are unchanged"
        ),
        "maximum_integrated_t_duality_relative_residual": residual,
        "data_csv": data_path.name,
        "provenance_csv": provenance_path.name,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"wrote {len(rows)} radii from {rows[0]['radius']:.6g} to "
        f"{rows[-1]['radius']:.6g}; duality residual={residual:.3e}"
    )


if __name__ == "__main__":
    run()
