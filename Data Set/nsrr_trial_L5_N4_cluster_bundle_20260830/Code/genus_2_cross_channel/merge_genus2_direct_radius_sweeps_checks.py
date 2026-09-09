#!/usr/bin/env python3
"""Checks for merging the two direct genus-two radius sweeps."""

from __future__ import annotations

try:
    from merge_genus2_direct_radius_sweeps import merge_rows
except ImportError:  # pragma: no cover
    from plumbing.merge_genus2_direct_radius_sweeps import merge_rows


def _row(radius: float, value: float) -> dict[str, str]:
    return {
        "radius": str(radius),
        "free_energy_over_gs_squared": str(value),
        "rqmc_scramble_standard_error": "0.2",
        "normalized_worldsheet_shape": str(value / 3.0),
        "normalized_worldsheet_shape_jackknife_se": "0.1",
    }


def run_checks() -> None:
    merged, provenance = merge_rows(
        [_row(0.5, 12.0), _row(1.0, 3.0)],
        [_row(1.0, 3.0), _row(2.0, 3.0)],
    )
    if [float(row["radius"]) for row in merged] != [0.5, 1.0, 2.0]:
        raise AssertionError("the direct radius grids were not merged correctly")
    if provenance[0]["source"] != "direct-low-radius-compact-theta-reweight":
        raise AssertionError("the low-radius provenance is missing")
    if provenance[1]["source"] != "shared-self-dual-base":
        raise AssertionError("the R=1 provenance is missing")
    print("merge_genus2_direct_radius_sweeps checks passed")


if __name__ == "__main__":
    run_checks()
