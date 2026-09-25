#!/usr/bin/env python3
"""Build a compact, broad validation manifest for the singlet formulas.

The selected kinematics include the largest structured-scan discrepancies and
energy-quantile representatives of the core, structured, soft, and near-real
families.  Every kinematic point is repeated for each numerical profile in
``spin23_formula_validation_settings.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from spin23_genuine_formulas import (
    s_to_sss_raw_candidate,
    s_to_svv_raw_candidate,
)


PROFILES = (
    "q7_p2p5_l20",
    "q6_p2p5_l20",
    "q6_p2p5_l24",
    "q6_p4_l20",
    "q6_p6_l20",
    "q6_p6_l24",
    "q6_p6_l28",
    "q6_p6_l24_rsmall",
    "q6_p6_l24_rlarge",
    "q6_p6_l24_dense",
)
FAMILIES = ("core", "structured", "soft", "near-real")
MANIFEST_COLUMNS = (
    "point_id",
    "family",
    "setting",
    "parent_id",
    "permutation",
    "note",
    "omega1_re",
    "omega1_im",
    "omega2_re",
    "omega2_im",
    "omega3_re",
    "omega3_im",
    "omega_re",
    "omega_im",
    "pole_margin",
    "max_pair_re_square",
)
_TINY = np.finfo(float).tiny


def _complex(row: Mapping[str, str], stem: str) -> complex:
    return complex(float(row[f"{stem}_re"]), float(row[f"{stem}_im"]))


def _load_primary_rows(results_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(results_dir.glob("result_*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            row = next(csv.DictReader(handle))
        if row.get("status") == "ok" and row.get("family") in FAMILIES:
            rows.append(row)
    if not rows:
        raise ValueError(f"no successful primary rows found in {results_dir}")
    return rows


def _formula_residuals(row: Mapping[str, str]) -> tuple[float, float]:
    omega = tuple(
        complex(float(row[f"omega{leg}_re"]), float(row[f"omega{leg}_im"]))
        for leg in (1, 2, 3)
    )
    numerical_svv = _complex(row, "ssvv_raw")
    numerical_sss = _complex(row, "ssss_raw")
    candidate_svv = s_to_svv_raw_candidate(*omega)
    candidate_sss = s_to_sss_raw_candidate(*omega)
    return (
        abs(numerical_svv - candidate_svv) / max(abs(numerical_svv), _TINY),
        abs(numerical_sss - candidate_sss) / max(abs(numerical_sss), _TINY),
    )


def _energy(row: Mapping[str, str]) -> float:
    return abs(complex(float(row["omega_re"]), float(row["omega_im"])))


def _quantile_rows(
    rows: Sequence[dict[str, str]],
    count: int,
) -> list[dict[str, str]]:
    ordered = sorted(rows, key=lambda row: (_energy(row), row["point_id"]))
    indices = np.linspace(0, len(ordered) - 1, count).round().astype(int)
    return [ordered[index] for index in indices]


def select_rows(
    rows: Sequence[dict[str, str]],
    *,
    target_count: int,
) -> list[dict[str, str]]:
    """Select broad kinematics while prioritizing structured outliers."""

    residuals = {row["point_id"]: _formula_residuals(row) for row in rows}
    selected: dict[str, dict[str, str]] = {}

    structured = [row for row in rows if row["family"] == "structured"]
    for process_index in (0, 1):
        ranked = sorted(
            structured,
            key=lambda row: residuals[row["point_id"]][process_index],
            reverse=True,
        )
        for row in ranked[:2]:
            selected[row["point_id"]] = row

    family_quantile_counts = {
        "core": 4,
        "structured": 4,
        "soft": 3,
        "near-real": 3,
    }
    for family, count in family_quantile_counts.items():
        family_rows = [row for row in rows if row["family"] == family]
        for row in _quantile_rows(family_rows, count):
            selected[row["point_id"]] = row

    ranked_all = sorted(
        rows,
        key=lambda row: max(residuals[row["point_id"]]),
        reverse=True,
    )
    for row in ranked_all:
        if len(selected) >= target_count:
            break
        selected[row["point_id"]] = row

    if len(selected) < target_count:
        raise ValueError(
            f"only {len(selected)} distinct rows were available for {target_count} targets"
        )
    return sorted(selected.values(), key=lambda row: row["point_id"])[:target_count]


def build_manifest(
    selected: Sequence[Mapping[str, str]],
    output: Path,
    *,
    profiles: Sequence[str] = PROFILES,
) -> list[dict[str, str]]:
    profiles = tuple(profiles)
    if not profiles:
        raise ValueError("at least one numerical profile is required")
    if len(set(profiles)) != len(profiles):
        raise ValueError("numerical profile names must be distinct")
    manifest: list[dict[str, str]] = []
    for point_index, source in enumerate(selected):
        for profile_index, profile in enumerate(profiles):
            row = {column: source.get(column, "") for column in MANIFEST_COLUMNS}
            row.update(
                {
                    "point_id": f"formula-validation-{point_index:02d}-{profile_index:02d}",
                    "family": "formula-validation",
                    "setting": profile,
                    "parent_id": source["point_id"],
                    "permutation": "012",
                    "note": (
                        f"source_family={source['family']}; "
                        f"source_id={source['point_id']}; profile={profile}"
                    ),
                }
            )
            manifest.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(manifest)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--points", type=int, default=20)
    parser.add_argument(
        "--profiles",
        nargs="+",
        help="profile names to repeat at every selected kinematic point",
    )
    parser.add_argument("--summary", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    rows = _load_primary_rows(args.results_dir)
    selected = select_rows(rows, target_count=args.points)
    profiles = tuple(args.profiles) if args.profiles else PROFILES
    manifest = build_manifest(selected, args.output, profiles=profiles)
    summary = {
        "manifest": str(args.output),
        "manifest_rows": len(manifest),
        "profiles": list(profiles),
        "selected_points": len(selected),
        "source_family_counts": dict(Counter(row["family"] for row in selected)),
        "source_point_ids": [row["point_id"] for row in selected],
    }
    summary_path = args.summary or args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
