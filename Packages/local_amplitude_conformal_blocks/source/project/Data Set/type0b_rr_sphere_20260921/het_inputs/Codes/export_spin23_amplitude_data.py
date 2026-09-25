#!/usr/bin/env python3
"""Export compact, documented tables from the completed Spin(23) scans."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence


CODE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = CODE_DIR.parent
DATA_DIR = WORKSPACE_DIR / "Data Set"
DEFAULT_VVVV_SOURCE = (
    DATA_DIR
    / "results"
    / "spin23_vvvv_stable_all_6888_optimized_v1"
    / "merged.csv"
)
DEFAULT_SINGLET_SOURCE = (
    DATA_DIR
    / "results"
    / "spin23_singlet_genuine_all_6888_q7_lens_v1"
    / "merged.csv"
)
DEFAULT_OUTPUT_DIR = DATA_DIR / "data_exports"

ENERGY_SOURCE_FIELDS = (
    "omega1_re",
    "omega1_im",
    "omega2_re",
    "omega2_im",
    "omega3_re",
    "omega3_im",
    "omega_re",
    "omega_im",
)
COMMON_OUTPUT_FIELDS = (
    "point_id",
    "family",
    "setting",
    "omega0_re",
    "omega0_im",
    "omega1_re",
    "omega1_im",
    "omega2_re",
    "omega2_im",
    "omega3_re",
    "omega3_im",
    "q_order",
    "lower_q_order",
)


def _read_primary_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"no rows found in {path}")

    primary = [row for row in rows if row.get("family") != "convergence"]
    if len(primary) != 6376:
        raise ValueError(
            f"expected 6376 primary rows in {path}, found {len(primary)}"
        )
    point_ids = [row["point_id"] for row in primary]
    if len(point_ids) != len(set(point_ids)):
        raise ValueError(f"duplicate primary point_id values in {path}")
    if any(row.get("status") != "ok" for row in primary):
        raise ValueError(f"non-successful primary evaluation found in {path}")

    for row in primary:
        for field in ENERGY_SOURCE_FIELDS:
            value = float(row[field])
            if not math.isfinite(value):
                raise ValueError(f"non-finite {field} at {row['point_id']}")
        outgoing = sum(
            complex(float(row[f"omega{i}_re"]), float(row[f"omega{i}_im"]))
            for i in range(1, 4)
        )
        incoming = complex(float(row["omega_re"]), float(row["omega_im"]))
        if abs(incoming - outgoing) > 2.0e-12 * max(1.0, abs(incoming)):
            raise ValueError(f"energy conservation failed at {row['point_id']}")
    return sorted(primary, key=lambda row: row["point_id"])


def _common_values(row: Mapping[str, str], lower_q_order: str) -> dict[str, str]:
    return {
        "point_id": row["point_id"],
        "family": row["family"],
        "setting": row["setting"],
        "omega0_re": row["omega_re"],
        "omega0_im": row["omega_im"],
        "omega1_re": row["omega1_re"],
        "omega1_im": row["omega1_im"],
        "omega2_re": row["omega2_re"],
        "omega2_im": row["omega2_im"],
        "omega3_re": row["omega3_re"],
        "omega3_im": row["omega3_im"],
        "q_order": row["q_order"],
        "lower_q_order": lower_q_order,
    }


def _write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, str]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _export_vvvv(source: Path, output: Path) -> set[str]:
    rows = _read_primary_rows(source)
    fields = list(COMMON_OUTPUT_FIELDS)
    for channel in range(1, 4):
        fields.extend(
            (
                f"M{channel}_re",
                f"M{channel}_im",
                f"M{channel}_adjacent_q_abs_change",
                f"M{channel}_adjacent_q_rel_change",
            )
        )

    exported: list[dict[str, str]] = []
    for row in rows:
        q_order = int(row["q_order"])
        out = _common_values(row, str(q_order - 1))
        for channel in range(1, 4):
            out[f"M{channel}_re"] = row[f"M{channel}_re"]
            out[f"M{channel}_im"] = row[f"M{channel}_im"]
            out[f"M{channel}_adjacent_q_abs_change"] = row[
                f"estimated_abs_error_M{channel}"
            ]
            out[f"M{channel}_adjacent_q_rel_change"] = row[
                f"estimated_rel_error_M{channel}"
            ]
        exported.append(out)
    _write_csv(output, fields, exported)
    return {row["point_id"] for row in rows}


def _export_singlet_process(
    source: Path,
    output: Path,
    process: str,
) -> set[str]:
    rows = _read_primary_rows(source)
    fields = list(COMMON_OUTPUT_FIELDS) + [
        "amplitude_re",
        "amplitude_im",
        "lower_q_amplitude_re",
        "lower_q_amplitude_im",
        "adjacent_q_abs_change",
        "adjacent_q_rel_change",
        "momentum_nodes",
        "maximum_gram_condition",
        "block_backend",
        "maximum_recursion_cancellation",
        "high_precision_recursion_node_count",
        "series_parameter",
        "momentum_scheme",
        "momentum_quadrature",
    ]
    exported: list[dict[str, str]] = []
    for row in rows:
        out = _common_values(row, row["lower_q_order"])
        out.update(
            {
                "amplitude_re": row[f"{process}_raw_re"],
                "amplitude_im": row[f"{process}_raw_im"],
                "lower_q_amplitude_re": row[f"{process}_raw_lower_q_re"],
                "lower_q_amplitude_im": row[f"{process}_raw_lower_q_im"],
                "adjacent_q_abs_change": row[
                    f"{process}_raw_adjacent_q_abs_change"
                ],
                "adjacent_q_rel_change": row[
                    f"{process}_raw_adjacent_q_rel_change"
                ],
                "momentum_nodes": row["momentum_nodes"],
                "maximum_gram_condition": row["maximum_gram_condition"],
                "block_backend": row.get("block_backend") or "inverse_gram_legacy",
                "maximum_recursion_cancellation": row.get("maximum_recursion_cancellation", ""),
                "high_precision_recursion_node_count": row.get("high_precision_recursion_node_count", ""),
                "series_parameter": row.get("series_parameter") or "elliptic_nome_legacy",
                "momentum_scheme": row.get("momentum_scheme") or "unrecorded_legacy",
                "momentum_quadrature": row.get("momentum_quadrature", ""),
            }
        )
        exported.append(out)
    _write_csv(output, fields, exported)
    return {row["point_id"] for row in rows}


def export_tables(
    vvvv_source: Path,
    singlet_source: Path,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    """Export the three primary numerical-amplitude tables."""

    output_dir.mkdir(parents=True, exist_ok=True)
    vvvv_output = output_dir / "spin23_v_to_vvv_numerical.csv"
    ssvv_output = output_dir / "spin23_s_to_svv_numerical.csv"
    ssss_output = output_dir / "spin23_s_to_sss_numerical.csv"

    vvvv_ids = _export_vvvv(vvvv_source, vvvv_output)
    ssvv_ids = _export_singlet_process(singlet_source, ssvv_output, "ssvv")
    ssss_ids = _export_singlet_process(singlet_source, ssss_output, "ssss")
    if not (vvvv_ids == ssvv_ids == ssss_ids):
        raise ValueError("the three exports do not cover identical point_id sets")
    return vvvv_output, ssvv_output, ssss_output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vvvv-source", type=Path, default=DEFAULT_VVVV_SOURCE)
    parser.add_argument("--singlet-source", type=Path, default=DEFAULT_SINGLET_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    outputs = export_tables(args.vvvv_source, args.singlet_source, args.output_dir)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
