#!/usr/bin/env python3
"""Join the certified period and fundamental-reduction indices.

The input archives are matched by both row id and base-table SHA-256.  The
result retains the version-2 plumbing-coordinate trees and adds a standardized
KD tree on Gottschling-domain period coordinates plus the exact
``Sp(4,Z)`` matrix taking every raw plumbing period to that representative.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np


SCHEMA_VERSION = 3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _scalar_string(archive: np.lib.npyio.NpzFile, key: str) -> str:
    return str(np.asarray(archive[key]).reshape(-1)[0])


def _fundamental_coordinates(omega_fund: np.ndarray) -> np.ndarray:
    values = np.asarray(omega_fund, dtype=np.complex128)
    if values.ndim != 2 or values.shape[1] != 3 or not np.all(np.isfinite(values)):
        raise ValueError("fundamental periods must have shape (N,3) and be finite")
    y11 = values[:, 0].imag
    y12 = values[:, 1].imag
    y22 = values[:, 2].imag
    if np.any(y11 <= 0.0) or np.any(y11 * y22 - y12 * y12 <= 0.0):
        raise ValueError("fundamental index contains a non-Riemann period matrix")
    return np.column_stack(
        (
            2.0 * values[:, 0].real,
            2.0 * values[:, 1].real,
            2.0 * values[:, 2].real,
            np.log(y11),
            np.log(y22 / y11),
            2.0 * y12 / y11,
        )
    ).astype(np.float64)


def build_combined_index(
    base_index_path: Path,
    fundamental_index_path: Path,
    canonical_table_path: Path,
    output_path: Path,
) -> dict[str, object]:
    with np.load(base_index_path, allow_pickle=False) as base, np.load(
        fundamental_index_path, allow_pickle=False
    ) as fundamental:
        base_version = int(np.asarray(base["schema_version"]).reshape(-1)[0])
        fundamental_version = int(
            np.asarray(fundamental["schema_version"]).reshape(-1)[0]
        )
        if base_version != 2:
            raise ValueError(f"expected a version-2 base period index, got {base_version}")
        if fundamental_version != 1:
            raise ValueError(
                f"expected a version-1 fundamental index, got {fundamental_version}"
            )
        base_sha = _scalar_string(base, "table_sha256")
        fundamental_base_sha = _scalar_string(fundamental, "base_table_sha256")
        if base_sha != fundamental_base_sha:
            raise ValueError("period and fundamental indices refer to different base tables")
        base_ids = np.asarray(base["row_ids"]).astype(str)
        fundamental_ids = np.asarray(fundamental["row_ids"]).astype(str)
        if not np.array_equal(base_ids, fundamental_ids):
            raise ValueError("period and fundamental indices have different row ordering")

        arrays = {key: np.asarray(base[key]).copy() for key in base.files}
        omega_fund = np.asarray(fundamental["omega_fund"], dtype=np.complex128)
        coordinates = _fundamental_coordinates(omega_fund)
        center = np.mean(coordinates, axis=0)
        scale = np.std(coordinates, axis=0)
        scale = np.where(scale > 1.0e-12, scale, 1.0)
        features = (coordinates - center) / scale
        canonical_sha = sha256(canonical_table_path)
        arrays.update(
            {
                "schema_version": np.asarray([SCHEMA_VERSION], dtype=np.int64),
                "canonical_table_sha256": np.asarray([canonical_sha]),
                "base_index_sha256": np.asarray([sha256(base_index_path)]),
                "fundamental_index_sha256": np.asarray(
                    [sha256(fundamental_index_path)]
                ),
                "omega_fund": omega_fund,
                "sp4_raw_to_fund": np.asarray(
                    fundamental["sp4_raw_to_fund"], dtype=np.int64
                ),
                "b_period_branches": np.asarray(
                    fundamental["b_period_branches"], dtype=np.int64
                ),
                "domain_margins": np.asarray(
                    fundamental["domain_margins"], dtype=np.float64
                ),
                "transform_residuals": np.asarray(
                    fundamental["transform_residuals"], dtype=np.float64
                ),
                "correction_indices": np.asarray(
                    fundamental["correction_indices"], dtype=np.int64
                ),
                "correction_depth_limits": np.asarray(
                    fundamental["correction_depth_limits"], dtype=np.int64
                ),
                "attempted_depths": np.asarray(fundamental["attempted_depths"]).copy(),
                "fundamental_feature_center": center.astype(np.float64),
                "fundamental_feature_scale": scale.astype(np.float64),
                "fundamental_features": features.astype(np.float64),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "row_count": int(len(base_ids)),
        "base_table_sha256": base_sha,
        "canonical_table": str(canonical_table_path),
        "canonical_table_sha256": canonical_sha,
        "base_index": str(base_index_path),
        "base_index_sha256": sha256(base_index_path),
        "fundamental_index": str(fundamental_index_path),
        "fundamental_index_sha256": sha256(fundamental_index_path),
        "combined_index": str(output_path),
        "combined_index_sha256": sha256(output_path),
        "minimum_domain_margin": float(np.min(arrays["domain_margins"])),
        "maximum_transform_residual": float(
            np.max(arrays["transform_residuals"])
        ),
        "query_coordinates": [
            "2 Re Omega11",
            "2 Re Omega12",
            "2 Re Omega22",
            "log Im Omega11",
            "log(Im Omega22/Im Omega11)",
            "2 Im Omega12/Im Omega11",
        ],
        "policy": (
            "The combined table supplies q and exact marking seeds only; every "
            "Monte Carlo node must be recomputed and certified by the live period solver."
        ),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    checksum_path = output_path.with_suffix(".sha256")
    checksum_path.write_text(f"{summary['combined_index_sha256']}  {output_path.name}\n")
    return summary


def run(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-index", type=Path, required=True)
    parser.add_argument("--fundamental-index", type=Path, required=True)
    parser.add_argument("--canonical-table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    print(
        json.dumps(
            build_combined_index(
                args.base_index,
                args.fundamental_index,
                args.canonical_table,
                args.output,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    run()
