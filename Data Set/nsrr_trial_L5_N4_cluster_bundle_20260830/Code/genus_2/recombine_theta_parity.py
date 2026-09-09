#!/usr/bin/env python3
"""Recombine archived theta-channel shards with the Type-0B parity sign.

The expensive conformal-block values in an old production archive remain
valid because the archive stores the even and odd theta sectors separately.
Only their nonchiral sewing was wrong: it added both sectors.  This utility
validates the archive provenance and replaces

    Z_theta = Z_even + Z_odd

by the theta-channel formula from ``Human Notes/SCblock.tex``,

    Z_theta = Z_even - Z_odd

for the even NS primaries integrated in the Type-0B calculation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from genus_2.theta_partition import (
    TYPE0B_NS_PRIMARY_PARITIES,
    theta_sector_pair,
)


OUTPUT_SCHEMA = "ns-genus2-theta-parity-recombination-v1"
SHARD_RE = re.compile(r"(?:^|/)shards/task-(\d{6})\.json$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _member_json(archive: tarfile.TarFile, member: tarfile.TarInfo) -> Any:
    stream = archive.extractfile(member)
    if stream is None:
        raise RuntimeError(f"could not read archive member {member.name}")
    return json.load(stream)


def _row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["point_id"],
        row["channel"],
        int(row["recursion_order"]),
        int(row["quadrature_order"]),
        float(row["finite_part_radius"]),
    )


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=5.0e-15, abs_tol=1.0e-300)


def recombine_archive(archive_path: Path) -> dict[str, Any]:
    """Return parity-corrected theta results from one Cannon archive."""

    archive_path = archive_path.resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)

    with tarfile.open(archive_path, "r:gz") as archive:
        summary_members = [
            member
            for member in archive.getmembers()
            if member.name == "summary.json" or member.name.endswith("/summary.json")
        ]
        if len(summary_members) != 1:
            raise RuntimeError(
                f"expected one summary.json, found {len(summary_members)}"
            )
        source_summary = _member_json(archive, summary_members[0])

        # Preserve physical tar order while reading.  Seeking among members of
        # a compressed tarball would repeatedly decompress the archive.
        shard_members: list[tuple[int, tarfile.TarInfo]] = []
        for member in archive.getmembers():
            match = SHARD_RE.search(member.name)
            if match:
                shard_members.append((int(match.group(1)), member))
        expected_count = int(source_summary["task_count"])
        indices = sorted(index for index, _member in shard_members)
        if indices != list(range(expected_count)):
            raise RuntimeError(
                "archive shards are incomplete, duplicated, or non-contiguous"
            )

        source_schema = source_summary["schema"]
        config_digest = source_summary["config_digest"]
        fingerprint = source_summary["implementation_fingerprint"]

        # Keep node contributions in task order.  math.fsum below then gives a
        # deterministic reduction independent of tar member ordering.
        sector_terms: dict[
            tuple[Any, ...], dict[int, list[tuple[int, float]]]
        ] = defaultdict(
            lambda: {0: [], 1: []}
        )
        theta_shard_count = 0

        for task_index, member in shard_members:
            shard = _member_json(archive, member)
            if int(shard["task_index"]) != task_index:
                raise RuntimeError(f"task index mismatch in {member.name}")
            if shard["schema"] != source_schema:
                raise RuntimeError(f"schema mismatch in {member.name}")
            if shard["config_digest"] != config_digest:
                raise RuntimeError(f"config digest mismatch in {member.name}")
            if shard["implementation_fingerprint"] != fingerprint:
                raise RuntimeError(
                    f"implementation fingerprint mismatch in {member.name}"
                )
            if shard["channel"] != "theta":
                continue

            theta_shard_count += 1
            base_key = (
                shard["point_id"],
                shard["channel"],
                int(shard["recursion_order"]),
                int(shard["quadrature_order"]),
            )
            for radius_result in shard["radius_results"]:
                sectors = radius_result["sectors"]
                if {int(item["sector"]) for item in sectors} != {0, 1}:
                    raise RuntimeError(
                        f"theta sectors are not exactly {{0,1}} in {member.name}"
                    )
                unsigned_node = math.fsum(
                    float(item["contribution"]) for item in sectors
                )
                if not _close(unsigned_node, float(radius_result["contribution"])):
                    raise RuntimeError(
                        f"stored unsigned sector sum mismatch in {member.name}"
                    )
                key = base_key + (float(radius_result["finite_part_radius"]),)
                for item in sorted(sectors, key=lambda value: int(value["sector"])):
                    sector = int(item["sector"])
                    sector_terms[key][sector].append(
                        (task_index, float(item["contribution"]))
                    )

    source_rows = {
        _row_key(row): row
        for row in source_summary["rows"]
        if row["channel"] == "theta"
    }
    if set(sector_terms) != set(source_rows):
        missing = sorted(set(source_rows) - set(sector_terms))
        extra = sorted(set(sector_terms) - set(source_rows))
        raise RuntimeError(f"theta row mismatch: missing={missing}, extra={extra}")

    corrected_rows: list[dict[str, Any]] = []
    for row in source_summary["rows"]:
        if row["channel"] != "theta":
            continue
        key = _row_key(row)
        even_sum = math.fsum(
            value for _index, value in sorted(sector_terms[key][0])
        )
        odd_sum = math.fsum(
            value for _index, value in sorted(sector_terms[key][1])
        )
        old_z = float(row["z_liouville"])
        reconstructed_old_z = math.fsum((even_sum, odd_sum))
        if not _close(old_z, reconstructed_old_z):
            raise RuntimeError(
                f"archived summary does not reproduce for {row['point_id']}: "
                f"summary={old_z!r}, sectors={reconstructed_old_z!r}"
            )

        signed_sector_sums = {
            sector: theta_sector_pair(
                sector,
                holomorphic_primary_parities=TYPE0B_NS_PRIMARY_PARITIES,
                antiholomorphic_primary_parities=TYPE0B_NS_PRIMARY_PARITIES,
            ).sign
            * sector_sum
            for sector, sector_sum in ((0, even_sum), (1, odd_sum))
        }
        corrected_z = math.fsum(signed_sector_sums.values())
        z_free = float(row["z_free_superfield"])
        corrected_q_l = corrected_z / z_free**9
        old_q_l = float(row["q_l"])

        corrected_rows.append(
            {
                "point_id": row["point_id"],
                "channel": "theta",
                "recursion_order": int(row["recursion_order"]),
                "quadrature_order": int(row["quadrature_order"]),
                "node_count": int(row["node_count"]),
                "finite_part_radius": float(row["finite_part_radius"]),
                "primary_parities": list(TYPE0B_NS_PRIMARY_PARITIES),
                "z_even": even_sum,
                "z_odd_unsigned": odd_sum,
                "z_liouville_old_unsigned": old_z,
                "z_liouville_corrected": corrected_z,
                "z_free_superfield": z_free,
                "q_l_old_unsigned": old_q_l,
                "q_l_corrected": corrected_q_l,
                "absolute_q_l_shift": corrected_q_l - old_q_l,
                "relative_q_l_shift": corrected_q_l / old_q_l - 1.0,
                "odd_fraction_of_old_unsigned_z": odd_sum / old_z,
                "old_summary_reduction_error": reconstructed_old_z - old_z,
            }
        )

    return {
        "schema": OUTPUT_SCHEMA,
        "method": "exact sector-wise recombination of archived block evaluations",
        "theta_formula": "Z_theta = Z_even - Z_odd for p=(0,0,0)",
        "q_l_formula": "Q_L = Z_L / Z_free_superfield^9",
        "block_reevaluation_required": False,
        "source": {
            "archive": str(archive_path),
            "archive_sha256": _sha256(archive_path),
            "schema": source_schema,
            "config_digest": config_digest,
            "implementation_fingerprint": fingerprint,
            "task_count": expected_count,
            "theta_shard_count": theta_shard_count,
        },
        "corrected_rows": corrected_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="production .tar.gz archive")
    parser.add_argument(
        "--output",
        type=Path,
        help="write the corrected result as JSON (stdout if omitted)",
    )
    args = parser.parse_args()

    result = recombine_archive(args.archive)
    text = json.dumps(result, indent=2, sort_keys=False) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(args.output.resolve())


if __name__ == "__main__":
    main()
