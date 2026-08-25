#!/usr/bin/env python3
"""Assemble the 20 immutable shards of the stable-q bosonic locality test."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

from run_bosonic_genus2_stable_q_check import _convergence_summary, _plot


EXPECTED_ORDERS = ((8, 10), (10, 10), (12, 10), (12, 12))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--shards-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)

    design = json.loads(args.design.read_text())
    point_ids = [str(value) for value in design["selected_ids"]]
    expected = {
        (point_id, recursion, momentum)
        for point_id in point_ids
        for recursion, momentum in EXPECTED_ORDERS
    }
    summaries = sorted(args.shards_dir.glob("*/summary.json"))
    rows: list[dict[str, object]] = []
    geometry_reference: object | None = None
    shard_records: list[dict[str, object]] = []
    failures: list[str] = []
    for path in summaries:
        payload = json.loads(path.read_text())
        geometry = payload.get("geometry_preflight")
        if geometry_reference is None:
            geometry_reference = geometry
        elif geometry != geometry_reference:
            failures.append(f"{path}: geometry preflight differs from the first shard")
        shard_rows = list(payload.get("rows", []))
        if len(shard_rows) != 1:
            failures.append(f"{path}: expected exactly one CFT row, found {len(shard_rows)}")
            continue
        row = dict(shard_rows[0])
        row["shard_summary"] = str(path)
        rows.append(row)
        shard_records.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "status": row.get("status"),
            }
        )
        if row.get("status") != "ok":
            failures.append(f"{path}: {row.get('error', 'failed without an error message')}")

    observed = {
        (str(row["point_id"]), int(row["recursion_order"]), int(row["momentum_order"]))
        for row in rows
    }
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if missing:
        failures.append(f"missing shards: {missing}")
    if unexpected:
        failures.append(f"unexpected shards: {unexpected}")
    if len(observed) != len(rows):
        failures.append("duplicate point/order shards are present")

    rows.sort(
        key=lambda row: (
            point_ids.index(str(row["point_id"])),
            EXPECTED_ORDERS.index(
                (int(row["recursion_order"]), int(row["momentum_order"]))
            ),
        )
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        _write_csv(args.out_dir / "frame_comparison.csv", rows)
        _plot(args.out_dir / "bosonic_stable_q_locality.png", rows)
    final_rows = [
        row
        for row in rows
        if (int(row["recursion_order"]), int(row["momentum_order"])) == (12, 12)
        and row.get("status") == "ok"
    ]
    summary = {
        "scope": "Bosonic genus-two theta/glasses locality at better-conditioned q.",
        "matching_assumed": False,
        "design": str(args.design),
        "design_sha256": _sha256(args.design),
        "runner": str(args.runner),
        "runner_sha256": _sha256(args.runner),
        "selected_ids": point_ids,
        "orders": [list(value) for value in EXPECTED_ORDERS],
        "expected_shards": len(expected),
        "observed_shards": len(rows),
        "geometry_preflight": geometry_reference,
        "convergence": _convergence_summary(rows),
        "final_R12_N12": {
            str(row["point_id"]): {
                "q_l_theta_over_glasses": float(row["theta_over_glasses"]),
                "relative_difference": float(row["relative_difference"]),
            }
            for row in final_rows
        },
        "failures": failures,
        "shards": shard_records,
        "rows": rows,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if failures:
        raise RuntimeError("stable-q reduction failed: " + "; ".join(failures))
    print(f"assembled {len(rows)} shards into {args.out_dir / 'summary.json'}")


if __name__ == "__main__":
    run()
