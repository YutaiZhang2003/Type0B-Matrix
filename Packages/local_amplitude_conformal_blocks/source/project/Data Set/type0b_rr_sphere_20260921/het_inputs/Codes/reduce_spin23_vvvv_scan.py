#!/usr/bin/env python3
"""Merge one-row Spin(23) scan outputs and summarize formula residuals."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _percentiles(values: list[float]) -> dict[str,float]:
    data=np.asarray(values,dtype=float)
    return {
        "min":float(np.min(data)),
        "median":float(np.median(data)),
        "p90":float(np.percentile(data,90)),
        "p99":float(np.percentile(data,99)),
        "max":float(np.max(data)),
    }


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir",type=Path)
    parser.add_argument("--expected",type=int,default=None)
    args=parser.parse_args()

    rows=[]
    for path in sorted(args.results_dir.glob("result_*.csv")):
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"no result CSV files found in {args.results_dir}")

    point_ids=[row["point_id"] for row in rows]
    if len(point_ids) != len(set(point_ids)):
        raise SystemExit("duplicate point_id values found")
    if args.expected is not None and len(rows) != args.expected:
        raise SystemExit(f"found {len(rows)} rows; expected {args.expected}")

    merged=args.results_dir/"merged.csv"
    fields=[]
    known_fields=set()
    for row in rows:
        for field in row:
            if field not in known_fields:
                fields.append(field)
                known_fields.add(field)
    with merged.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    runtimes=[float(row["runtime_seconds"]) for row in rows]
    summary={
        "completed_points":len(rows),
        "runtime_seconds":_percentiles(runtimes),
        "total_cpu_hours":float(sum(runtimes)/3600),
        "block_backend":{
            "node_pairs":sum(int(row.get("block_block_node_pairs",0)) for row in rows),
            "reference_node_pairs":sum(
                int(row.get("block_reference_block_node_pairs",0)) for row in rows
            ),
            "fast_node_pairs":sum(int(row.get("block_fast_block_node_pairs",0)) for row in rows),
            "max_fast_recursion_condition":max(
                float(row.get("block_max_fast_recursion_condition",1.0)) for row in rows
            ),
        },
        "channels":{},
    }
    reason_fields=sorted(
        field for field in fields if field.startswith("block_block_backend_reason_")
    )
    summary["block_backend"]["reasons"]={
        field.removeprefix("block_block_backend_reason_"):sum(
            int(row.get(field,0) or 0) for row in rows
        )
        for field in reason_fields
    }
    all_relative=[]
    all_estimated=[]
    for name in ("M1","M2","M3"):
        relative=[float(row[f"formula_rel_residual_{name}"]) for row in rows]
        estimated=[float(row[f"estimated_abs_error_{name}"]) for row in rows]
        estimated_relative=[float(row[f"estimated_rel_error_{name}"]) for row in rows]
        absolute=[float(row[f"formula_abs_residual_{name}"]) for row in rows]
        residual_over_change=[
            float(row[f"formula_residual_over_q_change_{name}"]) for row in rows
        ]
        summary["channels"][name]={
            "formula_relative_residual":_percentiles(relative),
            "formula_absolute_residual":_percentiles(absolute),
            "adjacent_q_order_absolute_change":_percentiles(estimated),
            "adjacent_q_order_relative_change":_percentiles(estimated_relative),
            "formula_residual_over_q_change":_percentiles(residual_over_change),
            "worst_formula_relative_residual":{
                "point_id":rows[int(np.argmax(relative))]["point_id"],
                "value":float(max(relative)),
            },
        }
        all_relative.extend(relative)
        all_estimated.extend(estimated)
    summary["all_channel_formula_relative_residual"]=_percentiles(all_relative)
    summary["all_channel_adjacent_q_order_absolute_change"]=_percentiles(all_estimated)
    summary_path=args.results_dir/"summary.json"
    summary_path.write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
