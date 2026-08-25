#!/usr/bin/env python3
"""Compare two NS genus-two Cannon runs node by node.

This is deliberately an integrand-level audit.  A node is not excused because
its quadrature or structure-constant weight is small.  Relative changes are
reported for every spin sector using the symmetric denominator
``max(abs(old), abs(new))`` so the statistic remains bounded near zeros.
"""

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _complex(pair):
    return complex(float(pair[0]), float(pair[1]))


def _load_nodes(directory, point_id, channel, recursion_order=None):
    nodes = {}  # type: Dict[Tuple[int, ...], Dict[str, Any]]
    for filename in glob.glob(str(Path(directory) / "*.json")):
        with open(filename, "r", encoding="utf-8") as handle:
            shard = json.load(handle)
        if shard.get("point_id") != point_id or shard.get("channel") != channel:
            continue
        if recursion_order is not None and int(shard.get("recursion_order", -1)) != int(
            recursion_order
        ):
            continue
        key = tuple(int(value) for value in shard["indices"])
        if key in nodes:
            raise RuntimeError(
                "duplicate node %r in %s; specify a recursion-order filter"
                % (key, directory)
            )
        sectors = {
            int(item["sector"]): _complex(item["block"])
            for item in shard["radius_results"][0]["sectors"]
        }
        nodes[key] = {
            "momenta": [float(value) for value in shard["momenta"]],
            "measure": float(shard["measure"]),
            "sectors": sectors,
        }
    return nodes


def _quantile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    mix = position - lower
    return (1.0 - mix) * ordered[lower] + mix * ordered[upper]


def compare(
    old_directory,
    new_directory,
    point_id,
    channel,
    old_recursion_order=None,
    new_recursion_order=None,
    relative_tolerance=1.0e-3,
    intersection_only=False,
):
    old = _load_nodes(old_directory, point_id, channel, old_recursion_order)
    new = _load_nodes(new_directory, point_id, channel, new_recursion_order)
    if not old or not new:
        raise RuntimeError(
            "empty node set after recursion-order filtering: old=%d new=%d"
            % (len(old), len(new))
        )
    old_source_count = len(old)
    new_source_count = len(new)
    if intersection_only:
        common = set(old) & set(new)
        old = {key: old[key] for key in common}
        new = {key: new[key] for key in common}
    missing_old = sorted(set(new) - set(old))
    missing_new = sorted(set(old) - set(new))
    if missing_old or missing_new:
        raise RuntimeError(
            f"node sets differ: missing_old={missing_old[:5]}, missing_new={missing_new[:5]}"
        )

    sector_rows = {}  # type: Dict[int, List[Dict[str, Any]]]
    for indices in sorted(old):
        if set(old[indices]["sectors"]) != set(new[indices]["sectors"]):
            raise RuntimeError(f"sector sets differ at node {indices}")
        for sector in old[indices]["sectors"]:
            old_value = old[indices]["sectors"][sector]
            new_value = new[indices]["sectors"][sector]
            denominator = max(abs(old_value), abs(new_value), 1.0e-300)
            sector_rows.setdefault(sector, []).append(
                {
                    "indices": list(indices),
                    "momenta": old[indices]["momenta"],
                    "old": [old_value.real, old_value.imag],
                    "new": [new_value.real, new_value.imag],
                    "absolute_change": abs(new_value - old_value),
                    "symmetric_relative_change": abs(new_value - old_value) / denominator,
                }
            )

    summaries = {}  # type: Dict[str, Any]
    for sector, rows in sorted(sector_rows.items()):
        relative = [row["symmetric_relative_change"] for row in rows]
        worst = max(rows, key=lambda row: row["symmetric_relative_change"])
        summaries[str(sector)] = {
            "node_count": len(rows),
            "median": _quantile(relative, 0.5),
            "p90": _quantile(relative, 0.9),
            "p99": _quantile(relative, 0.99),
            "maximum": max(relative),
            "count_above_1e-2": sum(value > 1.0e-2 for value in relative),
            "count_above_1e-3": sum(value > 1.0e-3 for value in relative),
            "count_above_1e-4": sum(value > 1.0e-4 for value in relative),
            "worst_node": worst,
        }
    return {
        "schema": "ns-genus2-pointwise-recursion-audit-v1",
        "point_id": point_id,
        "channel": channel,
        "old_directory": old_directory,
        "new_directory": new_directory,
        "old_recursion_order": old_recursion_order,
        "new_recursion_order": new_recursion_order,
        "relative_tolerance": relative_tolerance,
        "intersection_only": bool(intersection_only),
        "old_source_node_count": old_source_count,
        "new_source_node_count": new_source_count,
        "pointwise_pass": all(
            summary["maximum"] <= relative_tolerance
            for summary in summaries.values()
        ),
        "node_count": len(old),
        "sectors": summaries,
    }


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("old_directory")
    parser.add_argument("new_directory")
    parser.add_argument("point_id")
    parser.add_argument("channel", choices=("theta", "glasses"))
    parser.add_argument("--old-recursion-order", type=int)
    parser.add_argument("--new-recursion-order", type=int)
    parser.add_argument("--relative-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--intersection-only", action="store_true")
    parser.add_argument("--fail-on-nonconvergence", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = compare(
        args.old_directory,
        args.new_directory,
        args.point_id,
        args.channel,
        old_recursion_order=args.old_recursion_order,
        new_recursion_order=args.new_recursion_order,
        relative_tolerance=args.relative_tolerance,
        intersection_only=args.intersection_only,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if args.fail_on_nonconvergence and not result["pointwise_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    run()
