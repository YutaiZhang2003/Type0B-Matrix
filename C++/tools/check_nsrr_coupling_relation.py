#!/usr/bin/env python3
"""Audit the f=1 NSRR coupling label using saved fixed-spin node outputs.

The alternate calculation only exchanges C_{1,+} and C_{1,-}. It reuses
the algorithm's returned blocks and all geometric inputs unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def complex_number(value: dict[str, str]) -> complex:
    return complex(float(value["real"]), float(value["imag"]))


def check_point(point: Path) -> list[dict[str, float | int | str]]:
    comparison = json.loads((point / "comparison.json").read_text())
    row = comparison["comparisons"][-1]
    original: list[list[float]] = [[], []]
    exchanged: list[list[float]] = [[], []]
    nodes = sorted((point / "source").glob("node-*.json"))
    if len(nodes) != comparison["source_N"] ** 3:
        raise ValueError(f"incomplete source grid: {point}")
    for path in nodes:
        node = json.loads(path.read_text())
        forms = node["rows"][-1]["F_fixed"]
        couplings = [complex_number(value) for value in node["C_f_eta"][0]]
        if node["C_f_eta"][0] != node["C_f_eta"][1]:
            raise ValueError(f"unexpected f-dependent inputs: {path}")
        weight = (complex_number(node["measure"])
                  * abs(complex_number(node["primary"])) ** 2 / 4)
        for spin in (0, 1):
            block = [[complex_number(value) for value in pair]
                     for pair in forms[spin]]
            original[spin].append((weight * sum(
                couplings[eta] ** 2 * abs(block[f][eta]) ** 2
                for f in (0, 1) for eta in (0, 1)
            )).real)
            exchanged[spin].append((weight * sum(
                couplings[eta if f == 0 else 1 - eta] ** 2
                * abs(block[f][eta]) ** 2
                for f in (0, 1) for eta in (0, 1)
            )).real)
    answer = []
    for spin in (0, 1):
        source = math.fsum(original[spin])
        stored = complex_number(row["source_R_fixed"][spin]).real
        if abs(source - stored) > 1e-11 * max(1, abs(stored)):
            raise AssertionError(f"could not reproduce source contraction: {point}")
        ratio = complex_number(row["fixed_spin_ratios"][spin]).real
        answer.append({
            "point": point.name, "spin": spin,
            "same_eta_ratio": ratio,
            "exchanged_eta_ratio": ratio * math.fsum(exchanged[spin]) / source,
        })
    return answer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path,
                        help="directory containing fixed-spin point runs")
    args = parser.parse_args()
    points = sorted(path for path in args.runs.iterdir()
                    if path.is_dir() and (path / "comparison.json").exists())
    if not points:
        parser.error("no completed point runs found")
    rows = [row for point in points for row in check_point(point)]
    print(json.dumps({
        "points": len(points), "fixed_spin_comparisons": len(rows),
        "maximum_same_eta_deviation": max(abs(r["same_eta_ratio"] - 1)
                                          for r in rows),
        "maximum_exchanged_eta_deviation": max(
            abs(r["exchanged_eta_ratio"] - 1) for r in rows),
        "first_point": rows[:2],
    }, indent=2))


if __name__ == "__main__":
    main()
