#!/usr/bin/env python3
"""Diagnostic of the archived Python seed at the frozen complex overlap point.

This does not run in the native partition pipeline or modify the archived
implementation. The infinity-lift flip below is specific to this point.
"""
import argparse
import cmath
from functools import lru_cache
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
for folder in ("Code", "Code/c_Recursion", "Code/genus_2", "Code/genus_2_cross_channel"):
    sys.path.insert(0, str(ROOT / folder))
from ns_genus2_partition import NSGenus2CRecursion, _theta_schottky_data
from ns_vacuum_schottky import ns_schottky_vacuum_block


class CorrectedAtThisPoint(NSGenus2CRecursion):
    @lru_cache(maxsize=None)
    def _vacuum(self, lifts):
        return super()._vacuum((lifts[0], lifts[1], -lifts[2]))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-node", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    config = json.loads((ROOT / "Data Set/nsrr_bilinear_quadrature_20260915/config.json").read_text())
    old = json.loads((ROOT / "Data Set/nsrr_bilinear_quadrature_20260915/target/N10/node-0000.json").read_text())
    new = json.loads(args.native_node.read_text())
    q = tuple(map(complex, config["point"]["q_target"]))
    lifts = ((1, 1, 1), (1, -1, 1), (-1, 1, 1), (-1, -1, 1))
    leading = []
    for scale in (.3, .1):
        values = []
        for lift in lifts:
            generators, signs = _theta_schottky_data(tuple(scale * z for z in q), lift)
            values.append(ns_schottky_vacuum_block(generators, signs, max_word_length=3, max_mode=8).value)
        for i, j in ((0, 1), (0, 2), (1, 2)):
            wanted = -(cmath.sqrt(q[i]) * cmath.sqrt(q[j])) ** 3
            actual = sum((v - 1) * lift[i] * lift[j] for v, lift in zip(values, lifts)) / (4 * scale**3)
            leading.append(dict(scale=scale, geometry_edges=[i, j], old_to_PBW_leading_ratio=[(actual/wanted).real, (actual/wanted).imag]))
    rec = CorrectedAtThisPoint(channel="theta", q_values=q, global_method="resummed",
        global_tolerance=config["global_tolerance"], global_max_total_occupation=config["global_max_occupation"],
        vacuum_word_length=7, vacuum_max_mode=50)
    b = config["b"]
    rows = []
    for sector in (0, 1):
        for j, lift in enumerate(lifts):
            value = rec.collision_aware_block_mp(weights=old["weights"], sector=sector,
                recursion_order=12, lifts=lift, central_charge=1.5+3*(b+1/b)**2, working_precision=40)
            co = new["rows"][-1]["literal_blocks"][sector][j]
            native = complex(float(co["real"]), float(co["imag"]))
            rows.append(dict(sector=sector, lift=lift, corrected=[value.real, value.imag],
                relative_to_native_L8=abs(value-native)/max(abs(value), 1e-300)))
    report = dict(scope="one archived target node; fixture-specific diagnostic only", leading_links=leading,
        comparisons=rows, maximum_relative=max(r["relative_to_native_L8"] for r in rows),
        seconds=time.perf_counter()-started)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: report[k] for k in ("maximum_relative", "seconds")}))


if __name__ == "__main__":
    main()
