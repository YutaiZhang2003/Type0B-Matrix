"""Repeat the CFT momentum refinement at the highest audited sewing order."""

import argparse
import hashlib
import json
from pathlib import Path

import check_spin23_cft_repairs as audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a fresh output path")
    rows = []
    for path in args.reports:
        report = json.loads(path.read_text())
        for source in (Path(audit.__file__), Path(audit.s.__file__),
                       Path(audit.s.__file__).with_name("spin23_ns_c_recursion.py"),
                       Path(audit.s.ref.__file__).with_name("liouville_momentum_quadrature.py")):
            if hashlib.sha256(source.read_bytes()).hexdigest() != report["provenance"][source.name]:
                parser.error("source changed since reference report: " + str(source))
        for case in report["cases"]:
            orders = case["sewing_order_sweeps"]["original"]
            order = max(map(int, orders))
            energies = tuple(complex(*value) for value in case["energies"])
            atlas = audit.build(energies, order, (12, 64, 24))
            values = audit.snapshot(atlas, order)
            rows.append({"reference_report": str(path), "case": case["label"], "order": order,
                         "reference_nodes": 144, "candidate_nodes": 100, "values": values,
                         "change_from_144": audit.compare(orders[str(order)], values)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"status": "fixed_order_CFT_momentum_control", "rows": rows}, indent=2)+"\n")
    print(json.dumps({"max_CFT_relative": max(row["change_from_144"]["max_CFT_relative"] for row in rows),
                      "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
