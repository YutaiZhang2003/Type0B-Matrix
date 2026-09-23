#!/usr/bin/env python3
"""Verify completed tetrahedron data and a current-code prefix, without reruns."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from section6 import atomic_json, compare_values, graph_rows

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "C++/experiments/mercedes_all_ns_level10_2026-09-16"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True, help="Current total-level all-NS tetrahedron DV output")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix-field", choices=("double_virasoro", "c_recursion"), default="double_virasoro")
    args = parser.parse_args()
    metadata = json.loads((args.prefix / "metadata.json").read_text())
    if metadata["channel"] != "tetrahedron-ns" or metadata["truncation"] != "total" or metadata["level"] > 10:
        raise ValueError("prefix must be the all-NS tetrahedron at total level <=10")
    manifest = json.loads((ARCHIVE / "manifest_L10.json").read_text())
    checked = {}
    for name in ("coefficients_L10.jsonl", "c_recursion_L10.jsonl", "results_L10.json", "timing_dv_L10.json", "timing_ns_L10.json"):
        path = ARCHIVE / name
        key = str(path.relative_to(ROOT))
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != manifest[key]:
            raise ValueError(f"archived output changed: {key}")
        checked[key] = actual
    whole = dict(metadata, level=10)
    dv = graph_rows(ARCHIVE / "coefficients_L10.jsonl", "double_virasoro", whole)
    ns = graph_rows(ARCHIVE / "c_recursion_L10.jsonl", "c_recursion", whole)
    full = compare_values(dv, ns, "1e-18")
    if full["components"] != math.comb(26, 6):
        raise ValueError("incomplete archived total-level-10 domain")
    bound = 2 * metadata["level"]
    reference = dv if args.prefix_field == "double_virasoro" else ns
    prefix = compare_values(graph_rows(args.prefix / "coefficients.jsonl", args.prefix_field, metadata),
                            {k: v for k, v in reference.items() if sum(k[1:]) <= bound}, "1e-18")
    result = {"passed": full["passed"] and prefix["passed"], "new_block_evaluations": 0,
              "scope": "Recomparison of the complete archived level-10 coefficients and comparison of a fresh production prefix; not a fresh level-10 run.",
              "archived_total10": full, "current_code_prefix": prefix,
              "prefix_level": metadata["level"], "prefix_field": args.prefix_field, "verified_output_sha256": checked}
    atomic_json(args.output, result)
    print(json.dumps({"passed": result["passed"], "archived_components": full["components"],
                      "archived_max_scaled": full["max_scaled"], "prefix_max_scaled": prefix["max_scaled"]}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
