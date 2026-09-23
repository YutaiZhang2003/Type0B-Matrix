#!/usr/bin/env python3
"""Recompute the saved nonchiral ratio and expose its normalization assumption.

This is a saved-data audit, not a new momentum integral or a chiral-block test.
It never changes the underlying partition values or fits a normalization.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from section6 import atomic_json

REPO = Path(__file__).resolve().parents[2]
DEFAULT = REPO / "Data Set/nsrr_bilinear_quadrature_20260915/final_result.json"


def audit(source):
    data = json.loads(source.read_text())
    # The original summary stops at N7. The completed study holds the source
    # at N7 and refines the target to N10, recorded separately in final_result.
    inputs = {str(source): hashlib.sha256(source.read_bytes()).hexdigest()}
    if "results" in data:
        results = data["results"]
    else:
        frame_source = source.with_name("summary.json")
        frame_data = json.loads(frame_source.read_text())
        frames = [r["free_frame_power"] for r in frame_data["results"]]
        if not frames or any(f != frames[0] for f in frames):
            raise ValueError("the saved free-field frame factor is not constant")
        inputs[str(frame_source)] = hashlib.sha256(frame_source.read_bytes()).hexdigest()
        results = [dict(N=data["target_N"], complete=True, free_frame_power=frames[0],
                        values=data["values"])]
    rows = []
    for result in results:
        if not result["complete"]:
            raise ValueError("cannot audit an incomplete quadrature order")
        for name, values in result["values"].items():
            raw = values["source_Z"] / values["target_Z"] / result["free_frame_power"]
            if not math.isclose(raw, values["ratio"], rel_tol=1e-13, abs_tol=1e-15):
                raise ValueError(f"stored ratio is inconsistent at N={result['N']}, {name}")
            rows.append({"quadrature_order": result["N"], "component": name,
                         "raw_ratio": raw, "raw_deviation_from_one": abs(raw-1),
                         "ratio_with_assumed_factor_four": 4*raw,
                         "deviation_with_assumed_factor_four": abs(4*raw-1)})
    highest = max(r["quadrature_order"] for r in rows)
    fine = [r for r in rows if r["quadrature_order"] == highest]
    return {
        "status": "saved_ratios_reproduced_cross_channel_equality_not_established",
        "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "input_sha256": inputs,
        "new_block_evaluations": 0, "new_momentum_integrations": 0,
        "normalization_fitted": False, "factor_four_is_assumed_not_derived": True,
        "global_normalization_verified": False,
        "highest_completed_quadrature_order": highest,
        "highest_order_raw_ratio_range": [min(r["raw_ratio"] for r in fine), max(r["raw_ratio"] for r in fine)],
        "highest_order_max_deviation_with_assumed_factor_four": max(r["deviation_with_assumed_factor_four"] for r in fine),
        "saved_convergence_status": data["status"], "saved_scope": data["scope"],
        "source_quadrature_order": data.get("source_N"),
        "target_quadrature_order": data.get("target_N"),
        "interpretation": "Agreement of chiral coefficients does not establish the global nonchiral normalization or interacting spin transport. The raw local-BPW normalization gives ratios near one quarter; multiplying by four is a separately recorded assumption.",
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.source)
    atomic_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
