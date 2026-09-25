"""Compare integrated primary correlators, never individual channel blocks."""
import argparse
import json
from pathlib import Path

import numpy as np

from benchmark_so7e8_primary_crossing_identity import pairs, unpairs
from so7e8_sphere_branches import require_same_coordinate_branch
from so7e8_ramond_metric import (
    CONDITIONAL_RAMOND_METRIC,
    inverse_pairing_factor,
    metric_metadata,
)


def read(path):
    data = json.loads(Path(path).read_text())
    if data["status"] != "complete":
        raise ValueError(f"refusing an incomplete momentum integral: {path}")
    return data


def delta(a, b):
    return abs(a-b)/np.maximum(np.maximum(abs(a), abs(b)), 1e-300)


def compare(ns, ramond, previous_ns=None, previous_ramond=None,
            *, working_metric=CONDITIONAL_RAMOND_METRIC):
    identity_factor = inverse_pairing_factor(("R",), convention=CONDITIONAL_RAMOND_METRIC)
    working_factor = inverse_pairing_factor(("R",), convention=working_metric)
    if ns["status"] != "complete" or ramond["status"] != "complete":
        raise ValueError("cannot compare incomplete momentum integrals")
    if ns["parameters"]["channel"] != "NS" or ramond["parameters"]["channel"] != "R":
        raise ValueError("expected an NS report and an R report")
    for field in ("momenta", "points", "sources"):
        if ns["parameters"][field] != ramond["parameters"][field]:
            raise ValueError(f"cannot compare mismatched {field}")
    coordinate_branch = require_same_coordinate_branch(ns["parameters"], ramond["parameters"])
    for current, previous in ((ns, previous_ns), (ramond, previous_ramond)):
        if previous is None:
            continue
        if previous["status"] != "complete":
            raise ValueError("cannot refine against an incomplete momentum integral")
        require_same_coordinate_branch(current["parameters"], previous["parameters"])
        for field in ("channel", "momenta", "points", "sources"):
            if current["parameters"][field] != previous["parameters"][field]:
                raise ValueError(f"refinement has mismatched {field}")
    orders = sorted(set(ns["results"]) & set(ramond["results"]), key=int)
    result = dict(scope="P-integrated R+ V V R+ primary Liouville correlator, no PCO/GSO/free fields",
                  physical_heterotic_amplitude=False, production_enabled=False,
                  identity_metric_agrees_with_recorded_equal_norms=False,
                  normalization_multiplier_origin="identity limit E/C -> 1, O/C -> 0, Cplus=(E+O)/2",
                  fitted_to_crossing=False, momenta=ns["parameters"]["momenta"],
                  points=ns["parameters"]["points"], rows=[],
                  working_metric=metric_metadata(working_metric), coordinate_branch=coordinate_branch)
    previous_order = None
    for n in orders:
        a = unpairs(ns["results"][n]["total"])
        b = unpairs(ramond["results"][n]["total"])
        row = dict(maximum_twice_level=int(n), NS=pairs(a), R_equal_metric=pairs(b),
                   R_identity_metric=pairs(identity_factor*b), equal_metric_ratio=pairs(a/b),
                   identity_metric_ratio=pairs(a/(identity_factor*b)),
                   equal_metric_relative_defect=delta(a, b).tolist(),
                   identity_metric_relative_defect=delta(a, identity_factor*b).tolist(),
                   R_working_metric=pairs(working_factor*b),
                   working_metric_ratio=pairs(a/(working_factor*b)),
                   working_metric_relative_defect=delta(a, working_factor*b).tolist())
        row["quadrature_change"] = {}
        row["H_order_change"] = {}
        row["absolute_region_fraction"] = {}
        for name, current, previous in (("NS", ns, previous_ns), ("R", ramond, previous_ramond)):
            value = unpairs(current["results"][n]["total"])
            row["quadrature_change"][name] = None if previous is None else delta(
                value, unpairs(previous["results"][n]["total"])).tolist()
            row["H_order_change"][name] = None if previous_order is None else delta(
                value, unpairs(current["results"][previous_order]["total"])).tolist()
            region = unpairs(current["results"][n]["region_totals"])
            row["absolute_region_fraction"][name] = (abs(region)/np.maximum(abs(value), 1e-300)).tolist()
        result["rows"].append(row)
        previous_order = n
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ns", type=Path, required=True)
    parser.add_argument("--ramond", type=Path, required=True)
    parser.add_argument("--previous-ns", type=Path)
    parser.add_argument("--previous-ramond", type=Path)
    parser.add_argument("--working-metric", choices=("recorded_equal", CONDITIONAL_RAMOND_METRIC),
                        default=CONDITIONAL_RAMOND_METRIC,
                        help="explicit working prescription; both diagnostic metrics are retained")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(read(args.ns), read(args.ramond),
                     read(args.previous_ns) if args.previous_ns else None,
                     read(args.previous_ramond) if args.previous_ramond else None,
                     working_metric=args.working_metric)
    result["inputs"] = {k: str(v) for k, v in vars(args).items() if v is not None and k != "output"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result, indent=2))
