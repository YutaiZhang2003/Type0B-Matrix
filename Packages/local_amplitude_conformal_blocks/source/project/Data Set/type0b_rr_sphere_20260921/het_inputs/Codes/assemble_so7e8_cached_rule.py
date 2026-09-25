"""Assemble a complete rebalanced quadrature from one unchanged CFT producer.

No density interpolation or new engine is used. Every requested node must
already exist with the baseline's exact source/parameter cache signature.
This permits refining endpoint/bulk counts while reusing an infinite-tail
rule, instead of spending most of the budget on a negligible tail.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmark_so7e8_momentum_refinement import integrate_rule, relative_change, unpairs
from heterotic_so23_1to3_vvvv_fit_bundle.liouville_momentum_quadrature import threshold_weighted_rule


def assemble(baseline, cache_dir, counts, output):
    baseline=Path(baseline)
    baseline_bytes=baseline.read_bytes()
    report=json.loads(baseline_bytes)
    previous=report["rows"][-1]
    orders=tuple(sorted(map(int,previous["blocks"])))
    rule=threshold_weighted_rule(tuple(counts),options=previous["momentum_rule"]["parameters"])
    directory=Path(cache_dir)/report["cache_signature"]
    def path(p):
        return directory/(hashlib.sha256(float(p).hex().encode()).hexdigest()+".json")
    missing=[float(p) for p in rule.momenta if not path(p).exists()]
    if missing:
        return dict(status="missing_nodes",count=len(missing),first=missing[:8])
    def evaluate(p):
        data=json.loads(path(p).read_text())
        if data["signature"]!=report["cache_signature"] or data["P_hex"]!=float(p).hex():
            raise ValueError("cache signature or momentum mismatch")
        result={n:{name:unpairs(data["values"][str(n)][name]) for name in ("F0","F2")} for n in orders}
        if any(not np.all(np.isfinite(v)) or len(v)!=len(report["parameters"]["points"])
               for row in result.values() for v in row.values()):
            raise ArithmeticError("incomplete or nonfinite cached density")
        return result
    row=integrate_rule(evaluate,rule,orders)
    row["relative_previous_momentum_change"]={n:{name:relative_change(row["blocks"][n][name]["value"],
        previous["blocks"][str(n)][name]["value"]) for name in ("F0","F2")} for n in orders}
    row["relative_previous_block_change"]={n:{name:None if i==0 else relative_change(
        row["blocks"][n][name]["value"],row["blocks"][orders[i-1]][name]["value"])
        for name in ("F0","F2")} for i,n in enumerate(orders)}
    report["requested_nodes"]=[r["momentum_rule"]["node_count"] for r in report["rows"]]+[list(counts)]
    report["rows"].append(row)
    report["status"]="complete"
    report["rebalanced_rule"]=dict(baseline=str(baseline.resolve()),
        baseline_sha256=hashlib.sha256(baseline_bytes).hexdigest(),
        assembler_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        explanation="Reuse prior infinite-tail nodes; refine endpoint and bulk independently; no node omitted")
    Path(output).write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    return dict(status="complete",counts=counts,total=len(rule.momenta),
                momentum_changes=row["relative_previous_momentum_change"],
                block_changes=row["relative_previous_block_change"])


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline",type=Path,required=True)
    parser.add_argument("--cache-dir",type=Path,required=True)
    parser.add_argument("--counts",nargs=3,type=int,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(assemble(args.baseline,args.cache_dir,args.counts,args.output),indent=2))
