"""Refine NS momentum nodes with explicitly audited finite-part retries.

Every attempt retains the original coefficient tolerance. After a failed
contour, two independent successful settings must agree on the full CFT
density. No node is discarded; failure of all settings aborts the run.
Unchanged DensityCache producers allow reuse of already verified nodes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmark_so7e8_momentum_refinement import DensityCache, integrate_rule, relative_change
from heterotic_so23_1to3_vvvv_fit_bundle.liouville_momentum_quadrature import threshold_weighted_rule


CONTOURS = (
    dict(radius=.06,check_radius=.07,samples=32,tolerance=1e-8),
    dict(radius=.08,check_radius=.10,samples=48,tolerance=1e-8),
    dict(radius=.10,check_radius=.12,samples=32,tolerance=1e-8),
    dict(radius=.10,check_radius=.12,samples=48,tolerance=1e-8),
)


class CheckedContours:
    def __init__(self, directory, orders=(4,6,8)):
        self.caches=[DensityCache(directory,channel="crossed",orders=orders,limit_options=opts)
                     for opts in CONTOURS]
        self.orders=self.caches[0].orders
        self.identity=dict(self.caches[0].identity,
            contour_attempts=CONTOURS,
            adaptive_driver_source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        self.retries=[]

    def __call__(self, p):
        first=None
        attempts=[]
        for i,cache in enumerate(self.caches):
            try:
                current=cache(p)
            except ArithmeticError as exc:
                # Retry only the known finite-part accuracy failure, not an
                # unrelated nonfinite CFT value or an algebraic failure.
                if "radius check failed" not in str(exc):
                    raise
                attempts.append(dict(options=cache.limit_options,error=str(exc)))
                continue
            if i==0:
                return current
            attempts.append(dict(options=cache.limit_options,success=True,cache_signature=cache.signature))
            if first is None:
                first=current
                continue
            relative=[]
            for n in self.orders:
                for name in ("F0","F2"):
                    a,b=first[n][name],current[n][name]
                    relative.extend((abs(a-b)/np.maximum(np.maximum(abs(a),abs(b)),1e-300)).tolist())
                    if np.any(abs(a-b)>1e-14+1e-7*np.maximum(abs(a),abs(b))):
                        raise ArithmeticError(f"independent contour CFT checks disagree at P={p}")
            self.retries.append(dict(P=float(p),attempts=attempts,max_relative_density_change=max(relative)))
            return first
        self.retries.append(dict(P=float(p),attempts=attempts,status="failed"))
        raise ArithmeticError(f"no independently verified finite-part retry at P={p}")


def run(cache_dir, output, nodes=(24,48,96)):
    evaluator=CheckedContours(cache_dir)
    report=dict(schema="mixed-momentum-refinement-v1",channel="crossed",parameters=evaluator.identity,
        threshold_options=dict(beta=2.),physical_sewing_certified=False,full_pbw_used=False,
        requested_nodes=list(nodes),rows=[],status="running")
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    def save():
        report["contour_retries"]=evaluator.retries
        report["cache_hits"]=sum(c.hits for c in evaluator.caches)
        report["cache_misses"]=sum(c.misses for c in evaluator.caches)
        output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    save()
    try:
        for count in nodes:
            row=integrate_rule(evaluator,threshold_weighted_rule(count,options={"beta":2.}),
                               evaluator.orders,progress=True)
            row["relative_previous_momentum_change"]={}
            row["relative_previous_block_change"]={}
            previous=report["rows"][-1] if report["rows"] else None
            for i,n in enumerate(evaluator.orders):
                row["relative_previous_momentum_change"][n]={name:None if previous is None else
                    relative_change(row["blocks"][n][name]["value"],previous["blocks"][n][name]["value"])
                    for name in ("F0","F2")}
                row["relative_previous_block_change"][n]={name:None if i==0 else
                    relative_change(row["blocks"][n][name]["value"],row["blocks"][evaluator.orders[i-1]][name]["value"])
                    for name in ("F0","F2")}
            report["rows"].append(row)
            save()
            print(json.dumps(dict(completed_nodes=count,momentum_changes=row["relative_previous_momentum_change"],
                                  block_changes=row["relative_previous_block_change"])),flush=True)
    except BaseException as exc:
        report["status"]="failed"
        report["failure"]=dict(type=type(exc).__name__,message=str(exc))
        save()
        raise
    report["status"]="complete"
    save()
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--nodes",nargs="+",type=int,default=[24,48,96])
    args=parser.parse_args()
    run(args.cache_dir,args.output,args.nodes)
