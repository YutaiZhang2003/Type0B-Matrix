#!/usr/bin/env python3
"""One fixed-z momentum-resolution diagnostic; never changes production settings."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from benchmark_so7e8_four_ramond_convergence import MOMENTA
from heterotic_so23_1to3_vvvv_fit_bundle.liouville_momentum_quadrature import threshold_weighted_rule
from literature_component_blocks import SECTORS, crossing_phase
from literature_self_dual_correlator import AnalyticSewing, SelfDualConstants
from so7e8_literature_campaign import atomic_json, pairs


def run(nodes=32, order=5):
    started = time.monotonic()
    z = .5+.1j
    m = MOMENTA
    definitions = dict(
        mixed=((m[0],m[3],m[2],m[1]), ((0,0,(0,0),(0,0)), (1,0,(1,0),(0,0)))),
        rrrr=(m, ((0,0,0,0), (1,0,1,0))),
    )
    constants = SelfDualConstants(70)
    rows = []
    for name,(p,cases) in definitions.items():
        values = []
        for channel in ("s","t"):
            family = "rrrr" if name == "rrrr" else ("mixed_ns" if channel == "s" else "mixed_r")
            pp,zz,exts = (p,z,cases) if channel == "s" else (
                (p[2],p[1],p[0],p[3]),1-z,tuple((e[2],e[1],e[0],e[3]) for e in cases))
            rule = threshold_weighted_rule(nodes,dict(beta=0 if family == "mixed_r" else 2))
            total = np.zeros(len(cases),complex)
            for P,w in zip(rule.momenta,rule.weights):
                sewing = AnalyticSewing(family,pp,P,2*order,constants=constants)
                total += w*np.array([sewing.value(zz,e) for e in exts])
            values.append(total)
            print(name,channel,"nodes",nodes,"seconds",time.monotonic()-started,flush=True)
        sectors = SECTORS["rrrr" if name == "rrrr" else "mixed_ns"]
        for i,e in enumerate(cases):
            x,y = values[0][i],crossing_phase(sectors,e)*values[1][i]
            rows.append(dict(observable=name,external=e,lhs=pairs(x),rhs=pairs(y),
                             relative_error=float(abs(x-y)/max(abs(x),abs(y),1e-300))))
    source = Path(__file__).parent
    return dict(scope="focused fixed-z resolution diagnostic; not the prescribed production rule",
                production_node_count_unchanged=8,momentum_nodes=nodes,physical_block_order=order,
                z=pairs(z),momenta=pairs(m),epsilon=.01,rows=rows,
                integration="internal P only; no moduli integration",
                seconds=time.monotonic()-started,
                source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                    [source/n for n in (Path(__file__).name,"literature_double_virasoro.py",
                     "literature_self_dual_blocks.py","literature_self_dual_correlator.py",
                     "literature_component_blocks.py")]})


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes",type=int,default=32)
    parser.add_argument("--order",type=int,default=5)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=run(args.nodes,args.order)
    atomic_json(args.output,result)
    print(json.dumps(result["rows"],indent=2))
