"""Separate cold recursive preparation from warm geometry throughput."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.stats import qmc

from spin23_genus1_coefficient_bank import prepare_node
from spin23_genus1_banked_geometry import GeometryBatch,evaluate_density
from spin23_genus1_banked_integral import BankedConfig,map_geometry,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cutoff',type=int,default=8)
    parser.add_argument('--samples',type=int,default=1024)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--bank-output',type=Path)
    args=parser.parse_args()
    before=time.perf_counter()
    bank=prepare_node((.7,.9,1.1),(.25,.35),cutoff=args.cutoff)
    preparation=time.perf_counter()-before
    if args.bank_output: bank.save(args.bank_output)
    cfg=BankedConfig(samples=args.samples)
    grid=qmc.Sobol(6,scramble=True,seed=cfg.seed).random_base2(args.samples.bit_length()-1)
    tau,z,swap,weight,accepted,*_=map_geometry(grid,cfg)
    # One fixed spectral node, same ordered energy assignment throughout:
    # a throughput benchmark, not an integrated-amplitude estimate.
    cuts=tuple(sorted(set((max(0,args.cutoff-2),args.cutoff))))
    chunks=[];free_seconds=0.;contraction_seconds=0.
    for start in range(0,len(tau),64):
        indices=np.flatnonzero(accepted[start:start+64])+start
        if not len(indices):continue
        before=time.perf_counter();geometry=GeometryBatch.build(tau[indices],z[indices])
        free_seconds+=time.perf_counter()-before
        before=time.perf_counter();chunks.append(evaluate_density(bank,geometry,cuts))
        contraction_seconds+=time.perf_counter()-before
    values=np.concatenate(chunks)
    result=dict(scope='one energy and one fixed momentum triple; all 3 spins and 4 PCO components',
        cutoff_twice=args.cutoff,compared_twice_levels=cuts,accepted_geometries=int(accepted.sum()),
        preparation_seconds=preparation,free_geometry_seconds=free_seconds,
        bank_contraction_and_assembly_seconds=contraction_seconds,
        warm_seconds_per_geometry=(free_seconds+contraction_seconds)/accepted.sum(),
        all_values_finite=bool(np.isfinite(values).all()),ramond_checks=bank.metadata['ramond_checks'])
    write_json(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k!='ramond_checks'},indent=2))


if __name__=='__main__':main()
