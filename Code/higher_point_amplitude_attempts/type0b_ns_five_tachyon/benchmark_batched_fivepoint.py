#!/usr/bin/env python3
"""Cold/warm face benchmark and scalar agreement at unchanged c-series cutoff."""
import argparse
import json
from pathlib import Path
import resource
import sys
import time

from type0b_ns_five_tachyon import BRYNSFiveTachyonIntegrand, BOUNDARY_FACE_RAISED_ORBITS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--production-momentum',action='store_true')
    parser.add_argument('--store',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    kernel=BRYNSFiveTachyonIntegrand(
        outgoing_energies=(.25+.02j,)*4,central_charge_shift=0,
        global_max_twice_levels=(8,8),global_max_total_twice_level=16,
        momentum_orders=(6,7) if args.production_momentum else (2,3),
        momentum_maximum=2 if args.production_momentum else 1,
        structure_precision=22,block_working_precision=45,batch_c_evaluation=True,
        tensor_cache_mebibytes=1024,auxiliary_cache_limit=65536,c_coefficient_cache_path=args.store)
    results=[]
    try:
        for visit,q in enumerate((.22+.05j,.25-.03j)):
            for index,(ordering,multiplicity) in enumerate(BOUNDARY_FACE_RAISED_ORBITS):
                kw=dict(ordering=ordering,remaining_modulus=q,collar_radius=.01,
                        momentum_refinement_shells=4 if args.production_momentum else 1,
                        momentum_singularity_subtraction=True)
                before=time.perf_counter();value=kernel.boundary_face_finite_part_density(**kw)
                record=dict(visit=visit,orbit=index,ordering=ordering,
                            value=[value.real,value.imag],seconds=time.perf_counter()-before)
                if visit==1 and index in (0,6,15):
                    kernel.batch_c_evaluation=False
                    before=time.perf_counter();reference=kernel.boundary_face_finite_part_density(**kw)
                    record['scalar_seconds']=time.perf_counter()-before
                    record['scaled_error']=abs(value-reference)/max(1,abs(reference))
                    kernel.batch_c_evaluation=True
                    if record['scaled_error']>1e-9:
                        raise ArithmeticError(f'scalar disagreement: {record}')
                record['diagnostics']=kernel.cache_diagnostics();results.append(record)
                print(json.dumps(record),flush=True)
                rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                payload=dict(schema='fivepoint-batch-benchmark-v1',complete=False,
                             production_momentum=args.production_momentum,results=results,
                             peak_rss_mib=rss/(1024**2 if sys.platform=='darwin' else 1024))
                args.output.parent.mkdir(parents=True,exist_ok=True)
                args.output.write_text(json.dumps(payload,indent=2)+'\n')
        payload['complete']=True;args.output.write_text(json.dumps(payload,indent=2)+'\n')
    finally: kernel.close_runtime()

if __name__=='__main__': main()
