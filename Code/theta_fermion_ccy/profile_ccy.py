"""Performance profiling only: no coefficient comparison or validation.

Profile one actual punctured CCY series with the production benchmark
weights. The remaining weighted descendant cutoff is configurable.
"""
from pathlib import Path
import argparse
import cProfile
import hashlib
import importlib.util
import json
import pstats
import time
import mpmath as mp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--module', type=Path, default=Path(__file__).with_name('punctured_ccy.py'))
    parser.add_argument('--cutoff', type=int, default=16)
    parser.add_argument('--dps', type=int, default=100)
    parser.add_argument('--json', type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('profiled_punctured_ccy', args.module)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    mp.mp.dps = args.dps
    b = mp.mpf(7)/5
    q = b+1/b
    momenta = (mp.mpf(11)/23, mp.mpf(13)/29, mp.mpf(17)/31)
    labels = (mp.mpf(-1)/2, mp.mpf(-1)/4, mp.mpf(1)/4, mp.mpf(-1)/4)
    edge_momenta = (momenta[0], momenta[1], momenta[1], momenta[2])
    weights = tuple((q*q/4-(p+2*n*b)**2)/(2*(1-b*b))
                    for p,n in zip(edge_momenta,labels))
    central = 1+3*q*q/(1-b*b)
    external = -(1+2*b*b)/(2*(1-b*b))
    engine = module.PuncturedCCY(central, weights, external, dps=args.dps)
    indices = module.downward_indices(args.cutoff, degree_weights=(2,1,1,2))
    profiler = cProfile.Profile()
    started = time.perf_counter()
    profiler.enable()
    result = engine.reduced_series(indices=indices)
    profiler.disable()
    elapsed = time.perf_counter()-started
    stats = pstats.Stats(profiler)
    functions = []
    for (filename,line,name),(primitive,calls,self_seconds,cumulative,callers) in stats.stats.items():
        if Path(filename).name == 'punctured_ccy.py':
            functions.append(dict(function=name,line=line,primitive_calls=primitive,
                                  calls=calls,self_seconds=self_seconds,cumulative_seconds=cumulative))
    report = dict(status='profiled_without_crosschecks',module=str(args.module.resolve()),
                  source_sha256=hashlib.sha256(args.module.read_bytes()).hexdigest(),
                  remaining_weighted_descendant_cutoff=args.cutoff,dps=args.dps,
                  labels=['-1/2','-1/4','1/4','-1/4'],copy=1,
                  coefficients=len(result),profiled_seconds=elapsed,
                  cache=engine.cache_info(),
                  functions=sorted(functions,key=lambda x:-x['cumulative_seconds']))
    args.json.parent.mkdir(parents=True,exist_ok=True)
    args.json.write_text(json.dumps(report,indent=2)+'\n')
    profiler.dump_stats(str(args.json.with_suffix('.prof')))
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
