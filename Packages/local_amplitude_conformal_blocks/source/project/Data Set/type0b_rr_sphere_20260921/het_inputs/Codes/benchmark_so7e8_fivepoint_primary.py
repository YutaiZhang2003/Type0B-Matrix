"""Bounded five-point CFT-only numerical gate; never an amplitude driver."""
import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED

import numpy as np

from heterotic_so23_1to3_vvvv_fit_bundle.liouville_momentum_quadrature import threshold_weighted_rule
from so7e8_fivepoint_blocks import EDGES
from so7e8_fivepoint_primary import primary_density, REAL_MOMENTA, COMPLEX_MOMENTA, POINTS
from so7e8_ramond_metric import CONDITIONAL_RAMOND_METRIC, metric_metadata
from benchmark_so7e8_primary_crossing_identity import source_hashes as fourpoint_source_hashes


def pairs(value):
    a=np.asarray(value,complex)
    return np.stack((a.real,a.imag),axis=-1).tolist()


def unpairs(value):
    a=np.asarray(value,float)
    return a[...,0]+1j*a[...,1]


def source_hashes():
    root=Path(__file__).parent
    hashes=fourpoint_source_hashes()
    for name in (Path(__file__).name,"so7e8_fivepoint_blocks.py","so7e8_fivepoint_primary.py",
                 "so7e8_fivepoint_uniformization.py","virasoro_fivepoint_c_recursion.py","so7e8_ramond_metric.py",
                 "so7e8_fivepoint_collar.py","so7e8_type0b_boundary.py"):
        hashes[name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
    from so7e8_type0b_boundary import source_manifest
    hashes.update({'Type0B/'+k:v for k,v in source_manifest()['sha256'].items()})
    return hashes


def _node_entry(job):
    identity,signature,cache,i,j,pa,pb,momenta=job
    started=time.perf_counter()
    key=(pa.hex(),pb.hex())
    target=Path(cache)/(hashlib.sha256('|'.join(key).encode()).hexdigest()+'.json')
    if target.exists():
        entry=json.loads(target.read_text())
        if entry['signature']!=signature or tuple(entry['momenta_hex'])!=key:
            raise ValueError('five-point cache identity mismatch')
    else:
        values,diag=primary_density((pa,pb),channel=identity['channel'],
            maximum_twice_levels=tuple(identity['maximum_twice_levels']),
            momenta=momenta,convention=CONDITIONAL_RAMOND_METRIC,contour=identity['contour'],
            resummation=identity['resummation'])
        entry=dict(signature=signature,momenta_hex=key,
                   values={f'{a},{b}':pairs(v) for (a,b),v in values.items()},
                   raw_sewing=pairs(diag['raw_sewing']),radius_defect=diag['maximum_radius_defect'])
        target.write_text(json.dumps(entry,indent=2,allow_nan=False)+'\n')
    return i,j,entry,time.perf_counter()-started


def _node_results(jobs,workers):
    if workers==1:
        yield from map(_node_entry,jobs)
        return
    executor=ProcessPoolExecutor(max_workers=workers)
    pending={}
    iterator=iter(jobs)
    try:
        for _ in range(workers):
            job=next(iterator,None)
            if job is not None:
                pending[executor.submit(_node_entry,job)]=job
        while pending:
            completed,_=wait(pending,return_when=FIRST_COMPLETED)
            for future in completed:
                job=pending.pop(future)
                try:
                    yield future.result()
                except Exception as exc:
                    raise ArithmeticError(f'node {job[3:7]} failed: {exc}') from exc
                job=next(iterator,None)
                if job is not None:
                    pending[executor.submit(_node_entry,job)]=job
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True,cancel_futures=True)


def run(*,channel,nodes,maximum,output,cache_dir,momenta=REAL_MOMENTA,options=None,
        contour=None,workers=1,resummation='type0b_joint'):
    if type(workers) is not int or not 1<=workers<=8:
        raise ValueError('workers must be an integer from one to eight')
    if type(maximum) is not int or maximum<2 or maximum%2:
        raise ValueError('maximum must be a positive even twice-level cutoff, at least two')
    if len(nodes)!=2:
        raise ValueError('two quadrature sizes are required')
    started=time.perf_counter()
    if resummation not in ('type0b_joint','independent_nome_diagnostic'):
        raise ValueError('unknown five-point resummation')
    identity=dict(schema=2,channel=channel,maximum_twice_levels=[maximum,maximum],resummation=resummation,
        momenta=pairs(momenta),points=pairs(POINTS),sources=source_hashes(),
        convention=CONDITIONAL_RAMOND_METRIC,contour=dict(radius=.10,check_radius=.12,samples=32,tolerance=1e-7)|dict(contour or {}))
    signature=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    cache=Path(cache_dir)/signature
    cache.mkdir(parents=True,exist_ok=True)
    rules=tuple(threshold_weighted_rule(n,options=dict(beta=2 if sec=="NS" else 0,**(options or {})))
                for n,sec in zip(nodes,EDGES[channel]))
    cuts=tuple(product(range(2,maximum+1,2),repeat=2))
    totals={cut:np.zeros((3,3,len(POINTS)),complex) for cut in cuts}
    raw=np.zeros((3,3,len(POINTS)),complex)
    report=dict(status="running",parameters=identity,signature=signature,
                quadrature=[r.metadata() for r in rules],completed_nodes=0,
                total_nodes=len(rules[0].momenta)*len(rules[1].momenta),
                normalization=metric_metadata(CONDITIONAL_RAMOND_METRIC),
                primary_sewing_extension="candidate awaiting integrated crossing and identity gates",
                physical_amplitude=False,crossing_certified=False,convergence_certified=False,
                maximum_radius_defect=0.,node_timings=[],node_failures=[],workers=workers)
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    def save():
        report['seconds']=time.perf_counter()-started
        report['results']={f'{i},{j}':dict(total=pairs(v.sum(axis=(0,1))),nine_regions=pairs(v))
                           for (i,j),v in totals.items()}
        report['raw_sewing_total']=pairs(raw.sum(axis=(0,1)))
        output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    try:
        jobs=[(identity,signature,str(cache),i,j,float(rules[0].momenta[i]),float(rules[1].momenta[j]),tuple(momenta))
              for i,j in product(range(len(rules[0].momenta)),range(len(rules[1].momenta)))]
        for i,j,entry,elapsed in _node_results(jobs,workers):
            pa,pb=float(rules[0].momenta[i]),float(rules[1].momenta[j])
            weight=rules[0].weights[i]*rules[1].weights[j]/np.pi**2
            ra,rb=int(rules[0].regions[i]),int(rules[1].regions[j])
            for cut in cuts:
                value=unpairs(entry['values'][f'{cut[0]},{cut[1]}'])
                if value.shape!=(len(POINTS),) or not np.all(np.isfinite(value)):
                    raise ArithmeticError('invalid cached density')
                totals[cut][ra,rb]+=weight*value
            raw[ra,rb]+=weight*unpairs(entry['raw_sewing'])
            report['maximum_radius_defect']=max(report['maximum_radius_defect'],entry['radius_defect'])
            report['completed_nodes']+=1
            report['node_timings'].append(elapsed)
            save()
            print(channel,report['completed_nodes'],'/',report['total_nodes'],'P',pa,pb,
                  'seconds',round(report['seconds'],2),flush=True)
    except BaseException as exc:
        report['status']='failed'
        report['failure']=dict(type=type(exc).__name__,message=str(exc))
        # Failed node coordinates are carried in the worker exception; the
        # last completed node must not be misreported as the failing one.
        save()
        raise
    report['status']='complete_unconverged'
    save()
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--channel',choices=('A','B','C'),required=True)
    parser.add_argument('--nodes',type=int,nargs=2,default=(12,12))
    parser.add_argument('--maximum',type=int,default=4)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cache-dir',type=Path,required=True)
    parser.add_argument('--complex-momenta',action='store_true')
    parser.add_argument('--radius',type=float,default=.10)
    parser.add_argument('--check-radius',type=float,default=.12)
    parser.add_argument('--samples',type=int,default=32)
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--real-contour-symmetry',action='store_true')
    parser.add_argument('--resummation',choices=('type0b_joint','independent_nome_diagnostic'),default='type0b_joint')
    args=parser.parse_args()
    run(channel=args.channel,nodes=args.nodes,maximum=args.maximum,output=args.output,cache_dir=args.cache_dir,
        momenta=COMPLEX_MOMENTA if args.complex_momenta else REAL_MOMENTA,workers=args.workers,resummation=args.resummation,
        contour=dict(radius=args.radius,check_radius=args.check_radius,samples=args.samples,
                     contour_symmetry='real_primary' if args.real_contour_symmetry else 'none'))
