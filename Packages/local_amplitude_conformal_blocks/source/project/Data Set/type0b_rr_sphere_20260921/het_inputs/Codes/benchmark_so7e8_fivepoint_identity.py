"""Smeared identity-limit check of the conditional R tube cancellation.

This tests distributions against smooth compactly supported functions, not
only a value at the delta peak. The common identity-leg normalization is
removed by comparing each R-family integral with the NS integral. It does
not by itself prove the full five-point nonchiral sewing formula.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.special import roots_legendre

from spin23_super_liouville_data import ns_structure_constants, rr_ns_structure_constants


def smeared_identity(p,epsilon,nodes,*,method='global_tangent'):
    width=min(.23,.8*p)
    x,w=roots_legendre(nodes)
    if method=='global_tangent':
        theta_max=math.atan(width/epsilon)
        theta=theta_max*x
        deltas=epsilon*np.tan(theta)
        weights=theta_max*w*epsilon/np.cos(theta)**2
    elif method=='geometric_panels':
        positive=[epsilon]
        while 4*positive[-1]<width:
            positive.append(4*positive[-1])
        positive.append(width)
        bounds=[-v for v in reversed(positive)]+positive
        deltas=np.concatenate([(a+b)/2+(b-a)*x/2 for a,b in zip(bounds,bounds[1:])])
        weights=np.concatenate([(b-a)*w/2 for a,b in zip(bounds,bounds[1:])])
    else:
        raise ValueError('unknown identity quadrature')
    bump=np.exp(1-1/(1-(deltas/width)**2))
    # Three independent smooth test functions, all supported in the same
    # compact interval. The odd contribution is not discarded by symmetry.
    tests=np.array([bump,bump*(1+.7*deltas),bump*np.exp(-(p+deltas)**2)])
    values=[]
    for delta in deltas:
        q=float(p+delta)
        c=ns_structure_constants(p,q,1j*(1-epsilon),precision=50)[0]
        e,o=rr_ns_structure_constants(p,q,1j*(1-epsilon),precision=50)
        values.append((c,(e+o)/2,(e-o)/2))
    integrals=(tests*weights)@np.asarray(values)
    return integrals[:,1:]/integrals[:,0,None]


def run(*,adaptive=False):
    started=time.perf_counter()
    rows=[]
    for p in (.21,.7,1.1):
        for epsilon in ((.001,.0001,.00001) if adaptive else (.01,.001,.0001)):
            previous=None
            for nodes in ((16,24) if adaptive else (128,256)):
                ratios=smeared_identity(p,epsilon,nodes,method='geometric_panels' if adaptive else 'global_tangent')
                rows.append(dict(P=p,epsilon=epsilon,nodes=nodes,
                    R_family_over_NS=np.stack((ratios.real,ratios.imag),axis=-1).tolist(),
                    maximum_distance_from_half=float(np.max(abs(ratios-.5))),
                    quadrature_change=None if previous is None else float(np.max(abs(ratios-previous)))))
                previous=ratios
            print('identity',p,epsilon,'ratio error',rows[-1]['maximum_distance_from_half'],
                  'quadrature shift',rows[-1]['quadrature_change'],flush=True)
    final=[r for r in rows if r['epsilon']==(.00001 if adaptive else .0001) and r['nodes']==(24 if adaptive else 256)]
    passed=all(r['maximum_distance_from_half']<1e-4 and r['quadrature_change']<1e-7 for r in final)
    sources={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
             for name in (Path(__file__).name,'spin23_super_liouville_data.py','so7e8_ramond_metric.py')}
    return dict(status='passed' if passed else 'not_passed',rows=rows,sources=sources,
                quadrature='geometric panels in P-prime minus P; nodes are per panel' if adaptive else 'single global tangent rule',
                seconds=time.perf_counter()-started,
                tested='R identity kernel / NS identity kernel -> 1/2 after smearing',
                inference='conditional inverse R pairing 2 cancels the extra identity kernel; 4 -> 2',
                full_fivepoint_crossing_certified=False,physical_amplitude=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--adaptive',action='store_true')
    args=parser.parse_args()
    result=run(adaptive=args.adaptive)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(result['status'],flush=True)
