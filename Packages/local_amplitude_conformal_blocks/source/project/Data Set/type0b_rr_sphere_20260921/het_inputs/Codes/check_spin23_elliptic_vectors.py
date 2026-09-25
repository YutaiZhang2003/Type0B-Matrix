"""Vector amplitude nome-order check; no amplitude ansatz enters this audit."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import spin23_singlet_amplitudes as s
import ns_elliptic_conversion as e
from check_spin23_elliptic_production import CASES, pair


ROOT=Path(__file__).resolve().parent.parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',nargs='+',choices=tuple(CASES),default=list(CASES))
    parser.add_argument('--output',type=Path,default=ROOT/'Data Set/validation/spin23_elliptic_vector_orders.json')
    args=parser.parse_args()
    options=dict(p_nodes=(16,96,32),momentum_scheme='threshold_weighted',series_parameter='elliptic_nome',
                 epsilon0=.3,epsilon1=.24,theta_orders=(24,24,96),radial_order=36,disk_total_order=28,
                 lens_radial_order=32,lens_angular_order=96)
    files=['Codes/heterotic_so23_1to3_vvvv_fit_bundle/'+name for name in
           ('heterotic_so23_1to3.py','heterotic_so23_1to3_fast.py','ns_elliptic_conversion.py','liouville_momentum_quadrature.py')]
    report={**e.representation_metadata('elliptic_nome',9),'settings':options,'cases':{},
            'scope':'order-only vector test at two complex-energy points; not a total error bound',
            'source_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}}
    for label in args.cases:
        case={'energies':list(map(pair,CASES[label])),'orders':{}}
        caches=dict(structure_cache={},block_cache={},elliptic_cache={},direct_series_cache={})
        for n in (7,9):
            started=time.perf_counter(); diagnostics={}
            values=s.fast.vector_amplitude_coefficients_regularized(CASES[label],q_order=n,diagnostics=diagnostics,**options,**caches)
            if not np.all(np.isfinite(values)): raise ArithmeticError('nonfinite vector amplitude')
            case['orders'][str(n)]={'ABC':list(map(pair,values)),'seconds':time.perf_counter()-started,'diagnostics':diagnostics}
            print(json.dumps({'case':label,'order':n,'ABC':list(map(pair,values))}),flush=True)
        old,new=(np.array([complex(*v) for v in case['orders'][str(n)]['ABC']]) for n in (7,9))
        case['7_to_9_relative']=(abs(new-old)/np.maximum(np.maximum(abs(new),abs(old)),1e-300)).tolist()
        report['cases'][label]=case
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('Saved '+str(args.output),flush=True)


if __name__=='__main__':
    main()
