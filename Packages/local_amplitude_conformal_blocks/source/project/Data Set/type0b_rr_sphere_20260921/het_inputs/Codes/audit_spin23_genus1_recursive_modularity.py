"""Run the corrected SVV modular check with recursive blocks in both sectors."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from spin23_genus1_corrected_integrand import evaluate_svv_integrand
from spin23_genus1_threepoint_integral import complex_json
from spin23_type0b_reference import REFERENCE_COMMIT
from audit_spin23_genus1_threepoint_modularity import modular_ratio_record


def audit(*,cutoff,order,workers,geometry='balanced',orbit='NS_tilde_R',rectangle=False,p=.25,q=.35):
    if geometry=='balanced':tau=1j;points=(0j,.249+1j/3,.498+2j/3)
    elif geometry=='original':tau=.1+1.25j;points=(0j,.18+.36j,.39+.81j)
    else:raise ValueError('unknown geometry')
    tau2=-1/tau;points2=(0j,points[2]/tau+tau2,points[1]/tau+tau2)
    spins=('NS','NS') if orbit=='NS_NS' else ('NS_tilde','R')
    records=[];values=[];start=time.perf_counter()
    for spin,t,z,energy in zip(spins,(tau,tau2),(points,points2),((p,q),(q,p))):
        before=time.perf_counter()
        result=evaluate_svv_integrand(*energy,tau=t,points=z,maximum_twice_levels=cutoff,
            maximum_total_twice_level=None if rectangle else cutoff,
            quadrature_order=order,workers=workers,precision=24,spin_labels=(spin,),block_backend='recursion')
        components=[-.5j*x.value for x in result.fixed_spin[spin].components]
        values.append(components)
        records.append(dict(spin=spin,tau=complex_json(t),points=complex_json(z),
            components=complex_json(components),plumbing=complex_json(result.coordinates.plumbing_parameters),
            runtime_seconds=time.perf_counter()-before))
        print(spin,'complete',time.perf_counter()-start,flush=True)
    factor=abs(tau)**8
    files=('Codes/audit_spin23_genus1_recursive_modularity.py','Codes/spin23_genus1_ns_c_recursion.py',
        'Codes/spin23_genus1_branch_recursion.py','Codes/spin23_genus1_virasoro_necklace.py',
        'Codes/spin23_genus1_virasoro_collision.py',
        'Codes/spin23_genus1_recursive_sewing.py','Codes/spin23_genus1_corrected_integrand.py','Codes/spin23_type0b_reference.py')
    root=Path(__file__).resolve().parents[1]
    return dict(status='recursive fixed-geometry modular diagnostic; not an integrated string amplitude',
        assembly='canonical corrected SVV',block_backend='recursion',reference_commit=REFERENCE_COMMIT,
        maximum_twice_level=cutoff,maximum_total_twice_level=None if rectangle else cutoff,
        cutoff_scheme='rectangle' if rectangle else 'total',laguerre_order=order,workers=workers,
        geometry=geometry,orbit=orbit,energies=[p,q],evaluations=records,
        individual_ratios=[modular_ratio_record(values[1][i],factor*values[0][j]) for i,j in ((0,0),(1,2),(2,1),(3,3))],
        total_ratio=modular_ratio_record(sum(values[1]),factor*sum(values[0])),
        runtime_seconds=time.perf_counter()-start,
        source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cutoff',type=int,default=8)
    parser.add_argument('--order',type=int,default=3)
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--geometry',choices=('original','balanced'),default='balanced')
    parser.add_argument('--orbit',choices=('NS_NS','NS_tilde_R'),default='NS_tilde_R')
    parser.add_argument('--rectangle',action='store_true')
    parser.add_argument('--p',type=float,default=.25);parser.add_argument('--q',type=float,default=.35)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('choose a new output artifact')
    result=audit(**{k:v for k,v in vars(args).items() if k!='output'})
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps({key:result[key] for key in ('individual_ratios','total_ratio','runtime_seconds')},indent=2))
