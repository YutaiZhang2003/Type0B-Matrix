"""Reproduce Euclidean boundary, modular, and matched-integral controls.

No MQM target is imported. No coefficient construction occurs in geometry
integration. Outputs keep imaginary-energy certification separate from the
future analytic continuation of a fitted energy dependence.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np

from spin23_genus1_vv_bank import VVBank,VVGeometry,evaluate_density
from spin23_genus1_vv_factored import FactoredVVBank
from spin23_genus1_vv_ope import OPEBank
from spin23_genus1_vv_tail_bank import TailBank
from spin23_genus1_vv_matching import (CuspCollisionBank,cusp_annulus,puncture_tail,
                                     compact_cusp_subtracted,load_cusp_collision)
from spin23_genus1_vv_conventions import CONVENTION

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data_exports/spin23_genus1_vv'


def encode(v):
    if isinstance(v,np.ndarray):
        return dict(real=v.real.tolist(),imag=v.imag.tolist()) if np.iscomplexobj(v) else v.tolist()
    if isinstance(v,np.generic): return encode(v.item())
    if isinstance(v,complex): return dict(real=v.real,imag=v.imag)
    if isinstance(v,dict): return {k:encode(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [encode(x) for x in v]
    return v


def source_hashes():
    names=('Codes/audit_spin23_genus1_vv_completion.py','Codes/spin23_genus1_vv_matching.py',
           'Codes/spin23_genus1_vv_factored.py','Codes/spin23_genus1_vv_superboundary.py',
           'Codes/spin23_genus1_vv_conventions.py','Codes/spin23_genus1_vv_bank.py',
           'Codes/spin23_genus1_vv_ope.py','Codes/spin23_genus1_vv_tail.py',
           'Codes/spin23_genus1_vv_tail_bank.py','Codes/spin23_genus1_vv_boundaries.py')
    return {n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}


def modular(bank):
    rows=[]
    for tau,z in [(1j,.249+.5j),(.1+1.25j,.23+.55j),(.05+.95j,.3+.4j)]:
        ts=-1/tau;zs=z/tau
        zs-=math.floor(zs.imag/ts.imag)*ts;zs-=math.floor(zs.real)
        g=VVGeometry([tau],[[0,z]]);gs=VVGeometry([ts],[[0,zs]])
        for name,b in [('raw',bank),('factored',FactoredVVBank(bank))]:
            a=evaluate_density(b,g,(4,6,8))[0]
            d=evaluate_density(b,gs,(4,6,8))[0][:,(0,2,1)]*abs(tau)**-6
            rows.append(dict(tau=tau,z=z,method=name,cutoffs=[4,6,8],original=a,
                transformed=d,residual=abs(a-d)/np.maximum(abs(a),abs(d))))
    return rows


def compact_overlap(bank,ope):
    bank=FactoredVVBank(bank);rows=[]
    angles=(np.arange(40)+.5)*math.pi/40
    for tau in (1j,.13+1.2j):
        for radius in (.2,.12,.08,.05,.03):
            z=radius*np.exp(1j*angles)
            value=evaluate_density(bank,VVGeometry(np.full(len(z),tau),
                np.column_stack((np.zeros(len(z)),z))),(4,6,8)).mean(axis=0)
            leading=ope.density_components(1j*radius,tau,ope.metadata['cutoff']).sum(axis=1)
            rows.append(dict(tau=tau,radius=radius,necklace=value,collision=leading,
                total_ratio=value[-1].sum()/leading.sum(),
                spin_ratios=value[-1].sum(axis=-1)/leading.sum(axis=-1),
                relative_last_shell=abs(value[-1]-value[-2])/np.maximum(abs(value[-1]),1e-300)))
    return rows


def integral_worker(arguments):
    bank_path,tail_path,ope_path,cusp_path,start,radius,order=arguments
    bank=FactoredVVBank(VVBank.load(bank_path));tail=TailBank.load(tail_path);ope=OPEBank.load(ope_path)
    cusp=load_cusp_collision(cusp_path,bank.metadata['energy'])
    t=time.perf_counter()
    compact=compact_cusp_subtracted(bank,tail,ope,start=start,radius=radius,order=order)
    # Integrate the resummed leading cusp ONCE from the bottom of the full
    # modular strip. This removes an unnecessary finite-height quadrature
    # of that same term, which previously obscured matching-height tests.
    reference_height=1.+1e-12
    long=puncture_tail(tail,cusp,start=reference_height,radius=radius,order=16,s_max=64.)
    collision_transfer=cusp.disc_tail(start,radius).sum()-cusp.disc_tail(reference_height,radius).sum()
    total=compact['cap']+compact['massive_remainder']+compact['disk']+long['total']+collision_transfer
    return dict(compact=compact,resummed_strip_and_tail=long,
        collision_transfer=collision_transfer,total=total,seconds=time.perf_counter()-t,
        assembly='cap + massive strip remainder + full leading cusp - overlapping collision disk')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('modular','compact-overlap','integral'))
    p.add_argument('--bank',type=Path,default=DATA/'gaussian_rc/gaussian/banks/e0.npz')
    p.add_argument('--tail',type=Path,default=DATA/'tail_spectral_cache/4f9328bc44527d6e1a02323f289ca36900f66890a01d6302f6960159340658ef.npz')
    p.add_argument('--ope',type=Path,default=DATA/'completion/ope_24_16.npz')
    p.add_argument('--cusp',type=Path,default=DATA/'completion/cusp_collision_40_32.npz')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=3)
    p.add_argument('--geometry-orders',default='10,14',help='two increasing integration orders')
    args=p.parse_args();hashes=source_hashes();started=time.perf_counter()
    if args.mode=='modular': rows=modular(VVBank.load(args.bank))
    elif args.mode=='compact-overlap': rows=compact_overlap(VVBank.load(args.bank),OPEBank.load(args.ope))
    else:
        orders=[int(n) for n in args.geometry_orders.split(',')]
        if len(orders)!=2 or not 2<=orders[0]<orders[1]:
            raise ValueError('two increasing geometry orders >=2 required')
        configs=[(3.,.08,orders[0]),(3.,.08,orders[1]),(3.,.12,orders[1]),(4.,.08,orders[1])]
        tasks=[(args.bank,args.tail,args.ope,args.cusp,*c) for c in configs]
        rows=[]
        with ProcessPoolExecutor(args.workers) as pool:
            for row in pool.map(integral_worker,tasks):
                rows.append(row)
                print(json.dumps(encode(dict(compact_controls={k:row['compact'][k]
                    for k in ('start','radius','order')},total=row['total'],seconds=row['seconds']))),flush=True)
    if hashes!=source_hashes(): raise RuntimeError('source changed during the audit')
    inputs=[args.bank]+([args.tail,args.ope,args.cusp] if args.mode=='integral' else
                        [args.ope] if args.mode=='compact-overlap' else [])
    report=dict(schema='spin23-vv-euclidean-completion-audit-v1',mode=args.mode,
        physical_component_convention=CONVENTION,target='imaginary-energy genus-one VV',
        energy=VVBank.load(args.bank).metadata['energy'],
        euclidean_amplitude_certified=False,real_energy_continuation_required=False,
        rows=rows,source_hashes=hashes,input_hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in inputs},
        seconds=time.perf_counter()-started)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(encode(report),indent=2,allow_nan=False)+'\n')
    print('saved',args.output,flush=True)


if __name__=='__main__':main()
