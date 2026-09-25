"""Resolve the four-R crossing defect into discrete sewing components.

No fitted coefficient from this diagnostic is automatically applied to an
amplitude. It is a linear bootstrap probe of the component conventions.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time
from itertools import product
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"Codes"))
from benchmark_so7e8_four_ramond_convergence import MOMENTA,FAMILIES
from so7e8_four_ramond_assembly import (
    allowed_four_ramond_parity_sewings,physical_four_ramond_components,
    four_ramond_free_sector_trinion_product,normalized_time_ising_scalar_block,
    spin7_reexpress_pairing_coefficients,
)
from so7e8_four_ramond_elliptic_recursion import (
    general_four_ramond_sld_elliptic_h_series,four_ramond_ground_map,
)
from so7e8_direct_domain import spin7_global_grid
from spin23_singlet_amplitudes import _momentum_quadrature
from spin23_super_liouville_data import rr_ns_chiral_structure_constant

PARITIES=allowed_four_ramond_parity_sewings()
KEYS=tuple((i,sl,sr) for i in range(len(PARITIES)) for sl,sr in product((-1,1),repeat=2))


def current_weights():
    weights=[]
    for i,sl,sr in KEYS:
        q=PARITIES[i]
        total=0j
        for pol in physical_four_ramond_components(FAMILIES):
            h=four_ramond_ground_map(pol.holomorphic_ground_parities,
                component=("even","odd")[q.holomorphic_sld_parity],
                left_structure_sign=sl,right_structure_sign=sr).phase
            a=four_ramond_ground_map(pol.antiholomorphic_ground_parities,
                component=("even","odd")[q.antiholomorphic_sld_parity],
                left_structure_sign=sl,right_structure_sign=sr,chirality="antiholomorphic").phase
            total+=pol.coefficient*h*a*four_ramond_free_sector_trinion_product(
                pol.time_ising_parities,time_channel=q.time_channel,
                spin7_channel=q.spin7_channel,fermion_turning_phase=1)
        weights.append(total)
    return np.asarray(weights)


def columns(momenta,z,order,nodes):
    spin=spin7_global_grid(tuple(z.conjugate()))
    times={c:np.array([normalized_time_ising_scalar_block(c,x) for x in z])
           for c in ("identity","fermion")}
    p1,p2,p3,p4=momenta
    factor=np.exp(-2*p1*p2*np.log(abs(z))-2*p2*p3*np.log(abs(1-z))
                  -.25*(np.log(z)+np.log(1-z)))
    result=np.zeros((len(z),4,len(KEYS)),complex)
    ps,pw=_momentum_quadrature(nodes,4,.03,scheme="infinite_gauss")
    for p,weight in zip(ps,pw):
        blocks={}
        for sl,sr in product((-1,1),repeat=2):
            for parity,chirality in product((0,1),("holomorphic","antiholomorphic")):
                block=general_four_ramond_sld_elliptic_h_series(float(p),
                    external_momenta=momenta,external_ground_parities=(0,0,0,0),
                    maximum_twice_level=order,component=("even","odd")[parity],
                    left_structure_sign=sl,right_structure_sign=sr,
                    chirality=chirality,digits=max(80,4*order+40))
                blocks[sl,sr,parity,chirality]=block.value(z if chirality=="holomorphic" else z.conjugate(),
                    cut_side="upper" if chirality=="holomorphic" else "lower")
        structures={(sl,sr):.25*rr_ns_chiral_structure_constant(p4,p3,float(p),structure_sign=sl,precision=80)
                    *rr_ns_chiral_structure_constant(p2,p1,-float(p),structure_sign=sr,precision=80)
                    for sl,sr in product((-1,1),repeat=2)}
        for j,(i,sl,sr) in enumerate(KEYS):
            q=PARITIES[i]
            scalar=weight/math.pi*structures[sl,sr]*factor*times[q.time_channel]
            scalar*=blocks[sl,sr,q.holomorphic_sld_parity,"holomorphic"]
            scalar*=blocks[sl,sr,q.antiholomorphic_sld_parity,"antiholomorphic"]
            result[:,:,j]+=scalar[:,None]*spin[q.spin7_channel]
    return result


def run(order,nodes,output):
    started=time.perf_counter()
    u=np.array([.37+.23j,.68-.15j,.25-.4j,.71+.4j,.51+.31j,.24+.12j])
    cut=np.array([x+lip*1e-7j for x in (1.3,2.,3.) for lip in (1,-1)])
    original=columns(MOMENTA,np.r_[1/u,cut],order,nodes)
    m=MOMENTA
    swapped=columns((m[0],m[2],m[1],m[3]),u,order,nodes)
    folded=np.empty_like(swapped)
    for iz in range(len(u)):
        for k in range(len(KEYS)):
            folded[iz,:,k]=-spin7_reexpress_pairing_coefficients(swapped[iz,:,k],source_pairing="02|13",target_pairing="01|23")
    exterior=original[:len(u)]/abs(u[:,None,None])**4
    cutcols=original[len(u):]
    constraint=np.concatenate([(exterior-folded).reshape(-1,len(KEYS)),
        (cutcols[::2]-cutcols[1::2]).reshape(-1,len(KEYS))])
    weights=current_weights()
    active=abs(weights)>1e-10
    # Keep two primary-channel coefficients at their calibrated values.
    anchors=[KEYS.index((i,-1,1)) for i in (0,1)]
    reports={}
    for label,selected in (("eight_current_channels",np.flatnonzero(active)),("all_32_channels",np.arange(len(KEYS)))):
        free=[j for j in selected if j not in anchors]
        rhs=-constraint[:,anchors]@weights[anchors]
        solved=np.zeros(len(KEYS),complex);solved[anchors]=weights[anchors]
        solved[free]=np.linalg.lstsq(constraint[:,free],rhs,rcond=1e-10)[0]
        sv=np.linalg.svd(constraint[:,selected],compute_uv=False)
        def pairs(a):return np.stack([a.real,a.imag],axis=-1).tolist()
        reports[label]={"constraint_residual":float(np.linalg.norm(constraint@solved)/np.linalg.norm(exterior@weights)),
            "weights":pairs(solved),"singular_values":sv.tolist()}
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output.with_suffix(".npz"),exterior=exterior,folded=folded,cut=cutcols,
        constraint=constraint,current_weights=weights,keys=np.array(KEYS),u=u,cut_points=cut)
    meta=dict(order=order,nodes=nodes,momenta=[[p.real,p.imag] for p in MOMENTA],keys=KEYS,
        reports=reports,seconds=time.perf_counter()-started)
    output.write_text(json.dumps(meta,indent=2)+"\n")
    print(json.dumps(meta,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--order",type=int,default=7)
    parser.add_argument("--nodes",type=int,default=32)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();run(args.order,args.nodes,args.output)
