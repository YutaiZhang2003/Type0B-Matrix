"""Collision/cylinder overlap at fixed long-tube momentum.

The two sides integrate DIFFERENT intermediate momenta: the necklace short
edge and the collision bridge. Equality is a nontrivial factorization test,
not a comparison of individual momentum nodes. Leading bridge families omit
higher collision descendants; no normalization is fitted to the overlap.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from scipy.special import roots_genlaguerre

from audit_spin23_genus1_vv_boundaries import encoded
from spin23_genus1_vv_tail import leading_cusp_density
from spin23_genus1_vv_tail_bank import prepare_tail_bank
from spin23_genus1_onepoint_recursion import ns_onepoint
from spin23_super_liouville_data import ns_structure_constants
from spin23_genus1_vv_conventions import CONVENTION


def line_rule(order,scale=math.pi):
    x,w=roots_genlaguerre(order,-.5)
    return np.sqrt(x/scale),np.exp(np.log(w)+x)/(2*math.pi*math.sqrt(scale))


def collision_coefficients(bridge,loop,omega):
    h=(1+bridge**2)/2
    sphere=np.array(ns_structure_constants(omega,omega,bridge,precision=24))
    handle=np.array(ns_structure_constants(loop,bridge,loop,precision=24))
    ns,_=ns_onepoint(loop,bridge,1)
    value=sphere*handle/np.array([1.,(1+bridge**2)**2])
    return value*ns[:,0]*(ns[:,1]+23*ns[:,0])*(2*math.pi)**(2*h+np.arange(2))


def audit(order=16,q_order=8,loop=.7,omega=.25j):
    omega=complex(omega);height=4.
    p,w=line_rule(order)
    coupling=np.empty((order,2,2),complex)
    for i,short in enumerate(p):
        constants=np.array(ns_structure_constants(loop,omega,float(short),precision=24))
        coupling[i]=np.array([[1.,-1.],[1.,1.]])*constants[None]**2
    inputs=SimpleNamespace(metadata=dict(energy=[omega.real,omega.imag],bank_id=None),
        momenta=np.column_stack((p,np.full(order,loop))),weights=w,ns_weights=coupling)
    tail=prepare_tail_bank(inputs,q_order)
    # A different quadrature suited to the collision bridge; keeping it fixed
    # across radii avoids constructing coefficients during geometry evaluation.
    bridge,bw=line_rule(2*order)
    coeff=np.array([collision_coefficients(float(b),loop,omega) for b in bridge])
    h=(1+bridge**2)/2
    # Holomorphic cylinder-OPE and antiholomorphic spectator poles both
    # change sign under delta=-z; their product has positive orientation.
    pref=math.exp(-2*math.pi*loop**2*height)/(8*math.sqrt(8*math.pi**2*height))
    rows=[]
    for radius in (.12,.08,.05,.03,.02,.01):
        radial=np.column_stack(((1-h)*radius**(2*h-4),(h+.5)*radius**(2*h-3)))
        ope=pref*np.einsum('mf,mf,m->f',coeff,radial,bw)
        powers=np.column_stack((radius**(2*h-4),radius**(2*h-3)))
        gg=pref*np.einsum('mf,mf,m->',coeff,powers*np.column_stack((1+omega**2-h,h+.5+omega**2)),bw)
        pp=-omega**2*pref*np.einsum('mf,mf,m->',coeff,powers,bw)
        ope_pco=np.array([gg,pp])
        assert np.allclose(ope.sum(),ope_pco.sum(),rtol=1e-12,atol=1e-15)
        for angle in (math.pi/2,math.pi/4,3*math.pi/4):
            z=radius*np.exp(1j*angle)
            orders=[k for k in (4,6,8,10,12) if k<=q_order]
            values=np.array([leading_cusp_density(tail,z,height,k) for k in orders])
            rows.append(dict(radius=radius,angle=angle,necklace_pco=encoded(values),
                retained_q_orders=orders,collision_families=encoded(ope),
                collision_pco=encoded(ope_pco),pco_ratios=encoded(values[-1]/ope_pco),
                total_ratio=encoded(values[-1].sum()/ope.sum()),
                relative_q_convergence=(abs(values[-1]-values[-2])/
                    np.maximum(abs(values[-1]),1e-300)).tolist()))
    return dict(short_order=order,bridge_order=2*order,q_order=q_order,
        energy=[omega.real,omega.imag],loop_momentum=loop,height=height,rows=rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--order',type=int,default=16)
    parser.add_argument('--q-order',type=int,choices=(6,8,10,12),default=8)
    parser.add_argument('--loop',type=float,default=.7)
    parser.add_argument('--omega',type=complex,default=.25j)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();started=time.perf_counter()
    results=audit(args.order,args.q_order,args.loop,args.omega)
    names=('Codes/audit_spin23_genus1_vv_collision.py','Codes/spin23_genus1_vv_ope.py',
        'Codes/spin23_genus1_vv_tail_bank.py','Codes/spin23_genus1_vv_tail.py',
        'Codes/spin23_genus1_onepoint_recursion.py','Codes/spin23_genus1_vv_conventions.py')
    report=dict(schema='spin23-vv-collision-overlap-audit-v1',results=results,
        seconds=time.perf_counter()-started,physical_amplitude_certified=False,
        physical_component_convention=CONVENTION,
        source_hashes={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in names})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    for row in results['rows']:
        if abs(row['angle']-math.pi/2)<1e-12:
            print(row['radius'],row['total_ratio'],row['relative_q_convergence'],flush=True)
    print('seconds',report['seconds'],flush=True)


if __name__=='__main__':main()
