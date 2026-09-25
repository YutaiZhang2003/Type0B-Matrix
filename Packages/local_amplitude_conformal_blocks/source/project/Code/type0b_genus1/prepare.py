"""Prepare new Type 0B chiral banks with current NS recursion/constants.

R uses the explicitly preserved and validated PBW oracle. Neither the old
checkout pin nor any heterotic integrand is imported.
"""
from itertools import product
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import mpmath as mp
import numpy as np

from even import ROOT,ChiralBank,pairing_phase
from ns_recursion import FastMixedNSNecklaceCRecursion,MixedNSNecklaceCRecursion
from ramond_descendants import chiral_table
from super_liouville_structure_constants import (
    ns_structure_constant,ns_tilde_structure_constant,rr_ns_structure_constants,
)


def spectral_rule(order=8,pmax=4,power=1.25):
    edges=[]
    for edge in range(2):
        x,w=np.polynomial.legendre.leggauss(order+edge);t=(x+1)/2;a=power+edge*1e-6
        edges.append(list(zip(pmax*t**a,w/2*pmax*a*t**(a-1)/math.pi)))
    nodes=list(product(*edges))
    return np.array([[p for p,w in row] for row in nodes]),np.array([math.prod(w for p,w in row) for row in nodes])


def ns_coefficients(p,omega,word,forms,cutoff,long_ground=False):
    options=dict(central_charge=13.5,internal_weights=tuple((1+x*x)/2 for x in p),
        external_weights=((1+omega*omega)/2,)*2,vertex_sectors=forms,external_descendants=word)
    if cutoff>=15:
        # The binary recurrence can silently lose its pole cancellations at
        # physical level 7.5 near P=0. Raising the block order must therefore
        # also control arithmetic precision, even when no pole guard raises.
        maximum=(cutoff,0 if long_ground else cutoff)
        def precise(dps):
            try:
                return MixedNSNecklaceCRecursion(**options,working_precision=dps).coefficient_table(maximum,cutoff)
            except ZeroDivisionError:
                def average(radius,count):
                    result={}
                    with mp.workdps(dps+30):
                        for j in range(count):
                            delta=mp.mpf(radius)*mp.exp(2j*mp.pi*(mp.mpf(j)+mp.mpf('.5'))/count)
                            h=tuple(mp.mpc(x)+(i+1)*delta for i,x in enumerate(options['internal_weights']))
                            rec=MixedNSNecklaceCRecursion(**dict(options,internal_weights=h,working_precision=dps+30))
                            for k,v in rec.coefficient_table(maximum,cutoff).items():result[k]=result.get(k,0j)+v/count
                    return result
                a,b,c=average('.001',8),average('.002',8),average('.001',16)
                error=max(max(abs(a[k]-c[k]),abs(b[k]-c[k]))/max(1.,abs(c[k])) for k in c)
                if error>1e-10:raise ArithmeticError(f'NS contour precision has not converged: {error}')
                return c
        previous=precise(60)
        for dps in (80,100,120):
            current=precise(dps)
            error=max(abs(current[k]-previous[k])/max(1.,abs(current[k])) for k in current)
            if error<=1e-11:return current
            previous=current
        raise ArithmeticError(f'NS coefficient precision has not converged through {dps} digits: {error}')
    try:
        return FastMixedNSNecklaceCRecursion(**options).coefficient_table((cutoff,0 if long_ground else cutoff),cutoff)
    except (ZeroDivisionError,ArithmeticError):
        def average(radius):
            result={}
            with mp.workdps(80):
                for j in range(8):
                    d=mp.mpf(radius)*mp.exp(2j*mp.pi*(mp.mpf(j)+mp.mpf('.5'))/8)
                    h=tuple(mp.mpc(x)+(i+1)*d for i,x in enumerate(options['internal_weights']))
                    r=MixedNSNecklaceCRecursion(**dict(options,internal_weights=h,working_precision=80))
                    for k,v in r.coefficient_table((cutoff,0 if long_ground else cutoff),cutoff).items():result[k]=result.get(k,0j)+v/8
            return result
        a,b=average('.001'),average('.002')
        error=max(abs(v-b[k]) for k,v in a.items())/max(1,max(abs(v) for v in a.values()))
        if error>2e-8:raise ArithmeticError(f'NS confluent limit discrepancy {error}')
        return a


def node(p,omega,cutoff=8,precision=24,long_ground=False):
    p=tuple(map(float,p));omega=complex(omega)
    nl=np.array([k for k in product(range(cutoff+1),repeat=2) if sum(k)<=cutoff and (not long_ground or k[1]==0)],int)
    rl=np.array([k for k in product(range(0,cutoff+1,2),repeat=2) if sum(k)<=cutoff and (not long_ground or k[1]==0)],int)
    n=np.zeros((2,2,len(nl)),complex);r=np.zeros((2,2,4,len(rl)),complex)
    nw=np.zeros((2,2),complex);rw=np.zeros((2,2,4),complex)
    forms=((0,0),(1,1));signs=tuple(product((1,-1),repeat=2))
    nc=[(ns_structure_constant(p[v-1],omega,p[v],precision),
         ns_tilde_structure_constant(p[v-1],omega,p[v],precision)) for v in range(2)]
    rc=[rr_ns_structure_constants(p[v-1],p[v],omega,precision) for v in range(2)]
    for wi,word in enumerate(((1,1),(0,0))):
        for fi,f in enumerate(forms):
            values=ns_coefficients(p,omega,word,f,cutoff,long_ground=long_ground)
            n[wi,fi]=[values[tuple(k)] for k in nl]
            nw[wi,fi]=pairing_phase(f,word,(0,0))*math.prod(c[k] for c,k in zip(nc,f))
            for si,s in enumerate(signs):
                rw[wi,fi,si]=pairing_phase(f,word,(0,0),ramond=True)*math.prod(
                    c[0 if k==1 else 1]/2 for c,k in zip(rc,s))
        lv,r[wi],ff,ss=chiral_table(p,omega,word,cutoff=cutoff,long_ground=long_ground)
        assert np.array_equal(lv,rl) and ff==forms and ss==signs
    return dict(ns_levels=nl,r_levels=rl,ns=n,r=r,ns_weights=nw,r_weights=rw)


def prepare(omega,output,*,order=8,cutoff=8,pmax=4,source_grid=None,collision_grid=None,long_ground=False):
    if source_grid and collision_grid:raise ValueError('choose one source grid')
    if source_grid:
        old=ChiralBank.load(source_grid);ps,ws=old.arrays['momenta'],old.arrays['weights']
    elif collision_grid:
        from collision import CollisionBank
        old=CollisionBank.load(collision_grid);ps,ws=old.arrays['momenta'],old.arrays['weights']
    else:ps,ws=spectral_rule(order,pmax)
    started=time.time();nodes=[]
    for i,p in enumerate(ps):
        nodes.append(node(p,omega,cutoff,long_ground=long_ground))
        if i%16==0:print(json.dumps(dict(node=i,total=len(ps),seconds=time.time()-started)),flush=True)
    a={k:np.array([r[k] for r in nodes]) for k in ('ns','r','ns_weights','r_weights')}
    a.update(momenta=ps,weights=ws,ns_levels=nodes[0]['ns_levels'],r_levels=nodes[0]['r_levels'])
    meta=dict(schema='type0b-even-chiral-v1',energy=[complex(omega).real,complex(omega).imag],
        cutoff=cutoff,order=order,pmax=pmax,channel='leading-cusp' if long_ground else 'necklace',seconds=time.time()-started,
        grid_source=str(source_grid or collision_grid) if source_grid or collision_grid else None,
        grid_source_sha256=old.file_sha256 if source_grid or collision_grid else None,
        spectral_rule=old.metadata.get('spectral_rule') if source_grid or collision_grid else dict(kind='endpoint_power',order=order,pmax=pmax,power=1.25),
        array_sha256={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in a.items()})
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,metadata=json.dumps(meta),**a)
    print(json.dumps(dict(saved=str(output),seconds=time.time()-started)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--omega',type=complex,default=.25j);p.add_argument('--order',type=int,default=8)
    p.add_argument('--cutoff',type=int,default=8);p.add_argument('--pmax',type=float,default=4)
    p.add_argument('--source-grid');p.add_argument('--collision-grid');p.add_argument('--output',required=True)
    p.add_argument('--long-ground',action='store_true')
    a=p.parse_args();prepare(a.omega,a.output,order=a.order,cutoff=a.cutoff,pmax=a.pmax,source_grid=a.source_grid,collision_grid=a.collision_grid,long_ground=a.long_ground)
