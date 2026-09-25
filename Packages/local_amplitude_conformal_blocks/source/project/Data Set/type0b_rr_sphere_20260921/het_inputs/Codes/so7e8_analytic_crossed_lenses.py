#!/usr/bin/env python3
"""Integrate the existing crossed OPE lenses term by term, including inversion.

The lens is 0<|w|<R, |1-w|<1. Its exact angular aperture is
(-acos(r/2), acos(r/2)). No domain, momentum rule, block, or branch changes.
"""
from functools import lru_cache
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import numpy as np


@lru_cache(None)
def aperture_coefficients(spin, order=32):
    """Taylor coefficients of integral exp(i spin theta) over the lens arc."""
    k=abs(int(spin))
    out=np.zeros(order+1)
    if k==0:
        out[0]=math.pi
        for j in range((order+1)//2):
            out[2*j+1]=-math.comb(2*j,j)/(2.**(4*j)*(2*j+1))
    else:
        # 2 sin(k acos(r/2))/k = 2 sqrt(1-r²/4) U_(k-1)(r/2)/k.
        root=np.zeros(order+1)
        root[0]=1
        for j in range(1,order//2+1):
            root[2*j]=root[2*j-2]*(.5-(j-1))/j*(-.25)
        for j in range((k-1)//2+1):
            power=k-1-2*j
            if power<=order:
                out[power:]+=2/k*(-1)**j*math.comb(k-1-j,j)*root[:order+1-power]
    return out


@lru_cache(maxsize=8192)
def moment_matrix(he,ae,nh,na,radius):
    delta=he-ae
    if abs(delta-round(delta.real))>2e-8:
        raise ArithmeticError(f'nonintegral lens spin {delta}')
    beta=he+ae+2
    if beta.real<=0:
        raise ArithmeticError(f'lens outside absolute-convergence strip: {beta}')
    i,j=np.indices((nh,na))
    spins=round(delta.real)+i-j
    powers=beta+i+j
    result=np.empty((nh,na),complex)
    for spin in np.unique(spins):
        use=spins==spin
        coeff=aperture_coefficients(int(spin))
        nonzero=np.flatnonzero(coeff)
        exponent=powers[use,None]+nonzero
        result[use]=np.sum(coeff[nonzero]*np.exp(math.log(radius)*exponent)/exponent,axis=1)
    return result


def lens_pair(hol,anti,radius):
    moments=moment_matrix(hol.exponent,anti.exponent,len(hol.coefficients),len(anti.coefficients),radius)
    return hol.coefficients @ moments @ anti.coefficients


def inverted_series(I,local):
    """Pull back t=-w/(1-w), including one holomorphic Jacobian factor.

    The leading branch phase is applied to the complete hol/anti pair,
    not to independently re-rooted chiral functions.
    """
    n=len(local.coefficients)-1
    out=np.zeros(n+1,complex)
    for j,c in enumerate(local.coefficients):
        out[j:]+=c*(-1)**j*I.binomial(-local.exponent-j-2,n-j)
    return I.Local(local.exponent,out)


def inverted_pair(I,hol,anti,radius):
    spin=hol.exponent-anti.exponent
    if abs(spin-round(spin.real))>2e-8:
        raise ArithmeticError(f'nonintegral inversion spin {spin}')
    return (-1)**round(spin.real)*lens_pair(inverted_series(I,hol),inverted_series(I,anti),radius)


def load_integrator(frozen_root,baseline):
    sys.path.insert(0,str(Path(frozen_root).resolve()/'Codes'))
    source=Path(baseline)/'source'/'integrate_so7e8_literature_banks.py'
    expected=json.loads((Path(baseline)/'provenance.json').read_text())['source_sha256']['integrate_so7e8_literature_banks.py']
    if hashlib.sha256(source.read_bytes()).hexdigest()!=expected:
        raise ValueError('archived integrator changed')
    spec=importlib.util.spec_from_file_location('lens_archived_integrator',source)
    I=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=I
    spec.loader.exec_module(I)
    return I,expected


def integrate_lenses(I,records,constants,radius,degree,*,observable,momenta,orders=(4,5)):
    original_pair=I.disk_pair
    answer={}
    try:
        for exterior in (False,True):
            I.disk_pair=(lambda h,a,r:inverted_pair(I,h,a,r)) if exterior else lens_pair
            if observable=='rrrr':
                values,diagnostic=I.four_r_disk(records['rrrr','t'],constants,radius,degree,orders=orders)
                values={n:{'rrrr/reference':I.four_ramond_channel_to_base(v,1j,'t')} for n,v in values.items()}
            else:
                times=(momenta[0],-momenta[3],momenta[2],momenta[1])
                outer={key:I.chart_outer(key[0][::-1],times,'s',key[1])
                       for key in I.product(I.SPECIES,I.PICTURES)}
                raw,diagnostic=I.mixed_disk(records['mixed','s'],outer,constants,radius,degree,orders=orders)
                values={n:{'/'.join(k):v for k,v in vv.items()} for n,vv in raw.items()}
            answer['exterior_lens' if exterior else 'original_lens']=dict(values=values,diagnostic=diagnostic)
    finally:
        I.disk_pair=original_pair
    return answer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frozen-root',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--energy',required=True)
    p.add_argument('--observable',choices=('rrrr','mixed'),default='rrrr')
    p.add_argument('--local-degree',type=int,default=17)
    a=p.parse_args();started=time.monotonic()
    I,integrator_hash=load_integrator(a.frozen_root,a.baseline)
    m,records,_=I.load_records(a.run,a.energy)
    mom=tuple(I.unpairs(m['energies'][a.energy]))
    radius=m['config']['moduli']['settings']['crossed_ope_radius']
    constants=I.SelfDualConstants(m['config']['precision'])
    # Certify the entire small-coordinate circle containing both curved lenses.
    # The inversion maps |w|<=R into |t|<=R/(1-R); retain both cut lips.
    from resolve_so7e8_literature_atlas import channel_order_errors
    theta=2*np.pi*(np.arange(72)+.5)/72
    w=radius*np.exp(1j*theta)
    chart='t' if a.observable=='rrrr' else 's'
    changes,_=channel_order_errors(records[a.observable,chart],np.concatenate((w,-w/(1-w))))
    if max(changes)>.02:
        raise ArithmeticError('lens boundary fails original adjacent-order criterion')
    lenses=integrate_lenses(I,records,constants,radius,a.local_degree,observable=a.observable,momenta=mom)
    baseline=json.loads((a.baseline/(a.energy+'.json')).read_text())
    b=baseline['four_ramond' if a.observable=='rrrr' else 'mixed']
    rows={}
    for key,previous in b.items():
        orders={}
        for n in (4,5):
            pieces={name:I.unpairs(v) for name,v in previous['orders'][str(n)]['pieces'].items()}
            for name,row in lenses.items():
                pieces[name]=row['values'][n][key]
            orders[str(n)]=I.summarize_pieces(pieces)
        before=I.unpairs(previous['orders']['5']['value']);after=I.unpairs(orders['5']['value'])
        rows[key]=dict(orders=orders,baseline_order5=I.pairs(before),
            relative_component_changes=(abs(after-before)/np.maximum(abs(after),1e-300)).tolist())
    result=dict(status='analytic_lens_refinement_computed',energy=a.energy,observable=a.observable,
        local_degree=a.local_degree,physical_block_orders=[4,5],momentum_nodes=32,
        radius=radius,lens_aperture='theta in [-acos(r/2),acos(r/2)]',
        inversion='t=-w/(1-w), area Jacobian |1-w|^-4; leading hol/anti phase (-1)^(h-a)',
        maximum_boundary_block_change=float(max(changes)),rows=rows,
        lens_diagnostics={name:v['diagnostic'] for name,v in lenses.items()},
        source_manifest_sha256=m['manifest_sha256'],integrator_sha256=integrator_hash,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        baseline_sha256=hashlib.sha256((a.baseline/(a.energy+'.json')).read_bytes()).hexdigest(),
        formula_used=False,seconds=time.monotonic()-started,
        limitations=['Unchanged N32 spectral rule, finite block orders and epsilon extrapolation.',
                     'Existing bulk and other disk pieces retained; their independent errors still require assessment.',
                     'Boundary samples are diagnostics, not a continuum block-order proof.'])
    a.output.mkdir(parents=True,exist_ok=True)
    target=a.output/f'{a.energy}_{a.observable}_lens{a.local_degree}.json'
    I.atomic_json(target,result)
    print(json.dumps(dict(file=str(target),seconds=result['seconds'],changes={k:r['relative_component_changes'] for k,r in rows.items()})),flush=True)


if __name__=='__main__':
    main()
