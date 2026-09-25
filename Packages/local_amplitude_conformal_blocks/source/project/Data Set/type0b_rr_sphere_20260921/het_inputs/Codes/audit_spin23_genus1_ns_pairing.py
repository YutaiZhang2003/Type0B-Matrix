"""Isolate NS nonchiral pairing using the saved, unchanged Ramond data."""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import time
from itertools import product

from spin23_genus1_ns_pairing_audit import pairing_phase
from spin23_genus1_ns_point_sewing import form_contributions
from spin23_genus1_threepoint_integral import states_for_energy,_KERNEL_KEYS,complex_json
from spin23_genus1_spectral import gauss_laguerre_spectral_integral_batch
from spin23_genus1_amplitude import (evaluate_genus_one_integrand,
    liouville_annulus_to_additive_factor,antiholomorphic_liouville_words)
from audit_spin23_genus1_threepoint_modularity import trace_frame_factor


def audit(cutoff,order,workers=1):
    started=time.perf_counter();form_ledger=[]
    forms=tuple(f for f in product((0,1),repeat=3) if sum(f)%2==1)
    def callback(**options):
        kw={k:options[k] for k in _KERNEL_KEYS}
        def node(p):
            values=form_contributions(p,**kw)
            return tuple(values[f] for f in forms)
        ds=gauss_laguerre_spectral_integral_batch(node,output_size=4,
            gaussian_scales=tuple(-math.log(abs(q)) for q in kw['plumbing_parameters']),
            quadrature_order=order,workers=workers)
        phases=tuple(pairing_phase(f,kw['holomorphic_words'],kw['antiholomorphic_words']) for f in forms)
        form_ledger.append(dict(forms=forms,values=complex_json([d.value for d in ds]),phases=complex_json(phases)))
        return replace(ds[0],value=sum(p*d.value for p,d in zip(phases,ds)))
    states=states_for_energy(.25,.35);tau=.1+1.25j
    r=evaluate_genus_one_integrand(states,tau,(0j,.18+.36j,.39+.81j),
        maximum_twice_levels=cutoff,p_max=None,quadrature_order=order,
        structure_precision=24,block_backend='direct_fast',spin_labels=('NS_tilde',),
        liouville_evaluator=callback)
    external=tuple(s.liouville_momentum for s in states)
    anti=antiholomorphic_liouville_words(states);ledger=[]
    for item in r.fixed_spin['NS_tilde'].components:
        h=item.component.holomorphic_liouville_words
        ledger.append(-.5j*item.value*trace_frame_factor(external,h,anti)/
                      liouville_annulus_to_additive_factor(r.coordinates,external,h,anti))
    def c(z):return complex(z['real'],z['imag'])
    old=next(x for x in json.loads(Path('data_exports/genus1/spin23_genus1_threepoint_rc_results_2.json').read_text())['results']
             if x['maximum_twice_level']==cutoff and x['laguerre_order']==order)
    ramond=list(map(c,old['evaluations'][1]['trace_frame_components']))
    return dict(status='NS pairing diagnostic; separate derivation/validation required',
        cutoff=cutoff,order=order,form_ledger=form_ledger,
        ns_components=complex_json(ledger),ramond_components=complex_json(ramond),
        ratios=complex_json([ramond[i]/(abs(tau)**8*ledger[j]) for i,j in [(0,0),(1,2),(2,1),(3,3)]]),
        total=complex_json(sum(ramond)/(abs(tau)**8*sum(ledger))),
        runtime_seconds=time.perf_counter()-started)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cutoff',type=int,default=6)
    p.add_argument('--order',type=int,default=15);p.add_argument('--workers',type=int,default=1)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('output already exists')
    result=audit(a.cutoff,a.order,a.workers)
    with a.output.open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps({k:result[k] for k in ('ratios','total','runtime_seconds')},indent=2))
