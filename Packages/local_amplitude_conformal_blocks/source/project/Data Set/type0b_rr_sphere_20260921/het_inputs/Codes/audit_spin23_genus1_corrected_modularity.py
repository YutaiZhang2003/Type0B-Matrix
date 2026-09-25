"""Modular checks of parity-corrected finite-matrix three-point sewing."""
import argparse
import cmath
import hashlib
import json
import math
from pathlib import Path
import time

from spin23_genus1_amplitude import (evaluate_genus_one_integrand,
    liouville_annulus_to_additive_factor,antiholomorphic_liouville_words)
from spin23_genus1_spectral import gauss_laguerre_spectral_integral
from spin23_genus1_threepoint_integral import states_for_energy,complex_json,_KERNEL_KEYS
from spin23_genus1_ns_point_sewing import paired_momentum_integrand as ns_pair
from spin23_genus1_ramond_point_sewing import paired_momentum_integrand as r_pair
from audit_spin23_genus1_threepoint_modularity import trace_frame_factor,modular_ratio_record


def audit(*,cutoff=6,order=11,workers=1,p=.25,q=.35,geometry='original'):
    if geometry=='original':tau=.1+1.25j;points=(0j,.18+.36j,.39+.81j)
    elif geometry=='balanced':tau=1j;points=(0j,.249+1j/3,.498+2j/3)
    else:raise ValueError('unknown geometry')
    transformed_tau=-1/tau
    transformed_points=(0j,points[2]/tau+transformed_tau,points[1]/tau+transformed_tau)
    started=time.perf_counter();records=[]
    for spin,t,z,energy in [('NS_tilde',tau,points,(p,q)),('R',transformed_tau,transformed_points,(q,p))]:
        states=states_for_energy(*energy)
        def callback(**options):
            kw={k:options[k] for k in _KERNEL_KEYS};kw['strip_internal_gaussian']=True
            pair=ns_pair if options['sector']=='NS' else r_pair
            return gauss_laguerre_spectral_integral(lambda internal:pair(internal,**kw),
                gaussian_scales=tuple(-math.log(abs(x)) for x in kw['plumbing_parameters']),
                quadrature_order=order,workers=workers)
        result=evaluate_genus_one_integrand(states,t,z,maximum_twice_levels=cutoff,
            p_max=None,quadrature_order=order,structure_precision=24,block_digits=30,
            free_field_precision=30,condition_limit=1e11,block_backend='direct_fast',
            include_string_phase=True,spin_labels=(spin,),liouville_evaluator=callback)
        external=tuple(s.liouville_momentum for s in states)
        anti=antiholomorphic_liouville_words(states);values=[]
        for item in result.fixed_spin[spin].components:
            holo=item.component.holomorphic_liouville_words
            correction=trace_frame_factor(external,holo,anti)/liouville_annulus_to_additive_factor(
                result.coordinates,external,holo,anti)
            values.append(-.5j*item.value*correction)
        records.append(dict(spin=spin,tau=complex_json(t),points=complex_json(z),
            components=complex_json(values),plumbing=complex_json(result.coordinates.plumbing_parameters)))
        print(spin,'complete after',round(time.perf_counter()-started,2),'seconds',flush=True)
    def c(z):return complex(z['real'],z['imag'])
    original=list(map(c,records[0]['components']));transformed=list(map(c,records[1]['components']))
    names=('Codes/audit_spin23_genus1_corrected_modularity.py','Codes/spin23_genus1_ns_pairing_audit.py',
        'Codes/spin23_genus1_ns_point_sewing.py','Codes/spin23_genus1_ramond_point_sewing.py','Codes/spin23_ramond_point_precision.py',
        'Codes/audit_spin23_genus1_threepoint_modularity.py')
    root=Path(__file__).resolve().parents[1]
    return dict(status='fixed-geometry modular diagnostic; not an integrated string amplitude',
        geometry=geometry,energies=[p,q],maximum_twice_level=cutoff,laguerre_order=order,workers=workers,
        expected_ratio=1,evaluations=records,
        individual_ratios=[modular_ratio_record(transformed[i],abs(tau)**8*original[j]) for i,j in [(0,0),(1,2),(2,1),(3,3)]],
        total_ratio=modular_ratio_record(sum(transformed),abs(tau)**8*sum(original)),
        runtime_seconds=time.perf_counter()-started,
        source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cutoff',type=int,default=6);parser.add_argument('--order',type=int,default=11)
    parser.add_argument('--workers',type=int,default=1);parser.add_argument('--p',type=float,default=.25)
    parser.add_argument('--q',type=float,default=.35);parser.add_argument('--geometry',choices=('original','balanced'),default='original')
    parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    if a.output.exists():parser.error('choose a new output artifact')
    result=audit(cutoff=a.cutoff,order=a.order,workers=a.workers,p=a.p,q=a.q,geometry=a.geometry)
    with a.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps({key:result[key] for key in ('individual_ratios','total_ratio','runtime_seconds')},indent=2))
