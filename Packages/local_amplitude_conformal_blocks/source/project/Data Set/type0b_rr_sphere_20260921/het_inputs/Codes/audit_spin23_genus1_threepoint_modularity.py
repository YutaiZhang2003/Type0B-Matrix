#!/usr/bin/env python3
"""Reproducible modular/coordinate diagnostic; not an amplitude evaluator.

The trace returned by the necklace block has no external w^h prefactor.
A plane correlator would contain w^-h, which cancels the w^h in dw/dz.
This script records both the existing frame and the position-independent
trace-frame conversion, without changing the existing amplitude code.
"""
import argparse
import cmath
import hashlib
import json
import math
from pathlib import Path
import time

from spin23_genus1_amplitude import (
    evaluate_genus_one_integrand,liouville_annulus_to_additive_factor,
    antiholomorphic_liouville_words,
)
from spin23_genus1_spectral import gauss_laguerre_spectral_integral
from spin23_genus1_threepoint_integral import momentum_component,states_for_energy,complex_json


def trace_frame_factor(external_momenta,holomorphic_words,antiholomorphic_words):
    """Conversion of a trace of V(1) matrices to additive cylinder vertices."""
    exponent=0j
    for p,holo,anti in zip(external_momenta,holomorphic_words,antiholomorphic_words):
        h=(1+complex(p)**2)/2
        exponent+=(h+.5*bool(holo))*cmath.log(2j*math.pi)
        exponent+=(h+.5*bool(anti))*cmath.log(-2j*math.pi)
    return cmath.exp(exponent)


def modular_ratio_record(numerator,denominator):
    """A vanished truncated component has no ratio; preserve that fact."""
    return None if denominator==0 else complex_json(numerator/denominator)


def audit(*,cutoff=2,order=3,precision=24,workers=1):
    tau=.1+1.25j;points=(0j,.18+.36j,.39+.81j)
    transformed_tau=-1/tau
    transformed_points=(0j,points[2]/tau+transformed_tau,
                         points[1]/tau+transformed_tau)
    records=[];started=time.perf_counter()
    for spin,t,z,energies in [('NS_tilde',tau,points,(.25,.35)),
                             ('R',transformed_tau,transformed_points,(.35,.25))]:
        states=states_for_energy(*energies)
        def liouville(**options):
            return gauss_laguerre_spectral_integral(
                lambda p:momentum_component(p,options),
                gaussian_scales=tuple(-math.log(abs(v))
                                      for v in options['plumbing_parameters']),
                quadrature_order=order,workers=workers)
        result=evaluate_genus_one_integrand(states,t,z,maximum_twice_levels=cutoff,
            p_max=None,quadrature_order=order,structure_precision=precision,
            block_digits=30,free_field_precision=30,condition_limit=1e11,
            block_backend='direct_fast',include_string_phase=True,
            spin_labels=(spin,),liouville_evaluator=liouville)
        external=tuple(state.liouville_momentum for state in states)
        anti=antiholomorphic_liouville_words(states)
        raw=[];trace_frame=[];fermion_reordered=[];ratios=[]
        for item in result.fixed_spin[spin].components:
            words=item.component.holomorphic_liouville_words
            old=liouville_annulus_to_additive_factor(result.coordinates,external,words,anti)
            flat=trace_frame_factor(external,words,anti)
            value=-.5j*item.value
            raw.append(value);trace_frame.append(value*flat/old);ratios.append(flat/old)
            selected=set(item.component.time_fermion_indices)
            # This is a diagnostic grouping convention. Its relation to
            # the NS/R external-fermion cocycles is not yet certified.
            inversions=sum(i in selected and j not in selected
                           for i in range(3) for j in range(i+1,3))
            fermion_reordered.append(value*flat/old*(-1)**inversions)
        records.append(dict(spin=spin,tau=complex_json(t),points=complex_json(z),
            raw_components=complex_json(raw),trace_frame_components=complex_json(trace_frame),
            trace_and_reordering_trial_components=complex_json(fermion_reordered),
            frame_correction=complex_json(ratios)))
    def decode(z):
        return complex(z['real'],z['imag'])
    ratio_records={}
    for name in ('raw_components','trace_frame_components',
                 'trace_and_reordering_trial_components'):
        original=list(map(decode,records[0][name])); transformed=list(map(decode,records[1][name]))
        ratio_records[name]=dict(
            total=modular_ratio_record(sum(transformed),abs(tau)**8*sum(original)),
            individual=[modular_ratio_record(transformed[i],abs(tau)**8*original[j])
                        for i,j in [(0,0),(1,2),(2,1),(3,3)]])
    root=Path(__file__).resolve().parents[1]
    names=(('Codes/' + Path(__file__).name),
           'Codes/spin23_genus1_ramond_sewing_audit.py','Codes/spin23_genus1_threepoint_integral.py')
    return dict(status='unresolved modular diagnostic; no physical amplitude certified',
        expected_full_modular_ratio=1,expected_density_weight=8,
        maximum_twice_level=cutoff,laguerre_order=order,precision=precision,
        spectral_workers=workers,
        evaluations=records,modular_ratios=ratio_records,
        notes=['The reordering trial is not selected as a repair by fitting this ratio.',
               'A null ratio denotes a zero denominator at the selected truncation; raw components are retained.',
               'Separate block and spectral convergence and the complete fermionic convention remain to be checked.'],
        runtime_seconds=time.perf_counter()-started,
        source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cutoff',type=int,default=2)
    parser.add_argument('--order',type=int,default=3)
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        parser.error('choose a new output artifact')
    result=audit(cutoff=args.cutoff,order=args.order,workers=args.workers)
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(result['modular_ratios'],indent=2))


if __name__=='__main__':
    main()
