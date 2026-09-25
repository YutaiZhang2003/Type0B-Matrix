#!/usr/bin/env python3
"""Regulated worldsheet S -> VV torus integral and common-sample energy scan.

Nine integration variables: tau (2), two punctures (4), three positive
Liouville momenta (3). No MQM amplitude is imported or used. A finite cusp
and necklace-gap cutoff define the reported object; removing these cutoffs
and establishing the physical continuation remain necessary.

Run with PYTHONPATH=Codes:. and the scientific Python environment.
"""
from dataclasses import dataclass, asdict
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.stats import qmc

from spin23_genus1_amplitude import (
    GenusOneNSState, evaluate_genus_one_integrand, ordered_necklace_coordinates,
)
from spin23_genus1_spectral import (
    SpectralIntegralDiagnostics, ns_liouville_necklace_momentum_integrand,
    ramond_liouville_necklace_momentum_integrand, tensor_spectral_integral,
)
from spin23_genus1_moduli_integral import fundamental_domain_point, spectral_importance_point
from spin23_genus1_ramond_sewing_audit import paired_momentum_integrand

SPINS=('NS','NS_tilde','R')


@dataclass(frozen=True)
class Regulators:
    tau2_max: float = 2.0
    minimum_necklace_gap: float = 0.12
    minimum_torus_distance: float = 0.05

    def __post_init__(self):
        if not math.isfinite(self.tau2_max) or self.tau2_max<=1:
            raise ValueError('tau2_max must be finite and exceed one')
        if not 0<=self.minimum_necklace_gap<1/3:
            raise ValueError('minimum_necklace_gap must be in [0,1/3)')
        if not math.isfinite(self.minimum_torus_distance) or self.minimum_torus_distance<0:
            raise ValueError('minimum_torus_distance must be finite and nonnegative')


@dataclass(frozen=True)
class IntegrationPoint:
    tau: complex
    points: tuple
    permutation: tuple
    plumbing: tuple
    internal_momenta: tuple
    geometry_weight: float
    spectral_weight: float
    minimum_gap: float
    minimum_distance: float
    accepted: bool


def periodic_distance(z,tau):
    """Shortest Euclidean separation on a torus in the standard domain."""
    b=z.imag/tau.imag
    best=math.inf
    for n in range(math.floor(b)-1,math.floor(b)+3):
        shifted=z-n*tau
        m=round(shifted.real)
        best=min(best,abs(shifted-m))
    return best


def map_point(unit,regulators):
    """Map the full labelled puncture square; sorting adds no factorial."""
    u=tuple(float(x) for x in unit)
    if len(u)!=9 or any(not 0<x<1 for x in u):
        raise ValueError('nine coordinates strictly inside the unit cube are required')
    tau,tau_weight=fundamental_domain_point(u[0],u[1],tau2_max=regulators.tau2_max)
    original=(0j,u[2]+u[3]*tau,u[4]+u[5]*tau)
    permutation=(0,1,2) if u[3]<u[5] else (0,2,1)
    points=tuple(original[i] for i in permutation)
    bs=sorted((u[3],u[5]))
    gap=min(bs[0],bs[1]-bs[0],1-bs[1])
    distance=min(periodic_distance(points[j]-points[i],tau)
                 for i in range(3) for j in range(i+1,3))
    accepted=(gap>=regulators.minimum_necklace_gap and
              distance>=regulators.minimum_torus_distance)
    if gap<=1e-12:
        return IntegrationPoint(tau,points,permutation,(),(),tau_weight*tau.imag**2,
                                0.,gap,distance,False)
    coordinates=ordered_necklace_coordinates(tau,points)
    scales=tuple(-math.log(abs(q)) for q in coordinates.plumbing_parameters)
    spectral=tuple(spectral_importance_point(x,gaussian_scale=s,endpoint_power=0)
                   for x,s in zip(u[6:],scales))
    return IntegrationPoint(tau,points,permutation,coordinates.plumbing_parameters,
        tuple(p for p,w in spectral),tau_weight*tau.imag**2,
        math.prod(w for p,w in spectral),gap,distance,accepted)


def states_for_energy(p,q,permutation=(0,1,2),flavor=0):
    p,q=float(p),float(q)
    if any(not math.isfinite(v) or v<=0 for v in (p,q)):
        raise ValueError('outgoing energies must be positive finite real numbers')
    states=(GenusOneNSState.singlet(p+q,-p-q),
            GenusOneNSState.vector(p,p,flavor),GenusOneNSState.vector(q,q,flavor))
    return tuple(states[i] for i in permutation)


_KERNEL_KEYS=('external_momenta','plumbing_parameters','holomorphic_words',
             'antiholomorphic_words','maximum_twice_levels','temporal_lift_sign',
             'structure_precision','block_digits','condition_limit')


def momentum_component(internal,options,*,strip_gaussian=True,ramond_policy='homogeneous'):
    kwargs={key:options[key] for key in _KERNEL_KEYS}
    kwargs['strip_internal_gaussian']=strip_gaussian
    if options['sector']=='NS':
        return ns_liouville_necklace_momentum_integrand(internal,**kwargs,
                                                       block_backend='direct_fast')
    if ramond_policy=='homogeneous':
        return paired_momentum_integrand(internal,**kwargs)
    if ramond_policy=='legacy':
        return ramond_liouville_necklace_momentum_integrand(internal,**kwargs,
                                                           block_backend='direct_fast')
    raise ValueError('unknown Ramond policy')


def node_liouville_evaluator(internal,spectral_weight,*,ramond_policy='homogeneous'):
    """Insert one importance-weighted spectral node into the existing assembler."""
    def evaluate(**options):
        value=momentum_component(internal,options,ramond_policy=ramond_policy)
        return SpectralIntegralDiagnostics(value=value*spectral_weight,
            refined_value=None,extended_value=None,quadrature_absolute_error=None,
            tail_absolute_error=None,estimated_absolute_error=None,dimension=3,
            p_max=None,quadrature_order=0,refined_order=None,extended_p_max=None,
            function_evaluations=1,quadrature_method='joint_scrambled_sobol_node')
    return evaluate


def fixed_moduli_evaluation(p,q,*,tau=.1+1.25j,
        points=(0j,.18+.36j,.39+.81j),cutoff=2,order=3,p_max=2.,precision=28,
        ramond_policy='homogeneous'):
    """Independent tensor quadrature at fixed moduli, useful for regressions."""
    def evaluate(**options):
        return tensor_spectral_integral(lambda internal:momentum_component(internal,
            options,strip_gaussian=False,ramond_policy=ramond_policy),dimension=3,
            p_max=p_max,quadrature_order=order)
    return evaluate_genus_one_integrand(states_for_energy(p,q),tau,points,
        maximum_twice_levels=cutoff,p_max=p_max,quadrature_order=order,
        structure_precision=precision,block_digits=precision,
        free_field_precision=precision,condition_limit=1e11,
        block_backend='direct_fast',include_string_phase=True,liouville_evaluator=evaluate)


def evaluate_node(point,energies,cutoffs,*,precision=24,ramond_policy='homogeneous'):
    values=np.zeros((len(energies),len(cutoffs),3,4),dtype=complex)
    if not point.accepted:
        return values
    evaluator=node_liouville_evaluator(point.internal_momenta,point.spectral_weight,
                                       ramond_policy=ramond_policy)
    for ie,(p,q) in enumerate(energies):
        states=states_for_energy(p,q,point.permutation)
        for il,cutoff in enumerate(cutoffs):
            result=evaluate_genus_one_integrand(states,point.tau,point.points,
                maximum_twice_levels=cutoff,p_max=None,quadrature_order=2,
                structure_precision=precision,block_digits=precision,
                free_field_precision=precision,condition_limit=1e11,
                block_backend='direct_fast',include_string_phase=True,
                liouville_evaluator=evaluator)
            for ispin,label in enumerate(SPINS):
                for ic,component in enumerate(result.fixed_spin[label].components):
                    # The public assembler stores fixed-spin terms BEFORE GSO
                    # and BEFORE i^3. Both are included here exactly once.
                    values[ie,il,ispin,ic]=(-1j)*0.5*component.value*point.geometry_weight
    if not np.all(np.isfinite(values)):
        raise ArithmeticError('non-finite integrand sample; no sample was silently dropped')
    return values


def summarize(replicates):
    """Independent-replicate mean and real/imaginary covariance of the mean."""
    a=np.asarray(replicates,dtype=complex)
    n=a.shape[0]
    if n<2:
        raise ValueError('at least two independent replicates are required')
    center=a.mean(axis=0); dr=a.real-center.real; di=a.imag-center.imag
    return dict(mean=center,stderr_real=np.sqrt((dr*dr).sum(axis=0)/(n*(n-1))),
        stderr_imag=np.sqrt((di*di).sum(axis=0)/(n*(n-1))),
        covariance_real_imag=(dr*di).sum(axis=0)/(n*(n-1)))


def complex_json(value):
    array=np.asarray(value)
    if array.ndim:
        return [complex_json(v) for v in array]
    value=complex(array)
    return dict(real=value.real,imag=value.imag)


def scan(*,energies,cutoffs,regulators=Regulators(),replicates=4,
         samples_per_replicate=8,seed=9023,precision=24,ramond_policy='homogeneous',
         progress=None):
    energies=tuple(tuple(map(float,pair)) for pair in energies)
    for p,q in energies:
        states_for_energy(p,q)
    cutoffs=tuple(cutoffs)
    if not energies or not cutoffs or any(isinstance(c,bool) or not isinstance(c,int)
                                         or c<0 for c in cutoffs):
        raise ValueError('nonempty energy grid and nonnegative integer cutoffs required')
    if replicates<2 or samples_per_replicate<2 or samples_per_replicate & (samples_per_replicate-1):
        raise ValueError('use at least two replicates and a power-of-two sample count')
    started=time.perf_counter(); raw=[];accepted=[];sample_ledger=[]
    for rep in range(replicates):
        points=qmc.Sobol(d=9,scramble=True,seed=seed+rep).random_base2(
            m=samples_per_replicate.bit_length()-1)
        total=np.zeros((len(energies),len(cutoffs),3,4),dtype=complex); count=0
        for index,u in enumerate(points):
            point=map_point(u,regulators)
            values=evaluate_node(point,energies,cutoffs,precision=precision,
                                 ramond_policy=ramond_policy)
            total+=values;count+=int(point.accepted)
            sample_ledger.append(dict(replicate=rep,index=index,unit=u.tolist(),
                accepted=point.accepted,minimum_gap=point.minimum_gap,
                minimum_distance=point.minimum_distance,
                weighted_components=complex_json(values)))
            if progress:
                progress(rep,index,point.accepted,time.perf_counter()-started)
        raw.append(total/samples_per_replicate);accepted.append(count)
    raw=np.asarray(raw); amplitude=raw.sum(axis=(-1,-2))
    tree_reduced=amplitude/np.asarray([(p+q)*p*q for p,q in energies])[None,:,None]
    statistics=summarize(amplitude)
    shape_statistics=summarize(tree_reduced)
    level_differences=np.diff(amplitude,axis=2)
    source_names=('Codes/spin23_genus1_threepoint_integral.py',
        'Codes/spin23_genus1_ramond_sewing_audit.py','Codes/spin23_genus1_amplitude.py',
        'Codes/spin23_genus1_spectral.py','Codes/spin23_genus1_blocks.py',
        'Codes/spin23_ramond_fast.py','Codes/spin23_super_liouville_data.py')
    root=Path(__file__).resolve().parents[1]
    return dict(schema='spin23-genus1-threepoint-regulated-v1',
        status='regulated diagnostic; not a certified physical genus-one amplitude',
        dimension=9,energies=[dict(p=p,q=q,E=p+q) for p,q in energies],
        maximum_twice_levels=list(cutoffs),regulators=asdict(regulators),
        replicates=replicates,samples_per_replicate=samples_per_replicate,
        seed=seed,precision=precision,ramond_policy=ramond_policy,accepted=accepted,
        spin_labels=list(SPINS),
        conventions=dict(raw='unnormalized singlet descendant; includes i^3 and diagonal GSO',
            canonical='raw/sqrt((1+E^2)*E*p*q), up to an energy-independent constant',
            tree_reduced='raw/(E*p*q); external singlet and LSZ factors cancel',
            measures='d^2tau d^2z_2 d^2z_3 product(dP_e/pi); no extra 2!'),
        mean=complex_json(statistics['mean']),
        stderr_real=statistics['stderr_real'].tolist(),
        stderr_imag=statistics['stderr_imag'].tolist(),
        covariance_real_imag=statistics['covariance_real_imag'].tolist(),
        tree_reduced_mean=complex_json(shape_statistics['mean']),
        tree_reduced_stderr_real=shape_statistics['stderr_real'].tolist(),
        tree_reduced_stderr_imag=shape_statistics['stderr_imag'].tolist(),
        level_difference_mean=complex_json(level_differences.mean(axis=0)),
        level_difference_stderr_real=summarize(level_differences)['stderr_real'].tolist(),
        level_difference_stderr_imag=summarize(level_differences)['stderr_imag'].tolist(),
        replicate_components=complex_json(raw),samples=sample_ledger,
        runtime_seconds=time.perf_counter()-started,
        source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                       for name in source_names},
        outstanding=['three-point modular/crossing and PCO-sign checks',
            'descendant-level and sampling convergence',
            'necklace chart completion and puncture-collision prescription',
            'cusp removal and Lorentzian continuation',
            'external-leg and vacuum/background treatment for the physical S-matrix'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--energy',action='append',help='outgoing p,q; repeat for energy grid')
    parser.add_argument('--cutoffs',default='0,2,4')
    parser.add_argument('--replicates',type=int,default=4)
    parser.add_argument('--samples',type=int,default=8)
    parser.add_argument('--seed',type=int,default=9023)
    parser.add_argument('--tau2-max',type=float,default=2.)
    parser.add_argument('--gap',type=float,default=.12)
    parser.add_argument('--distance',type=float,default=.05)
    parser.add_argument('--precision',type=int,default=24)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        parser.error('output already exists; choose a new artifact name')
    energies=[tuple(map(float,v.split(','))) for v in (args.energy or ['0.25,0.35','0.3,0.3'])]
    def progress(rep,index,accepted,elapsed):
        print(f'replicate {rep+1}/{args.replicates}, point {index+1}/{args.samples}, '
              f'accepted={accepted}, elapsed={elapsed:.1f}s',flush=True)
    result=scan(energies=energies,cutoffs=tuple(map(int,args.cutoffs.split(','))),
        regulators=Regulators(args.tau2_max,args.gap,args.distance),
        replicates=args.replicates,samples_per_replicate=args.samples,seed=args.seed,
        precision=args.precision,progress=progress)
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(f'Wrote {args.output}; regulated diagnostic only.',flush=True)


if __name__=='__main__':
    main()
