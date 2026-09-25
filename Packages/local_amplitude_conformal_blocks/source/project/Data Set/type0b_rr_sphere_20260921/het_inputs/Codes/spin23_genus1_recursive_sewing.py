"""Spectral sewing with recursive NS and two-Virasoro Ramond blocks.

No direct descendant block is used by these entry points. The NS collision
fallback is the high-precision reference c-recursion, with a checked Cauchy
limit in internal weights. Ramond uses the assembled b -> 1 branch sum.
"""
import cmath
from functools import lru_cache
from itertools import product
import math
import mpmath as mp
import numpy as np

from ns_algebra.ns_sca import fermion_parity
from spin23_genus1_ns_c_recursion import MixedNSNecklaceCRecursion,FastMixedNSNecklaceCRecursion
from spin23_genus1_branch_recursion import self_dual_chiral_table,self_dual_ramond_tables
from spin23_genus1_ns_pairing_audit import pairing_phase
from spin23_super_liouville_data import ns_structure_constants,rr_ns_structure_constants


@lru_cache(maxsize=128)
def ns_coefficients(internal,external,words,forms,cutoffs,total_cutoff):
    h=tuple((1+p*p)/2 for p in internal);he=tuple((1+p*p)/2 for p in external)
    options=dict(central_charge=13.5,internal_weights=h,external_weights=he,
        vertex_sectors=forms,external_descendants=words)
    try:
        recursion=FastMixedNSNecklaceCRecursion(**options)
        table=recursion.coefficient_table(cutoffs,total_cutoff)
    except (ZeroDivisionError,ArithmeticError):
        # Equal quadrature nodes can make fixed-weight c poles coincide.
        # Take the analytic limit of the COMPLETE recursively assembled table.
        def average(radius):
            result={}
            with mp.workdps(80):
                for j in range(8):
                    delta=mp.mpf(radius)*mp.exp(2j*mp.pi*(mp.mpf(j)+mp.mpf('.5'))/8)
                    weights=tuple(mp.mpc(x)+(i+1)*delta for i,x in enumerate(h))
                    recursive=MixedNSNecklaceCRecursion(**dict(options,internal_weights=weights,working_precision=80))
                    values=recursive.coefficient_table(cutoffs,total_cutoff)
                    for k,v in values.items():result[k]=result.get(k,0j)+v/8
            return result
        table=average('0.001');check=average('0.002')
        error=max(abs(v-check[k]) for k,v in table.items())
        if error>2e-8*max(1.,max(abs(v) for v in table.values())):
            raise ArithmeticError(f'NS confluent-pole limit did not converge: {error:.3e}')
    return np.array(list(table),dtype=int),np.array(list(table.values()),dtype=complex)


@lru_cache(maxsize=256)
def ramond_coefficients(internal,external,words,forms,signs,cutoffs,total_cutoff):
    table,diagnostics=self_dual_chiral_table(sector='R',internal_momenta=internal,
        external_momenta=external,external_descendants=words,form_parities=forms,
        signs=signs,maximum_twice_levels=cutoffs,maximum_total_twice_level=total_cutoff)
    return np.array(list(table),dtype=int),np.array(list(table.values()),dtype=complex),diagnostics


@lru_cache(maxsize=32)
def ramond_batch(internal,external,words,cutoffs,total_cutoff):
    return self_dual_ramond_tables(internal_momenta=internal,external_momenta=external,
        external_descendants=words,maximum_twice_levels=cutoffs,
        maximum_total_twice_level=total_cutoff)


def _inputs(internal,options):
    internal=tuple(float(p) for p in internal);external=tuple(map(complex,options['external_momenta']))
    holo=tuple(map(tuple,options['holomorphic_words']));anti=tuple(map(tuple,options['antiholomorphic_words']))
    n=len(internal);cut=options['maximum_twice_levels'];cuts=(cut,)*n if isinstance(cut,int) else tuple(cut)
    if n<2 or any(len(x)!=n for x in (external,holo,anti,cuts,options['plumbing_parameters'])):
        raise ValueError('one momentum, word and cutoff per edge required')
    if any(not math.isfinite(p) or p<=0 for p in internal):raise ValueError('positive finite internal momenta required')
    if any(not isinstance(c,int) or c<0 for c in cuts):raise ValueError('nonnegative integer cutoffs required')
    for word in holo+anti:
        if word and not (len(word)==1 and word[0].kind=='G' and word[0].twice_index==-1):
            raise NotImplementedError('P and G_-1/2 external fields are supported')
    return internal,external,holo,anti,cuts


def _evaluate(data,q,lifts):
    levels,values=data[:2]
    logs=np.array([cmath.log(z) for z in q])
    factors=np.exp(levels@logs/2)*np.prod(np.power(np.array(lifts),levels),axis=1)
    return complex(values@factors)


def ns_pair(internal_momenta,**options):
    internal,external,holo,anti,cuts=_inputs(internal_momenta,options);n=len(internal)
    q=tuple(map(complex,options['plumbing_parameters']));qb=tuple(z.conjugate() for z in q)
    a=tuple(map(fermion_parity,holo));b=tuple(map(fermion_parity,anti))
    lift=options.get('temporal_lift_sign',1)
    lifts=tuple(options.get('edge_lift_signs') or ((lift,)+(1,)*(n-1)))
    if len(lifts)!=n or any(x not in (-1,1) for x in lifts):raise ValueError('invalid edge lifts')
    cutoff=options.get('maximum_total_twice_level')
    constants=[ns_structure_constants(internal[v-1],external[v],internal[v],
        precision=options.get('structure_precision',24)) for v in range(n)]
    value=0j
    for forms in product((0,1),repeat=n):
        if sum(forms)%2!=sum(a)%2 or sum(forms)%2!=sum(b)%2:continue
        left=ns_coefficients(internal,external,a,forms,cuts,cutoff)
        right=left if a==b else ns_coefficients(internal,external,b,forms,cuts,cutoff)
        value+=pairing_phase(forms,holo,anti)*math.prod(c[f] for c,f in zip(constants,forms))*\
            _evaluate(left,q,lifts)*_evaluate(right,qb,lifts)
    value*=math.exp(sum(math.log(abs(z))*(-1/8+(0 if options.get('strip_internal_gaussian',False) else p*p))
                        for z,p in zip(q,internal)))
    return value


def r_pair(internal_momenta,**options):
    internal,external,holo,anti,cuts=_inputs(internal_momenta,options);n=len(internal)
    if options.get('temporal_lift_sign',1)!=1:
        raise NotImplementedError('This entry point covers the ordinary R trace in the even-spin SVV amplitude')
    q=tuple(map(complex,options['plumbing_parameters']));qb=tuple(z.conjugate() for z in q)
    a=tuple(map(fermion_parity,holo));b=tuple(map(fermion_parity,anti))
    cutoff=options.get('maximum_total_twice_level');cuts=tuple(c-c%2 for c in cuts)
    constants=[rr_ns_structure_constants(internal[v-1],internal[v],external[v],
        precision=options.get('structure_precision',24)) for v in range(n)]
    if sum(a)%2!=sum(b)%2:return 0j
    left=ramond_batch(internal,external,a,cuts,cutoff)
    right=left if a==b else ramond_batch(internal,external,b,cuts,cutoff)
    left_values=left[1]@np.exp(left[0]@np.array([cmath.log(z) for z in q])/2)
    right_values=right[1]@np.exp(right[0]@np.array([cmath.log(z) for z in qb])/2)
    value=0j
    for si,signs in enumerate(left[3]):
        weight=math.prod(c[0 if s==1 else 1]/2 for c,s in zip(constants,signs))
        for fi,forms in enumerate(left[2]):
            lp=tuple((f+x)%2 for f,x in zip(forms,a));rp=tuple((f+x)%2 for f,x in zip(forms,b))
            phase=(-1j)**sum(b)*(-1)**(sum(f*x for f,x in zip(forms,b))+
                sum(rp[i]*lp[j] for i in range(n) for j in range(i+1,n)))
            value+=weight*phase*left_values[fi,si]*right_values[fi,si]
    if not options.get('strip_internal_gaussian',False):
        value*=math.exp(sum(math.log(abs(z))*p*p for z,p in zip(q,internal)))
    return value
