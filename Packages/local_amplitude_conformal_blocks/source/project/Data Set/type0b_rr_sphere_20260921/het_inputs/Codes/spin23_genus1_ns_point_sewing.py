"""NS finite-matrix evaluation, retaining the homogeneous form ledger."""
from functools import lru_cache
from itertools import product
import math
import numpy as np

from ns_algebra.ns_sca import fermion_parity
from spin23_ns_fast import _fast_edge_factor,_prewhitened_vertex_coefficients,_evaluate_matrix_polynomial
from spin23_super_liouville_data import ns_weight,ns_structure_constants
from spin23_genus1_ramond_point_sewing import chiral_trace
from spin23_genus1_ns_pairing_audit import pairing_phase


@lru_cache(maxsize=8192)
def vertex(left_level,right_level,p_left,p_right,h,word,left_lift,right_lift,limit):
    coefficients=_prewhitened_vertex_coefficients(left_level,word,right_level,
        p_left,p_right,left_lift,right_lift,limit)
    return _evaluate_matrix_polynomial(coefficients,h)


def layout(momentum,cutoff,lift,limit):
    levels=tuple(range(cutoff+1));offsets=[0];parities=[];powers=[]
    for level in levels:
        basis=_fast_edge_factor(level,momentum,lift,limit).basis
        offsets.append(offsets[-1]+len(basis))
        parities.extend(fermion_parity(s) for s in basis);powers.extend([level/2]*len(basis))
    return levels,offsets,tuple(np.flatnonzero(np.asarray(parities)==p) for p in (0,1)),np.asarray(powers)


def form_contributions(internal_momenta,*,external_momenta,plumbing_parameters,
    holomorphic_words,antiholomorphic_words,maximum_twice_levels,
    temporal_lift_sign=1,structure_precision=24,block_digits=30,
    condition_limit=1e11,strip_internal_gaussian=True,edge_lift_signs=None):
    internal=tuple(float(p) for p in internal_momenta);external=tuple(complex(p) for p in external_momenta)
    n=len(internal);holo=tuple(map(tuple,holomorphic_words));anti=tuple(map(tuple,antiholomorphic_words))
    cutoffs=(maximum_twice_levels,)*n if isinstance(maximum_twice_levels,int) else tuple(maximum_twice_levels)
    if not n or any(len(x)!=n for x in (external,plumbing_parameters,holo,anti,cutoffs)):
        raise ValueError('all necklace input lengths must agree')
    if any(not math.isfinite(p) or p<=0 for p in internal):raise ValueError('positive finite momenta required')
    if any(not isinstance(c,int) or c<0 or c>8 for c in cutoffs):raise ValueError('twice levels from zero to eight required')
    if temporal_lift_sign not in (-1,1):raise ValueError('temporal lift must be +1 or -1')
    lifts=((temporal_lift_sign,)+(1,)*(n-1) if edge_lift_signs is None
           else tuple(edge_lift_signs))
    if len(lifts)!=n or any(x not in (-1,1) for x in lifts):
        raise ValueError('one +1 or -1 lift per edge is required')
    q=tuple(complex(x) for x in plumbing_parameters)
    layouts=tuple(layout(p,c,l,condition_limit) for p,c,l in zip(internal,cutoffs,lifts))
    def vertices(words,plumbing):
        result=[]
        for v in range(n):
            left_levels,left_offsets,_,_=layouts[v-1];right_levels,right_offsets,_,powers=layouts[v]
            full=np.zeros((1,left_offsets[-1],right_offsets[-1]),dtype=complex)
            for i,ll in enumerate(left_levels):
                for j,rr in enumerate(right_levels):
                    full[0,left_offsets[i]:left_offsets[i+1],right_offsets[j]:right_offsets[j+1]]=vertex(
                        ll,rr,internal[v-1],internal[v],complex(ns_weight(external[v])),words[v],
                        lifts[v-1],lifts[v],condition_limit)
            full*=np.power(plumbing[v],powers)[None,None,:]
            result.append(full)
        return result
    left=vertices(holo,q);right=vertices(anti,tuple(x.conjugate() for x in q))
    constants=tuple(ns_structure_constants(internal[v-1],external[v],internal[v],
        precision=structure_precision) for v in range(n))
    primary=math.exp(sum(math.log(abs(x))*(-1/8+(0 if strip_internal_gaussian else p*p))
                        for x,p in zip(q,internal)))
    result={}
    for f in product((0,1),repeat=n):
        if sum(f)%2!=sum(map(fermion_parity,holo))%2 or sum(f)%2!=sum(map(fermion_parity,anti))%2:continue
        result[f]=primary*math.prod(c[x] for c,x in zip(constants,f))*chiral_trace(
            left,layouts,(-1,)*n,f,holo)*chiral_trace(right,layouts,(-1,)*n,f,anti)
    return result


def paired_momentum_integrand(internal_momenta,**options):
    values=form_contributions(internal_momenta,**options)
    return sum(value*pairing_phase(f,options['holomorphic_words'],options['antiholomorphic_words'])
               for f,value in values.items())
