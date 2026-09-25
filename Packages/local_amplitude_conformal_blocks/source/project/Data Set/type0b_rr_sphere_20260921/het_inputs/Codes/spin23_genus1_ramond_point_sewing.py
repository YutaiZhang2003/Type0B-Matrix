"""Evaluate finite Ramond rectangles directly at the requested external weight.

This is algebraically the homogeneous sewing audit, with no external-weight
polynomial reconstruction. Full finite matrices sum descendant levels before
the necklace trace. No physical coefficients or parity phases are changed.
"""
from functools import lru_cache
from itertools import product
import math
import numpy as np

from ns_algebra.ns_sca import fermion_parity
from spin23_generated.ramond_b1_kernels import evaluate_vertex_parts,GENERATED_MAX_TWICE_LEVEL
from spin23_ramond_fast import _fast_edge_factor,_whiten_vertex_sign_pair
from spin23_super_liouville_data import ns_weight,rr_ns_structure_constants
from spin23_genus1_ramond_sewing_audit import _cutoffs
from spin23_ramond_point_precision import checked_parts


@lru_cache(maxsize=8192)
def vertex_pair(left_level,right_level,p_left,p_right,h,word,left_lift,right_lift,limit):
    left=_fast_edge_factor(left_level,p_left,left_lift,limit)
    right=_fast_edge_factor(right_level,p_right,right_lift,limit)
    if word and not (len(word)==1 and word[0].kind=='G' and word[0].twice_index==-1):
        raise NotImplementedError('only primary and G_-1/2 vertices are supported')
    parts=checked_parts('GG' if word else 'PP',left_level,right_level,p_left,h,p_right)
    return np.stack(_whiten_vertex_sign_pair(parts[0],parts[1],left,right))


def layout(momentum,cutoff,lift,limit):
    levels=tuple(range(0,cutoff+1,2));offsets=[0];parities=[];powers=[]
    for level in levels:
        basis=_fast_edge_factor(level,momentum,lift,limit).basis
        offsets.append(offsets[-1]+len(basis))
        parities.extend(s.parity for s in basis);powers.extend([level//2]*len(basis))
    return levels,offsets,tuple(np.flatnonzero(np.asarray(parities)==p) for p in (0,1)),np.asarray(powers)


def chiral_trace(vertices,layouts,signs,forms,words):
    total=0j
    for initial_parity in (0,1):
        previous_parity=initial_parity;matrices=[]
        for v in range(len(vertices)):
            next_parity=(previous_parity+forms[v]+fermion_parity(words[v]))%2
            rows=layouts[v-1][2][previous_parity];cols=layouts[v][2][next_parity]
            matrices.append(vertices[v][0 if signs[v]==-1 else 1][np.ix_(rows,cols)])
            previous_parity=next_parity
        if previous_parity!=initial_parity:
            continue
        if len(matrices)==1:
            total+=np.trace(matrices[0]);continue
        value=matrices[0]
        for matrix in matrices[1:-1]:value=value@matrix
        total+=np.einsum('ij,ji->',value,matrices[-1])
    return complex(total)


def paired_momentum_integrand(internal_momenta,*,external_momenta,plumbing_parameters,
    holomorphic_words,antiholomorphic_words,maximum_twice_levels=0,
    temporal_lift_sign=1,structure_precision=24,block_digits=30,
    condition_limit=1e11,strip_internal_gaussian=False):
    internal=tuple(float(p) for p in internal_momenta)
    external=tuple(complex(p) for p in external_momenta)
    n=len(internal);holo=tuple(map(tuple,holomorphic_words));anti=tuple(map(tuple,antiholomorphic_words))
    if not n or any(len(x)!=n for x in (external,plumbing_parameters,holo,anti)):
        raise ValueError('all necklace input lengths must agree')
    if any(not math.isfinite(p) or p<=0 for p in internal):raise ValueError('positive finite internal momenta required')
    cutoffs=_cutoffs(maximum_twice_levels,n)
    if max(cutoffs)>GENERATED_MAX_TWICE_LEVEL:raise NotImplementedError('rectangle exceeds generated kernels')
    a=tuple(map(fermion_parity,holo));b=tuple(map(fermion_parity,anti))
    if sum(a)%2!=sum(b)%2:return 0j
    lifts=(temporal_lift_sign,)+(1,)*(n-1)
    q=tuple(complex(x) for x in plumbing_parameters)
    layouts=tuple(layout(p,c,l,condition_limit) for p,c,l in zip(internal,cutoffs,lifts))
    def vertices(words,plumbing):
        result=[]
        for v in range(n):
            left_levels,left_offsets,_,_=layouts[v-1]
            right_levels,right_offsets,_,powers=layouts[v]
            full=np.zeros((2,left_offsets[-1],right_offsets[-1]),dtype=complex)
            for i,ll in enumerate(left_levels):
                for j,rr in enumerate(right_levels):
                    full[:,left_offsets[i]:left_offsets[i+1],right_offsets[j]:right_offsets[j+1]]=vertex_pair(
                        ll,rr,internal[v-1],internal[v],complex(ns_weight(external[v])),
                        words[v],lifts[v-1],lifts[v],condition_limit)
            full*=np.power(plumbing[v],powers)[None,None,:]
            result.append(full)
        return result
    left=vertices(holo,q);right=vertices(anti,tuple(x.conjugate() for x in q))
    constants=tuple(rr_ns_structure_constants(internal[v-1],internal[v],external[v],
        precision=structure_precision) for v in range(n))
    total=0j
    for signs in product((-1,1),repeat=n):
        weight=math.prod(constants[v][0 if s==1 else 1]/2 for v,s in enumerate(signs))
        for forms in product((0,1),repeat=n):
            if sum(forms)%2!=sum(a)%2:continue
            lp=tuple((f+x)%2 for f,x in zip(forms,a));rp=tuple((f+x)%2 for f,x in zip(forms,b))
            crossings=sum(rp[i]*lp[j] for i in range(n) for j in range(i+1,n))
            phase=(-1j)**sum(b)*(-1)**(sum(f*x for f,x in zip(forms,b))+crossings)
            total+=weight*phase*chiral_trace(left,layouts,signs,forms,holo)*chiral_trace(right,layouts,signs,forms,anti)
    if not strip_internal_gaussian:total*=math.exp(sum(math.log(abs(x))*p*p for x,p in zip(q,internal)))
    return complex(total)
