"""BRY-normalized graded NS pairing and an independent finite-series oracle.

HJS hep-th/0611266, printed pp. 10--11, supplies the chiral decompositions.
Use Ctilde_HJS=i*Ctilde_BRY and -i per right superdescendant. Multiplying
graded tensors supplies the crossing sign retained below. Diagonal pairings
are unchanged; mixed left/right words require the form-dependent signs.
"""
from itertools import product
import math

from ns_algebra.ns_sca import fermion_parity
from spin23_ns_fast import direct_fast_b1_ns_necklace_series
from spin23_super_liouville_data import ns_structure_constants


def pairing_phase(forms,holo,anti):
    """Graded pairing after the HJS-to-BRY component conversion.

    Each odd three-form contributes -i; each right descendant contributes
    (-i)(-1)^f. The rest comes from interchanging homogeneous chiral factors.
    See the independent eight-channel sphere pairing tests.
    """
    a=tuple(fermion_parity(w) for w in holo)
    b=tuple(fermion_parity(w) for w in anti)
    left=tuple((f+x)%2 for f,x in zip(forms,a))
    right=tuple((f+x)%2 for f,x in zip(forms,b))
    crossings=sum(right[i]*left[j] for i in range(len(forms))
                  for j in range(i+1,len(forms)))
    return (-1j)**(sum(forms)+sum(b))*(-1)**(
        sum(f*x for f,x in zip(forms,b))+crossings)


def form_contributions(internal_momenta,*,external_momenta,plumbing_parameters,
                       holomorphic_words,antiholomorphic_words,
                       maximum_twice_levels,temporal_lift_sign=1,
                       structure_precision=24,condition_limit=1e11,
                       strip_internal_gaussian=True,block_digits=30):
    internal=tuple(internal_momenta);external=tuple(external_momenta)
    holo=tuple(holomorphic_words);anti=tuple(antiholomorphic_words)
    n=len(internal);q=tuple(complex(x) for x in plumbing_parameters)
    constants=tuple(ns_structure_constants(internal[v-1],external[v],internal[v],
        precision=structure_precision) for v in range(n))
    options=dict(internal_momenta=internal,external_ns_momenta=external,
        maximum_twice_levels=maximum_twice_levels,
        edge_lift_signs=(temporal_lift_sign,)+(1,)*(n-1),condition_limit=condition_limit)
    primary=math.exp(sum(math.log(abs(qe))*(-1/8+(0 if strip_internal_gaussian else p*p))
                        for qe,p in zip(q,internal)))
    result={}
    for f in product((0,1),repeat=n):
        if sum(f)%2!=sum(map(fermion_parity,holo))%2 or sum(f)%2!=sum(map(fermion_parity,anti))%2:
            continue
        weights=tuple((1,0) if x==0 else (0,1) for x in f)
        left=direct_fast_b1_ns_necklace_series(**options,external_words=holo,form_weights=weights)
        right=left if holo==anti else direct_fast_b1_ns_necklace_series(
            **options,external_words=anti,form_weights=weights)
        result[f]=primary*math.prod(c[x] for c,x in zip(constants,f))*left.descendant_value(q)*right.descendant_value(tuple(x.conjugate() for x in q))
    return result
