"""CCY c-recursion for an ordinary Virasoro necklace with arbitrary primaries.

The residue products are those already used by the repository's two-Virasoro
Ramond construction. Here each of the two endpoint fusion polynomials keeps
its own external weight. The large-c seed is the global necklace times
prod_(m>=2)(1-Q^m)^-1, Q=prod(q_i), in annulus trace coordinates.
"""
import cmath
import math
from functools import lru_cache
from itertools import product

from spin23_virasoro_torus_recursion import (
    _normalized_rho_necklace_two_edge,_b_square_rs_from_h,_c_rs_from_h,
    _fusion_polynomial,_minus_dc_dh_times_a_rs,
    _rising_pochhammer,
)


@lru_cache(maxsize=256)
def vacuum_coefficients(cutoff):
    a=[1]+[0]*cutoff
    for m in range(2,cutoff+1):
        for k in range(m,cutoff+1):a[k]+=a[k-m]
    return tuple(a)


def coefficient_table(*,c,internal_weights,external_weights,maximum_levels,maximum_total_level=None):
    weights=tuple(map(complex,internal_weights));external=tuple(map(complex,external_weights))
    cutoffs=tuple(maximum_levels);n=len(weights)
    if n<2 or len(external)!=n or len(cutoffs)!=n:
        raise ValueError('one external weight and cutoff per necklace edge required')
    if any(not isinstance(k,int) or k<0 for k in cutoffs):raise ValueError('nonnegative integer levels required')

    @lru_cache(maxsize=None)
    def global_coefficient(levels,h):
        value=1+0j
        for v in range(n):
            left_norm=math.factorial(levels[v-1])*_rising_pochhammer(2*h[v-1],levels[v-1])
            right_norm=math.factorial(levels[v])*_rising_pochhammer(2*h[v],levels[v])
            # Cancel the pairwise square root before sewing. Independently
            # chosen sqrt(N_left*N_right) can leave a spurious sign in a
            # three-cycle at complex branch weights.
            value*=_normalized_rho_necklace_two_edge(levels[v-1],levels[v],h[v-1],external[v],h[v])\
                *cmath.sqrt(left_norm*right_norm)/right_norm
        return value

    @lru_cache(maxsize=None)
    def residue(edge,r,s,h):
        b=cmath.sqrt(_b_square_rs_from_h(r,s,h[edge]));next_vertex=(edge+1)%n
        return _minus_dc_dh_times_a_rs(r,s,h[edge])*_fusion_polynomial(
            r,s,b,h[edge-1],external[edge])*_fusion_polynomial(r,s,b,h[next_vertex],external[next_vertex])

    @lru_cache(maxsize=None)
    def recurse(levels,current_c,h):
        value=sum(a*global_coefficient(tuple(k-j for k in levels),h)
                  for j,a in enumerate(vacuum_coefficients(min(levels))) if a)
        for edge,cutoff in enumerate(levels):
            for r in range(2,cutoff+1):
                for s in range(1,cutoff//r+1):
                    shift=r*s;pole=_c_rs_from_h(r,s,h[edge]);denominator=current_c-pole
                    if abs(denominator)<1e-11*max(1,abs(current_c),abs(pole)):
                        raise ArithmeticError('Confluent Virasoro c poles require a complete finite part')
                    child_levels=list(levels);child_levels[edge]-=shift
                    child_weights=list(h);child_weights[edge]+=shift
                    value+=residue(edge,r,s,h)/denominator*recurse(tuple(child_levels),pole,tuple(child_weights))
        return complex(value)

    return {levels:recurse(levels,complex(c),weights)
            for levels in product(*(range(k+1) for k in cutoffs))
            if maximum_total_level is None or sum(levels)<=maximum_total_level}
