"""Two-R/three-NS five-point chiral blocks in the three review channels.

Generic-b double-Virasoro coefficient generator plus a level<=2 PBW oracle.
These are chiral tensors, NOT a nonchiral correlator or a GSO projector.
R external states are w+; stars are unrescaled NS G_-1/2 insertions.
"""
from __future__ import annotations

import cmath
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from itertools import product
import math

import numpy as np
import sympy as sp

from ns_algebra.ns_sca import G, gram_matrix as ns_gram
from ramond_algebra.ramond_sca import ground_state, gram_matrix as r_gram
from ramond_algebra.nrr_three_point_tensor import hjs_to_polynomial_ground_tensor
from so7e8_internal_ramond_double_virasoro import ramond_branch_numbers, _sphere_rnsr_branch_form
from so7e8_mixed_double_virasoro import _external_resolution, _rr_branch_form, _rr_ward_template
from spin23_ns_torus_two_virasoro import _oriented_ns_branching_coefficient, _ns_three_point_numeric_template
from spin23_two_virasoro_ramond import embedded_branch_state, _rr_three_point_numeric_template
from virasoro_fivepoint_c_recursion import regular_fivepoint_c_coefficients


SECTORS = {"A": ("NS", "NS", "NS", "R", "R"),
           "B": ("R", "NS", "R", "NS", "NS"),
           "C": ("R", "NS", "NS", "NS", "R")}
EDGES = {"A": ("NS", "NS"), "B": ("R", "NS"), "C": ("R", "R")}
GWORD = (G(Fraction(-1, 2)),)


@dataclass(frozen=True)
class MixedFivePointSeries:
    coefficients: dict
    channel: str
    b: complex
    internal_weights: tuple
    external_weights: tuple
    stars: tuple
    edge_parities: tuple
    structure_signs: tuple
    maximum_twice_levels: tuple
    backend: str

    @property
    def leading_powers(self):
        effective = tuple(h + s/2 for h, s in zip(self.external_weights, self.stars))
        return (self.internal_weights[0]-effective[0]-effective[1],
                self.internal_weights[1]-sum(effective[:3]))

    def value(self, q1, q2, *, logarithms=None):
        logs = tuple(map(cmath.log, (q1,q2))) if logarithms is None else tuple(logarithms)
        a, b = self.leading_powers
        return sum(v*cmath.exp((a+i/2)*logs[0]+(b+j/2)*logs[1])
                   for (i,j),v in self.coefficients.items())


def _inputs(channel, b, internal_momenta, external_momenta, maximum_twice_levels,
            edge_parities, structure_signs, stars):
    if channel not in SECTORS:
        raise ValueError("channel must be A, B, or C")
    b = complex(b)
    internal, external = tuple(map(complex, internal_momenta)), tuple(map(complex, external_momenta))
    cut = ((maximum_twice_levels,)*2 if isinstance(maximum_twice_levels, int)
           else tuple(maximum_twice_levels))
    parities, signs, stars = tuple(edge_parities), tuple(structure_signs), tuple(stars)
    if len(internal) != 2 or len(external) != 5 or len(cut) != 2:
        raise ValueError("two internal momenta/cutoffs and five external momenta required")
    if any(type(x) is not int or x < 0 for x in cut):
        raise ValueError("twice-level cutoffs must be nonnegative integers")
    if len(parities) != 2 or any(x not in (0,1) for x in parities):
        raise ValueError("two edge parities required")
    if len(signs) != 3 or any(x not in (-1,1) for x in signs):
        raise ValueError("three structure signs required (NS-vertex signs are unused)")
    if len(stars) != 5 or any(x not in (0,1) for x in stars):
        raise ValueError("five NS star flags required")
    if any(s and sec == "R" for s, sec in zip(stars, SECTORS[channel])):
        raise ValueError("G_-1/2 is an NS insertion, not a Ramond descendant")
    if b == 0 or any(not (math.isfinite(z.real) and math.isfinite(z.imag))
                     for z in (b,*internal,*external)):
        raise ValueError("finite momenta and nonzero finite b required")
    q = b+1/b
    c = 1.5+3*q*q
    weight = lambda p, s: (c/24 if s == "R" else q*q/8)+p*p/2
    hi = tuple(weight(p,s) for p,s in zip(internal, EDGES[channel]))
    he = tuple(weight(p,s) for p,s in zip(external, SECTORS[channel]))
    forms = ((parities[0]+stars[0]+stars[1])%2,
             (parities[0]+parities[1]+stars[2])%2,
             (parities[1]+stars[3]+stars[4])%2)
    levels = tuple(tuple(range(p if s == "NS" else 0, n+1, 2))
                   for s,p,n in zip(EDGES[channel],parities,cut))
    return b, internal, external, cut, parities, signs, stars, c, hi, he, forms, levels


@lru_cache(maxsize=8192)
def _ns_form(a,b,c,p):
    return _oriented_ns_branching_coefficient(a,b,c,orientation="left",super_form_parity=p)


def _branches(b,p,sector,parity,maximum):
    if sector == "R":
        labels = ramond_branch_numbers(maximum)
    else:
        limit = math.isqrt(maximum)
        labels = tuple(Fraction(k,2) for k in range(-limit,limit+1) if k%2 == parity)
    for n in labels:
        state = embedded_branch_state(b=b,sector=sector,physical_momentum=p,
                                      branch_number=n,parity=parity)
        onset = int(4*n*n - (Fraction(1,4) if sector == "R" else 0))
        yield state,onset


@lru_cache(maxsize=8192)
def _two_virasoro_series(internals, externals, orders):
    factors = tuple(regular_fivepoint_c_coefficients(
        c=getattr(internals[0],f"c_{copy}"),
        internal_weights=tuple(getattr(x,f"h_{copy}") for x in internals),
        external_weights=tuple(getattr(x,f"h_{copy}") for x in externals),orders=orders)
        for copy in (1,2))
    return {(i,j):sum(factors[0][a,d]*factors[1][i-a,j-d]
                     for a in range(i+1) for d in range(j+1))
            for i in range(orders[0]+1) for j in range(orders[1]+1)}


def double_virasoro_fivepoint(*, channel, b, internal_momenta, external_momenta,
        maximum_twice_levels, edge_parities=(0,0), structure_signs=(1,1,1), stars=(0,)*5):
    args = _inputs(channel,b,internal_momenta,external_momenta,maximum_twice_levels,
                   edge_parities,structure_signs,stars)
    b,pi,pe,cut,pa,signs,stars,c,hi,he,forms,levels = args
    if abs(b*b-1) < 1e-12:
        raise ValueError("double-Virasoro embedding requires generic b; take assembled finite part")
    coefficients = dict.fromkeys(product(*levels),0j)
    resolutions = tuple(_external_resolution(b,p,s,star)[0]
                        for p,s,star in zip(pe,SECTORS[channel],stars))
    branches = tuple(tuple(_branches(b,p,s,parity,n))
                     for p,s,parity,n in zip(pi,EDGES[channel],pa,cut))
    for (a,oa),(d,od) in product(*branches):
        orders = ((cut[0]-oa)//2,(cut[1]-od)//2)
        for ext in product(*resolutions):
            prefactor = math.prod(x for x,_ in ext)
            e1,e2,e3,e4,e5 = (s for _,s in ext)
            if channel == "A":
                right = _ns_form(a,e2,e1,forms[0])
                middle = _ns_form(d,e3,a,forms[1])
                left = _rr_branch_form(d,e4,e5,signs[2],forms[2])
            elif channel == "B":
                right = _sphere_rnsr_branch_form(a,e2,e1,signs[0],forms[0])
                middle = _rr_branch_form(d,e3,a,signs[1],forms[1])
                left = _ns_form(e5,e4,d,forms[2])
            else:
                right = _sphere_rnsr_branch_form(a,e2,e1,signs[0],forms[0])
                middle = _sphere_rnsr_branch_form(d,e3,a,signs[1],forms[1])
                left = _sphere_rnsr_branch_form(e5,e4,d,signs[2],forms[2])
            weight = prefactor*left*middle*right/(a.norm*d.norm)
            if weight == 0:
                continue
            descendants = _two_virasoro_series((a.parameters,d.parameters),
                                               tuple(s.parameters for _,s in ext),orders)
            for (i,j),v in descendants.items():
                coefficients[oa+2*i,od+2*j] += weight*v
    # A's terminal R,R,NS vertex is represented by its cyclic NS,R,R tensor,
    # with the same fixed orientation as the validated four-point adapter.
    phase = (-1j)**pa[1] if channel == "A" else 1
    return MixedFivePointSeries({k:complex(v*phase) for k,v in coefficients.items()},
                channel,b,hi,he,stars,pa,signs,cut,"double Virasoro + five-point c-recursion")


@lru_cache(maxsize=256)
def _gram(sector,level,c,h,parity):
    basis,matrix = (ns_gram if sector == "NS" else r_gram)(level,h=sp.sympify(h),c=sp.sympify(c))
    indices = [i for i,s in enumerate(basis) if sector == "NS" or s.parity == parity]
    states = tuple(basis[i] for i in indices)
    values = np.array(matrix.extract(indices,indices),dtype=complex)
    return states,np.linalg.inv(values)


def direct_fivepoint_oracle(*, channel,b,internal_momenta,external_momenta,
        maximum_twice_levels,edge_parities=(0,0),structure_signs=(1,1,1),stars=(0,)*5):
    args = _inputs(channel,b,internal_momenta,external_momenta,maximum_twice_levels,
                   edge_parities,structure_signs,stars)
    b,pi,pe,cut,pa,signs,stars,c,hi,he,forms,levels = args
    if max(cut)>4:
        raise ValueError("PBW oracle deliberately limited to level two on each edge")
    ext = tuple(ground_state(0) if s=="R" else GWORD if star else ()
                for s,star in zip(SECTORS[channel],stars))
    phase = cmath.exp(1j*math.pi/4)
    def ns(a,m,d,ha,hm,hd):
        return complex(_ns_three_point_numeric_template(a,m,d)(ha,hm,hd,c))
    def rnsr(a,m,d,ha,hm,hd,pout,pin,sign):
        tensor = hjs_to_polynomial_ground_tensor(1j*pout/math.sqrt(2),1j*pin/math.sqrt(2),structure_sign=sign)
        entries = tuple(complex(tensor[i,j]) for i in range(2) for j in range(2))
        return complex(_rr_three_point_numeric_template(a,m,d)(ha,hm,hd,c,*entries))
    def nsrr(a,m,d,ha,hm,hd,pm,pd,sign):
        bm,bd=1j*pm/math.sqrt(2),1j*pd/math.sqrt(2)
        terminal=(1,phase*bd,sign*1j*phase*bm,sign*1j*bm*bd)
        return complex(_rr_ward_template(a,m,d)(ha,hm,hd,c,*terminal))
    e1,e2,e3,e4,e5=ext
    p1,p2,p3,p4,p5=pe
    h1,h2,h3,h4,h5=he
    ha,hb=hi
    coefficients={}
    for i,j in product(*levels):
        aa,ia=_gram(EDGES[channel][0],i,c,ha,pa[0])
        bb,ib=_gram(EDGES[channel][1],j,c,hb,pa[1])
        if channel=="A":
            right=np.array([ns(a,e2,e1,ha,h2,h1) for a in aa])
            middle=np.array([[ns(d,e3,a,hb,h3,ha) for a in aa] for d in bb])
            left=np.array([nsrr(d,e4,e5,hb,h4,h5,p4,p5,signs[2]) for d in bb])
        elif channel=="B":
            right=np.array([rnsr(a,e2,e1,ha,h2,h1,pi[0],p1,signs[0]) for a in aa])
            middle=np.array([[nsrr(d,e3,a,hb,h3,ha,p3,pi[0],signs[1]) for a in aa] for d in bb])
            left=np.array([ns(e5,e4,d,h5,h4,hb) for d in bb])
        else:
            right=np.array([rnsr(a,e2,e1,ha,h2,h1,pi[0],p1,signs[0]) for a in aa])
            middle=np.array([[rnsr(d,e3,a,hb,h3,ha,pi[1],pi[0],signs[1]) for a in aa] for d in bb])
            left=np.array([rnsr(e5,e4,d,h5,h4,hb,p5,pi[1],signs[2]) for d in bb])
        coefficients[i,j]=complex(left@ib@middle@ia@right)*((-1j)**pa[1] if channel=="A" else 1)
    return MixedFivePointSeries(coefficients,channel,b,hi,he,stars,pa,signs,cut,"direct PBW oracle, level<=2")
