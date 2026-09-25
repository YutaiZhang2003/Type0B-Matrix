#!/usr/bin/env python3
"""Fast double-precision backend for heterotic_so23_1to3.py.

It preserves the formulas and quadrature layout of the reference implementation,
but evaluates Barnes/Gamma functions with mpmath.fp and the c-recursion in native
complex arithmetic.  It is intended for dense scans and analytic fitting; final
points should be spot-checked against the high-precision backend.
"""
from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import Any, MutableMapping, Sequence

import mpmath as mp
import numpy as np
from numpy.polynomial.legendre import leggauss

if __package__:
    from . import heterotic_so23_1to3 as ref
else:
    import heterotic_so23_1to3 as ref


class FastRecursionFailure(ArithmeticError):
    """Raised when the binary64 recursion cannot produce finite coefficients."""


def _isfinite_complex(value: complex) -> bool:
    value = complex(value)
    return math.isfinite(value.real) and math.isfinite(value.imag)


def h_of_p(p: complex) -> complex:
    p = complex(p)
    return (1.0 + p*p)/2.0


def gamma_ratio(x: complex) -> complex:
    x = complex(x)
    return complex(mp.fp.gamma(x) / mp.fp.gamma(1-x))


def log_gamma_ratio(x: complex) -> complex:
    x = complex(x)
    return complex(mp.fp.loggamma(x) - mp.fp.loggamma(1-x))


def _upsilon_1_barnes_direct(x: complex) -> complex:
    x = complex(x)
    return complex(mp.fp.barnesg(x) * mp.fp.barnesg(2-x))


def log_upsilon_1(x: complex) -> complex:
    """Binary64 log-Upsilon using Upsilon_1(x+1)=gamma(x)Upsilon_1(x)."""
    x = complex(x)
    shift = math.floor(x.real - 0.5)
    base = x - shift
    result = cmath.log(mp.fp.barnesg(base)) + cmath.log(mp.fp.barnesg(2-base))
    if shift > 0:
        for offset in range(shift):
            result += log_gamma_ratio(base + offset)
    elif shift < 0:
        for offset in range(-shift):
            result -= log_gamma_ratio(x + offset)
    return result


def upsilon_1(x: complex) -> complex:
    return cmath.exp(log_upsilon_1(x))


def log_upsilon_ns(x: complex) -> complex:
    x = complex(x)
    return log_gamma_ratio(x/2) + 2*log_upsilon_1(x/2)


def log_upsilon_r(x: complex) -> complex:
    x = complex(x)
    return 2*log_upsilon_1((x+1)/2)


def upsilon_ns(x: complex) -> complex:
    return cmath.exp(log_upsilon_ns(x))


def upsilon_r(x: complex) -> complex:
    return cmath.exp(log_upsilon_r(x))


def _log_ns_leg(momentum: complex) -> complex:
    momentum = complex(momentum)
    return complex(
        mp.fp.loggamma(1+1j*momentum)
        - mp.fp.loggamma(1-1j*momentum)
    ) + log_upsilon_ns(2j*momentum)


def log_c_even(p1: complex, p2: complex, p3: complex) -> complex:
    ps = [complex(p1),complex(p2),complex(p3)]
    total = sum(ps)
    result = cmath.log(0.5j) - log_upsilon_ns(1+1j*total)
    for pi in ps:
        result += _log_ns_leg(pi)
        result -= log_upsilon_ns(1+1j*(total-2*pi))
    return result


def log_c_odd(p1: complex, p2: complex, p3: complex) -> complex:
    ps = [complex(p1),complex(p2),complex(p3)]
    total = sum(ps)
    result = cmath.log(1j) - log_upsilon_r(1+1j*total)
    for pi in ps:
        result += _log_ns_leg(pi)
        result -= log_upsilon_r(1+1j*(total-2*pi))
    return result


def c_even(p1: complex, p2: complex, p3: complex) -> complex:
    return cmath.exp(log_c_even(p1,p2,p3))


def c_odd(p1: complex, p2: complex, p3: complex) -> complex:
    return cmath.exp(log_c_odd(p1,p2,p3))


def _rf(a: complex, n: int) -> complex:
    out = 1+0j
    for k in range(n):
        out *= a+k
    return out


def _seed_coefficient(level2: int, h4: complex, h3: complex, h2: complex,
                      h1: complex, h: complex, star3=False, star2=False) -> complex:
    if level2 % 2 == 0:
        n = level2//2
        a3 = h+h3-h4+(0.5 if star3 else 0.0)
        a2 = h+h2-h1+(0.5 if star2 else 0.0)
        return _rf(a3,n)*_rf(a2,n)/(math.factorial(n)*_rf(2*h,n))
    n=(level2-1)//2
    if not star3 and not star2:
        return _rf(h+h3-h4+0.5,n)*_rf(h+h2-h1+0.5,n)/(math.factorial(n)*_rf(2*h,n+1))
    if not star3 and star2:
        return _rf(h+h3-h4+0.5,n)*_rf(h+h2-h1,n+1)/(math.factorial(n)*_rf(2*h,n+1))
    if star3 and not star2:
        return -_rf(h+h3-h4,n+1)*_rf(h+h2-h1+0.5,n)/(math.factorial(n)*_rf(2*h,n+1))
    return -_rf(h+h3-h4,n+1)*_rf(h+h2-h1,n+1)/(math.factorial(n)*_rf(2*h,n+1))


def _c_rs_and_derivative(h: complex,r:int,s:int):
    disc=16*h*h+8*(r*s-1)*h+(r-s)**2
    root=cmath.sqrt(disc)
    y=-(4*h+r*s-1+root)/(r*r-1)
    dy=-(4+(16*h+4*(r*s-1))/root)/(r*r-1)
    c_rs=7.5+3*y+3/y
    dc=3*dy*(1-1/(y*y))
    return c_rs,dc,cmath.sqrt(y)


def _a_rs(b: complex,r:int,s:int) -> complex:
    prod=1+0j
    for p in range(1-r,r+1):
        for q in range(1-s,s+1):
            if (p+q)%2: continue
            if (p==0 and q==0) or (p==r and q==s): continue
            prod *= math.sqrt(2)/(p*b+q/b)
    return prod/2


def _a_from_h(h:complex,Q:complex)->complex:
    return cmath.sqrt(Q*Q/4-2*h)


def _fusion_polynomial(h_a:complex,h_b:complex,star_b:bool,b:complex,r:int,s:int)->complex:
    Q=b+1/b
    a1=_a_from_h(h_a,Q); a2=_a_from_h(h_b,Q)
    target=0 if star_b else 2
    prod=1+0j
    for p in range(1-r,r,2):
        for q in range(1-s,s,2):
            if ((p+q)-(r+s))%4 != target: continue
            x=p*b+q/b
            prod *= (2*a1-2*a2-x)/(2*math.sqrt(2))
            prod *= (2*a1+2*a2+x)/(2*math.sqrt(2))
    return prod


@dataclass
class FastNSBlockComputer:
    h4: complex
    h3: complex
    h2: complex
    h1: complex
    star3: bool=False
    star2: bool=False
    c: complex=13.5
    max_level2: int=16

    def __post_init__(self):
        self.h4=complex(self.h4); self.h3=complex(self.h3)
        self.h2=complex(self.h2); self.h1=complex(self.h1); self.c=complex(self.c)
        self._memo={}
        self._condition_memo={}
        self.max_condition=1.0

    def coefficient(self,level2:int,h:complex,c:complex|None=None)->complex:
        if level2==0: return 1+0j
        h=complex(h); c=self.c if c is None else complex(c)
        key=(int(level2),h,c)
        if key in self._memo: return self._memo[key]
        value=_seed_coefficient(level2,self.h4,self.h3,self.h2,self.h1,h,self.star3,self.star2)
        absolute_sum=abs(value)
        for r in range(2,level2+1):
            for s in range(1,level2+1):
                rs=r*s
                if rs<=1 or rs>level2 or (r+s)%2!=0: continue
                c_rs,dc,b=_c_rs_and_derivative(h,r,s)
                sign=(-1)**rs if self.star3 else 1
                if level2%2==0:
                    star12=self.star2; star43=self.star3
                else:
                    star12=not self.star2; star43=not self.star3
                p12=_fusion_polynomial(self.h1,self.h2,star12,b,r,s)
                p43=_fusion_polynomial(self.h4,self.h3,star43,b,r,s)
                residue=sign*(-dc)*_a_rs(b,r,s)*p12*p43
                denominator=c-c_rs
                if denominator == 0:
                    raise FastRecursionFailure(
                        f"binary64 recursion encountered c-c_rs=0 at level2={level2}, "
                        f"(r,s)=({r},{s})"
                    )
                term=residue/denominator*self.coefficient(level2-rs,h+rs/2,c_rs)
                if not _isfinite_complex(term):
                    raise FastRecursionFailure(
                        f"nonfinite binary64 recursion term at level2={level2}, "
                        f"(r,s)=({r},{s})"
                    )
                absolute_sum += abs(term)
                value += term
        if not _isfinite_complex(value):
            raise FastRecursionFailure(
                f"nonfinite binary64 recursion coefficient at level2={level2}"
            )
        condition=absolute_sum/max(abs(value),np.finfo(float).tiny)
        self._condition_memo[key]=condition
        self.max_condition=max(self.max_condition,float(condition))
        self._memo[key]=value
        return value


def _preflight_fast_block(
    block: FastNSBlockComputer,
    h_internal: complex,
    max_level2: int,
) -> float:
    """Evaluate every required coefficient and return its worst cancellation ratio."""
    for level2 in range(1,max_level2+1):
        block.coefficient(level2,h_internal)
    return block.max_condition


def _record_backend(
    diagnostics: MutableMapping[str, Any] | None,
    backend: str,
    condition: float | None,
    reason: str,
) -> None:
    if diagnostics is None:
        return
    diagnostics["block_node_pairs"] = int(diagnostics.get("block_node_pairs",0))+1
    key="reference_block_node_pairs" if backend == "reference" else "fast_block_node_pairs"
    diagnostics[key]=int(diagnostics.get(key,0))+1
    if condition is not None and math.isfinite(condition):
        diagnostics["max_fast_recursion_condition"]=max(
            float(diagnostics.get("max_fast_recursion_condition",1.0)),float(condition)
        )
    reason_key=f"block_backend_reason_{reason}"
    diagnostics[reason_key]=int(diagnostics.get(reason_key,0))+1


def _select_block_pair(
    *,
    h4: complex,
    h3: complex,
    h2: complex,
    h1: complex,
    h_internal: complex,
    p: float,
    max_level2: int,
    reference_p_max: float,
    cancellation_limit: float,
    diagnostics: MutableMapping[str, Any] | None,
    block_cache: MutableMapping[tuple[Any,...],tuple[Any,Any,str,float|None,int]]|None=None,
):
    """Choose binary64 or high-precision recursion after a coefficient preflight."""
    cache_key=_block_cache_key(
        h4,h3,h2,h1,p,reference_p_max,cancellation_limit
    )
    if block_cache is not None and cache_key in block_cache:
        primary,starstar,backend,condition,cached_level2=block_cache[cache_key]
        if cached_level2 >= max_level2:
            _record_backend(diagnostics,backend,condition,"cache")
            return primary,starstar,backend

    def finish(primary,starstar,backend,condition,reason):
        _record_backend(diagnostics,backend,condition,reason)
        if block_cache is not None:
            block_cache[cache_key]=(
                primary,starstar,backend,condition,int(max_level2)
            )
        return primary,starstar,backend

    if p <= reference_p_max:
        return finish(
            ref.NSBlockComputer(h4,h3,h2,h1,False,False,max_level2=max_level2),
            ref.NSBlockComputer(h4,h3,h2,h1,True,True,max_level2=max_level2),
            "reference",None,"low_p",
        )

    primary=FastNSBlockComputer(h4,h3,h2,h1,False,False,max_level2=max_level2)
    starstar=FastNSBlockComputer(h4,h3,h2,h1,True,True,max_level2=max_level2)
    try:
        primary_condition=_preflight_fast_block(primary,h_internal,max_level2)
        star_condition=_preflight_fast_block(starstar,h_internal,max_level2)
        condition=max(primary_condition,star_condition)
    except (ArithmeticError,OverflowError,ValueError,ZeroDivisionError):
        return finish(
            ref.NSBlockComputer(h4,h3,h2,h1,False,False,max_level2=max_level2),
            ref.NSBlockComputer(h4,h3,h2,h1,True,True,max_level2=max_level2),
            "reference",None,"fast_exception",
        )

    if not math.isfinite(condition) or condition > cancellation_limit:
        return finish(
            ref.NSBlockComputer(h4,h3,h2,h1,False,False,max_level2=max_level2),
            ref.NSBlockComputer(h4,h3,h2,h1,True,True,max_level2=max_level2),
            "reference",condition,"cancellation",
        )

    return finish(primary,starstar,"fast",condition,"validated_fast")


def _block_cache_key(
    h4: complex,
    h3: complex,
    h2: complex,
    h1: complex,
    p: float,
    reference_p_max: float,
    cancellation_limit: float,
) -> tuple[Any,...]:
    """Return the complete numerical-policy key for one recursion pair."""
    return (
        complex(h4),complex(h3),complex(h2),complex(h1),float(p),
        float(reference_p_max),float(cancellation_limit),
    )


@dataclass
class PKernel:
    p: float
    weight: float
    primary: ref.SewingNSBlock
    starstar: ref.SewingNSBlock
    even_structure: complex
    odd_structure: complex
    backend: str="fast"
    recursion_condition: float|None=None



def _p_quadrature(p_nodes,p_max:float,p_cut:float=0.03,*,scheme="cutoff",threshold_options=None):
    """Composite quadrature for the internal Liouville momentum.

    The endpoint rule integrates the even Taylor basis 1,P^2,P^4,P^6
    exactly on [0,p_cut], avoiding loss of precision at P=0.  If
    ``p_nodes`` is an integer, a single Gauss--Legendre rule is used on
    [p_cut,p_max].  If it is ``"segmented"`` (or a sequence of seven
    integers), a piecewise rule resolves the moderately narrow structures
    that develop near P≈0.4 on the Figure-3 continuation path.
    """
    if scheme == "threshold_weighted":
        rule=ref.threshold_weighted_rule(p_nodes,threshold_options)
        return rule.momenta,rule.weights
    if scheme != "cutoff" or threshold_options is not None:
        raise ValueError("vector momentum quadrature needs threshold_weighted or explicit cutoff without threshold options")
    fractions=np.array([0.20,0.45,0.70,0.95],dtype=float)
    endpoint=p_cut*fractions
    powers=[0,2,4,6]
    matrix=np.vstack([endpoint**k for k in powers])
    rhs=np.array([p_cut**(k+1)/(k+1) for k in powers],dtype=float)
    endpoint_weights=np.linalg.solve(matrix,rhs)

    if isinstance(p_nodes,str):
        if p_nodes.lower() != 'segmented':
            raise ValueError("p_nodes string must be 'segmented'")
        counts=[16,18,24,18,16,14,14]
    elif isinstance(p_nodes,(tuple,list,np.ndarray)):
        counts=list(map(int,p_nodes))
        if len(counts)!=7:
            raise ValueError('segmented p_nodes must contain seven counts')
    else:
        nodes,weights=leggauss(int(p_nodes))
        bulk=(p_cut+p_max)/2+(p_max-p_cut)*nodes/2
        bulk_weights=(p_max-p_cut)*weights/2
        return np.concatenate([endpoint,bulk]),np.concatenate([endpoint_weights,bulk_weights])

    # The first six break points isolate the peak region while keeping a
    # modest number of nodes in the rapidly decaying tail.  The final point
    # is p_max, so the same rule works for nearby choices of the cutoff.
    fixed=[p_cut,0.18,0.36,0.56,0.85,1.30,2.10]
    if p_max <= fixed[-1]:
        raise ValueError('segmented quadrature requires p_max > 2.10')
    bounds=fixed+[float(p_max)]
    pieces=[endpoint]; wp=[endpoint_weights]
    for lo,hi,n in zip(bounds[:-1],bounds[1:],counts):
        x,w=leggauss(n)
        pieces.append((lo+hi)/2+(hi-lo)*x/2)
        wp.append((hi-lo)*w/2)
    return np.concatenate(pieces),np.concatenate(wp)

def _canonical_pair(first: complex,second: complex) -> tuple[complex,complex]:
    """Return an order-independent key for a symmetric three-point constant."""
    values=(complex(first),complex(second))
    return tuple(sorted(values,key=lambda value:(value.real,value.imag)))


def _single_structure_constants(
    first: complex,
    second: complex,
    momentum: float,
    cache: MutableMapping[tuple[Any,...],tuple[complex,complex]]|None,
) -> tuple[complex,complex]:
    pair=_canonical_pair(first,second)
    key=(pair,float(momentum))
    if cache is not None and key in cache:
        return cache[key]
    try:
        even=c_even(pair[0],pair[1],momentum)
        odd=c_odd(pair[0],pair[1],momentum)
        if not (_isfinite_complex(even) and _isfinite_complex(odd)):
            raise FastRecursionFailure("nonfinite binary64 structure constant")
    except (ArithmeticError,OverflowError,ValueError,ZeroDivisionError):
        even,odd=(
            complex(ref.c_even(pair[0],pair[1],momentum)),
            complex(ref.c_odd(pair[0],pair[1],momentum)),
        )
    result=(even,odd)
    if cache is not None:
        cache[key]=result
    return result


def _structure_products(
    left: tuple[complex,complex,float],
    right: tuple[complex,complex,float],
    cache: MutableMapping[tuple[Any,...],tuple[complex,complex]]|None=None,
) -> tuple[complex,complex]:
    """Evaluate products of symmetric Liouville structure constants."""
    left_even,left_odd=_single_structure_constants(*left,cache)
    right_even,right_odd=_single_structure_constants(*right,cache)
    return left_even*right_even,left_odd*right_odd


def build_s_channel_data(
    energies:Sequence[complex],
    q_order:int,
    p_nodes:int,
    p_max:float,
    p_cut:float=0.03,
    reference_p_max:float=0.18,
    cancellation_limit:float=1.0e7,
    diagnostics:MutableMapping[str,Any]|None=None,
    structure_cache:MutableMapping[tuple[Any,...],tuple[complex,complex]]|None=None,
    block_cache:MutableMapping[tuple[Any,...],tuple[Any,Any,str,float|None,int]]|None=None,
    elliptic_cache:MutableMapping[tuple[Any,...],tuple[Any,Any,str,float|None]]|None=None,
    momentum_scheme="cutoff",momentum_threshold_options=None,
    series_parameter="elliptic_nome",
):
    ref.elliptic_conversion.validate_representation(series_parameter)
    if diagnostics is not None:
        diagnostics["series_parameter"] = series_parameter
        diagnostics["q_definition"] = ("qhat=exp(-pi*K(1-z)/K(z)); plane measure" if series_parameter == "elliptic_nome" else "q_s=z; q_t=1-z")
        diagnostics["numerical_algorithm"] = (ref.elliptic_conversion.ALGORITHM_VERSION if series_parameter == "elliptic_nome" else "ns_c_recursion_sewing_v1")
        diagnostics["local_ope_policy"] = ("back_expansion_of_truncated_nome" if series_parameter == "elliptic_nome" else "truncated_sewing")
    w1,w2,w3,w0=map(complex,energies)
    h1,h2,h3,h4=map(h_of_p,(w1,w2,w3,w0))
    ps,ws=_p_quadrature(p_nodes,p_max,p_cut,scheme=momentum_scheme,threshold_options=momentum_threshold_options)
    if diagnostics is not None:
        diagnostics["momentum_quadrature"]=(ref.threshold_weighted_rule(p_nodes,momentum_threshold_options).metadata()
            if momentum_scheme == "threshold_weighted" else {"scheme":momentum_scheme,"node_count":len(ps)})
    data=[]
    for p,w in zip(ps,ws):
        p=float(p); hp=h_of_p(p)
        elliptic_key=(
            series_parameter, ref.elliptic_conversion.ALGORITHM_VERSION,
            _block_cache_key(
                h4,h3,h2,h1,p,reference_p_max,cancellation_limit
            ),
            int(q_order),
        )
        if elliptic_cache is not None and elliptic_key in elliptic_cache:
            primary,starstar,backend,condition=elliptic_cache[elliptic_key]
            _record_backend(diagnostics,backend,condition,"plane_block_cache")
        else:
            pc,sc,backend=_select_block_pair(
                h4=h4,h3=h3,h2=h2,h1=h1,h_internal=hp,p=p,
                max_level2=2*q_order+1,reference_p_max=reference_p_max,
                cancellation_limit=cancellation_limit,diagnostics=diagnostics,
                block_cache=block_cache,
            )
            condition=max(pc.max_condition,sc.max_condition) if backend == "fast" else None
            primary=ref.plane_block(pc,hp,q_order,series_parameter)
            starstar=ref.plane_block(sc,hp,q_order,series_parameter)
            if elliptic_cache is not None:
                elliptic_cache[elliptic_key]=(primary,starstar,backend,condition)
        ce,co=_structure_products((w1,w2,p),(w3,w0,p),structure_cache)
        data.append(PKernel(p,float(w)/math.pi,
            primary,starstar,
            ce,co,backend,condition))
    return data


@dataclass
class DirectBlockSeries:
    """Precomputed direct-channel polynomial data for one recursion block."""

    exponent: complex
    even: np.ndarray
    odd: np.ndarray
    odd_sign: int


def _build_direct_block_series(
    block: Any,
    h_internal: complex,
    order: int,
) -> DirectBlockSeries:
    """Materialize coefficients that :func:`ref._direct_block` rebuilds per call."""
    h=complex(h_internal)
    h1=complex(block.h1)
    h2=complex(block.h2)
    h2a=h2+(0.5 if block.star2 else 0.0)
    even=np.asarray(
        [1+0j]+[complex(block.coefficient(2*n,h)) for n in range(1,order+1)],
        dtype=complex,
    )
    odd=np.asarray(
        [complex(block.coefficient(2*n+1,h)) for n in range(order+1)],
        dtype=complex,
    )
    return DirectBlockSeries(
        exponent=h-h1-h2a,
        even=even,
        odd=odd,
        odd_sign=-1 if block.star2 and block.star3 else 1,
    )


def _cached_direct_block_series(
    block: Any,
    h_internal: complex,
    order: int,
    cache: MutableMapping[tuple[int,int],DirectBlockSeries]|None,
) -> DirectBlockSeries:
    key=(id(block),int(order))
    if cache is not None and key in cache:
        return cache[key]
    series=_build_direct_block_series(block,h_internal,order)
    if cache is not None:
        cache[key]=series
    return series


@dataclass
class TKernel:
    p: float
    weight: float
    primary: Any
    starstar: Any
    even_structure: complex
    odd_structure: complex
    h_internal: complex
    primary_series: DirectBlockSeries
    starstar_series: DirectBlockSeries
    backend: str="fast"
    recursion_condition: float|None=None
    resummed_primary: ref.ResummedNSBlock|None=None
    resummed_adjacent: Any=None


def build_t_channel_data(
    energies:Sequence[complex],
    series_order:int,
    p_nodes:int,
    p_max:float,
    p_cut:float=0.03,
    reference_p_max:float=0.18,
    cancellation_limit:float=1.0e7,
    diagnostics:MutableMapping[str,Any]|None=None,
    structure_cache:MutableMapping[tuple[Any,...],tuple[complex,complex]]|None=None,
    block_cache:MutableMapping[tuple[Any,...],tuple[Any,Any,str,float|None,int]]|None=None,
    direct_series_cache:MutableMapping[tuple[int,int],DirectBlockSeries]|None=None,
    momentum_scheme="cutoff",momentum_threshold_options=None,
    series_parameter="elliptic_nome",
):
    ref.elliptic_conversion.validate_representation(series_parameter)
    w1,w2,w3,w0=map(complex,energies)
    h1,h2,h3,h4=map(h_of_p,(w1,w2,w3,w0))
    ps,ws=_p_quadrature(p_nodes,p_max,p_cut,scheme=momentum_scheme,threshold_options=momentum_threshold_options)
    data=[]
    for p,w in zip(ps,ws):
        p=float(p); hp=h_of_p(p)
        pc,sc,backend=_select_block_pair(
            h4=h4,h3=h1,h2=h2,h1=h3,h_internal=hp,p=p,
            max_level2=2*series_order+1,reference_p_max=reference_p_max,
            cancellation_limit=cancellation_limit,diagnostics=diagnostics,
            block_cache=block_cache,
        )
        condition=max(pc.max_condition,sc.max_condition) if backend == "fast" else None
        ce,co=_structure_products((w2,w3,p),(w1,w0,p),structure_cache)
        primary_series=_cached_direct_block_series(
            pc,hp,series_order,direct_series_cache
        )
        starstar_series=_cached_direct_block_series(
            sc,hp,series_order,direct_series_cache
        )
        data.append(TKernel(
            p,float(w)/math.pi,pc,sc,ce,co,complex(hp),
            primary_series,starstar_series,backend,condition
        ))
        if series_parameter == "elliptic_nome":
            data[-1].resummed_primary=ref.ResummedNSBlock(pc,hp,series_order)
            data[-1].resummed_adjacent=data[-1].resummed_primary.adjacent_component(
                ref.ResummedNSBlock(sc,hp,series_order))
    return data


def _integrate_one_p_kernel(kernel:PKernel,energies,z_data,geometry=None):
    return ref._integrate_one_p_kernel(kernel,energies,z_data,geometry)


def integrate_s_channel(data,energies,z_data):
    total=np.zeros(3,dtype=complex)
    geometry=(ref.elliptic_conversion.nome_geometry(z_data[0])
              if data and data[0].primary.series_parameter == "elliptic_nome" else None)
    for k in data: total += k.weight*_integrate_one_p_kernel(k,energies,z_data,geometry)
    return total


def full_plane_rest(original_data,swapped_data,energies,epsilon,theta_orders,radial_order,
                    original_t=None,swapped_t=None,order=None):
    zdata=ref._unit_disk_excluding_lens(epsilon,theta_orders,radial_order,sewing=True)
    o=sewing_bulk_integral(original_data,original_t,energies,zdata,order)
    se=[energies[0],energies[2],energies[1],energies[3]]
    s=sewing_bulk_integral(swapped_data,swapped_t,se,zdata,order)
    return np.array([o[0]+s[1],o[1]+s[0],o[2]+s[2]])


def _evaluate_direct_series(
    series: DirectBlockSeries,
    z: np.ndarray,
    log_z: np.ndarray,
    parity: str,
    *,
    derivative: bool=False,
) -> np.ndarray:
    if parity == "e":
        coefficients=series.even
        shift=0.0
        sign=1
    else:
        coefficients=series.odd
        shift=0.5
        sign=series.odd_sign
    exponent=series.exponent+shift
    if derivative:
        powers=exponent+np.arange(coefficients.size)
        polynomial=np.polynomial.polynomial.polyval(z,powers*coefficients)
        exponent-=1
    else:
        polynomial=np.polynomial.polynomial.polyval(z,coefficients)
    return sign*np.exp(exponent*log_z)*polynomial


def _t_kernel_integral(kernel:TKernel,energies,w,weights,order,geometry=None):
    if kernel.resummed_primary is not None:
        return ref._resummed_t_kernel_integral(kernel,energies,w,weights,order,geometry)
    w1,w2,w3,_=map(complex,energies); wb=np.conj(w)
    logw=np.log(w); logwb=np.log(wb)
    primary=kernel.primary_series; starstar=kernel.starstar_series
    fe=_evaluate_direct_series(primary,w,logw,'e')
    fo=_evaluate_direct_series(primary,w,logw,'o')
    feb=_evaluate_direct_series(primary,wb,logwb,'e')
    fob=_evaluate_direct_series(primary,wb,logwb,'o')
    fes=_evaluate_direct_series(starstar,w,logw,'e')
    fos=_evaluate_direct_series(starstar,w,logw,'o')
    pe=fos-_evaluate_direct_series(primary,w,logw,'e',derivative=True)
    po=fes-_evaluate_direct_series(primary,w,logw,'o',derivative=True)
    G=kernel.even_structure*fe*feb+kernel.odd_structure*fo*fob
    L=kernel.even_structure*pe*feb+kernel.odd_structure*po*fob
    bracket=L+(w2*w3)/w*G
    z=1-w
    tf=np.exp(-2*w1*w2*np.log(np.abs(z))-2*w2*w3*np.log(np.abs(w)))
    base=weights*tf*bracket
    return np.array([np.sum(base/(1-wb)),np.sum(-base),np.sum(base/wb)])


def sewing_bulk_integral(s_data,t_data,energies,zdata,order):
    """Choose the smaller expansion coordinate; retain the plane d^2z measure."""
    if t_data is None:
        return integrate_s_channel(s_data,energies,zdata)
    z,weights=zdata[:2]
    resummed=bool(s_data and s_data[0].primary.series_parameter == "elliptic_nome")
    if resummed:
        use_t=abs(ref.elliptic_conversion.nome_geometry(1-z)[0])<abs(ref.elliptic_conversion.nome_geometry(z)[0])
    else:
        use_t=np.abs(1-z)<np.abs(z)
    total=np.zeros(3,dtype=complex)
    if np.any(~use_t):
        total += integrate_s_channel(s_data,energies,(z[~use_t],weights[~use_t]))
    if np.any(use_t):
        geometry=ref.elliptic_conversion.nome_geometry(1-z[use_t]) if resummed else None
        for kernel in t_data:
            total += kernel.weight*_t_kernel_integral(kernel,energies,1-z[use_t],weights[use_t],order,geometry)
    return total


def lens_integral(data,energies,epsilon,order,radial_order,angular_order,radial_power):
    w,weights=ref._lens_grid(epsilon,radial_order,angular_order,radial_power)
    total=np.zeros(3,dtype=complex)
    geometry=ref.elliptic_conversion.nome_geometry(w) if data and data[0].resummed_primary is not None else None
    for k in data: total += k.weight*_t_kernel_integral(k,energies,w,weights,order,geometry)
    return total


def full_lens_integral(original_data,swapped_data,energies,epsilon,order,radial_order,angular_order,radial_power):
    o=lens_integral(original_data,energies,epsilon,order,radial_order,angular_order,radial_power)
    se=[energies[0],energies[2],energies[1],energies[3]]
    s=lens_integral(swapped_data,se,epsilon,order,radial_order,angular_order,radial_power)
    return np.array([o[0]+s[1],o[1]+s[0],o[2]+s[2]])


def one_ordering(energies:Sequence[complex],*,q_order=6,p_nodes=32,p_max=5.0,
                 epsilon=.05,theta_orders=(64,64,160),radial_order=56,
                 lens_radial_order=48,lens_angular_order=120,lens_power=3.0,
                 p_cut=.03,reference_p_max=.18,cancellation_limit=1.0e7,
                 diagnostics=None,structure_cache=None,block_cache=None,
                 elliptic_cache=None,direct_series_cache=None,
                 momentum_scheme="threshold_weighted",momentum_threshold_options=None,
                 series_parameter="elliptic_nome"):
    swapped=[energies[0],energies[2],energies[1],energies[3]]
    backend_options=dict(
        series_parameter=series_parameter,
        momentum_scheme=momentum_scheme,momentum_threshold_options=momentum_threshold_options,
        p_cut=p_cut,reference_p_max=reference_p_max,
        cancellation_limit=cancellation_limit,diagnostics=diagnostics,
        structure_cache=structure_cache,block_cache=block_cache,
    )
    s=build_s_channel_data(
        energies,q_order,p_nodes,p_max,
        elliptic_cache=elliptic_cache,**backend_options,
    )
    ss=build_s_channel_data(
        swapped,q_order,p_nodes,p_max,
        elliptic_cache=elliptic_cache,**backend_options,
    )
    t=build_t_channel_data(
        energies,q_order,p_nodes,p_max,
        direct_series_cache=direct_series_cache,**backend_options,
    )
    ts=build_t_channel_data(
        swapped,q_order,p_nodes,p_max,
        direct_series_cache=direct_series_cache,**backend_options,
    )
    return full_plane_rest(s,ss,energies,epsilon,theta_orders,radial_order,t,ts,q_order)+full_lens_integral(
        t,ts,energies,epsilon,q_order,lens_radial_order,lens_angular_order,lens_power)


def vector_amplitude_coefficients(energies:Sequence[complex],**kwargs):
    if abs(complex(energies[3]-sum(energies[:3])))>1e-11:
        raise ValueError('energy conservation required')
    kwargs=dict(kwargs)
    kwargs.setdefault("structure_cache",{})
    kwargs.setdefault("block_cache",{})
    kwargs.setdefault("elliptic_cache",{})
    kwargs.setdefault("direct_series_cache",{})
    o=one_ordering(energies,**kwargs)
    cyc=[energies[1],energies[2],energies[0],energies[3]]
    c=one_ordering(cyc,**kwargs)
    return np.array([o[0],o[1],c[0]])


def equal_outgoing_coefficient(omega0:complex,**kwargs)->complex:
    energies=[omega0/3]*3+[omega0]
    return one_ordering(energies,**kwargs)[0]

# ---------------------------------------------------------------------------
# Improved small-|z| treatment for dense scans
# ---------------------------------------------------------------------------

def sewing_annulus_grid(epsilon0:float,epsilon1:float,theta_orders,radial_order):
    """Plane-area quadrature, with no elliptic or pillow coordinate factors.

    q_s=z on epsilon0 <= |z| <= 1, excluding |1-z|<epsilon1.
    """
    theta_inner=2*np.arcsin(epsilon1/2)
    theta_outer=np.arcsin(epsilon1)
    intervals=[(-np.pi,-theta_outer),(-theta_outer,-theta_inner),(-theta_inner,0),
               (0,theta_inner),(theta_inner,theta_outer),(theta_outer,np.pi)]
    orders=[theta_orders[2],theta_orders[1],theta_orders[0],theta_orders[0],theta_orders[1],theta_orders[2]]
    rn,rw=leggauss(radial_order)
    zparts=[]; wparts=[]
    for (a,b),order in zip(intervals,orders):
        tn,tw=leggauss(order); th=(a+b)/2+(b-a)*tn/2; thw=(b-a)*tw/2
        for angle,aw in zip(th,thw):
            co=np.cos(angle); si=np.sin(angle); allowed=[]
            if abs(si)>=epsilon1 or co<=0:
                allowed=[(epsilon0,1.0)]
            else:
                delta=np.sqrt(max(0.0,epsilon1*epsilon1-si*si))
                rminus=max(0.0,co-delta); rplus=min(1.0,co+delta)
                if rminus>epsilon0: allowed.append((epsilon0,rminus))
                if rplus<1: allowed.append((max(epsilon0,rplus),1.0))
            for r0,r1 in allowed:
                if r1<=r0: continue
                # Split at Re(z)=1/2 so a quadrature panel never crosses the
                # s/t sewing-chart interface at finite block order.
                switch=0.5/co if co>0 else float("inf")
                cuts=[r0,switch,r1] if r0<switch<r1 else [r0,r1]
                for lo,hi in zip(cuts[:-1],cuts[1:]):
                    rr=(lo+hi)/2+(hi-lo)*rn/2; rrw=(hi-lo)*rw/2
                    zparts.append(rr*np.exp(1j*angle)); wparts.append(aw*rrw*rr)
    return np.concatenate(zparts),np.concatenate(wparts)


def annulus_excluding_lens(epsilon0:float,epsilon1:float,theta_orders,radial_order):
    """Legacy nome-grid utility, retained for independent historical checks."""
    z,weights=sewing_annulus_grid(epsilon0,epsilon1,theta_orders,radial_order)
    q,theta=ref._q_grid(z)
    return z,weights,q,theta


def full_plane_annulus(original_data,swapped_data,energies,epsilon0,epsilon1,theta_orders,radial_order,
                       original_t=None,swapped_t=None,order=None):
    zdata=sewing_annulus_grid(epsilon0,epsilon1,theta_orders,radial_order)
    o=sewing_bulk_integral(original_data,original_t,energies,zdata,order)
    se=[energies[0],energies[2],energies[1],energies[3]]
    s=sewing_bulk_integral(swapped_data,swapped_t,se,zdata,order)
    return np.array([o[0]+s[1],o[1]+s[0],o[2]+s[2]])


def disk_grid(epsilon:float,radial_order:int,angular_order:int,radial_power:float=2.5):
    rn,rw=leggauss(radial_order); x=(rn+1)/2; xw=rw/2
    rho=epsilon*x**radial_power
    rhow=epsilon*radial_power*x**(radial_power-1)*xw
    tn,tw=leggauss(angular_order); theta=np.pi*tn; thetaw=np.pi*tw
    z=(rho[:,None]*np.exp(1j*theta[None,:])).ravel()
    weights=(rhow[:,None]*thetaw[None,:]*rho[:,None]).ravel()
    return z,weights


def _s_direct_kernel_integral(kernel:PKernel,energies,z,weights,order):
    if kernel.primary.series_parameter == "elliptic_nome":
        if order != kernel.primary.q_order:
            raise ValueError("disk nome order must match the constructed kernel")
        return _integrate_one_p_kernel(kernel,energies,(z,weights))
    w1,w2,w3,_=map(complex,energies); zb=np.conj(z)
    pc=kernel.primary.block; sc=kernel.starstar.block; hp=complex(kernel.primary.h)
    fe=ref._direct_block(pc,hp,z,'e',order); fo=ref._direct_block(pc,hp,z,'o',order)
    fes=ref._direct_block(sc,hp,z,'e',order); fos=ref._direct_block(sc,hp,z,'o',order)
    feb=ref._direct_block(pc,hp,zb,'e',order); fob=ref._direct_block(pc,hp,zb,'o',order)
    G=kernel.even_structure*fe*feb+kernel.odd_structure*fo*fob
    L=kernel.even_structure*fos*feb+kernel.odd_structure*fes*fob
    bracket=L+(w2*w3)/(1-z)*G
    tf=np.exp(-2*w1*w2*np.log(np.abs(z))-2*w2*w3*np.log(np.abs(1-z)))
    base=weights*tf*bracket
    return np.array([np.sum(base/zb),np.sum(-base),np.sum(base/(1-zb))])


def s_disk_integral(data,energies,epsilon,order,radial_order,angular_order,radial_power=2.5):
    z,w=disk_grid(epsilon,radial_order,angular_order,radial_power)
    total=np.zeros(3,dtype=complex)
    for k in data: total += k.weight*_s_direct_kernel_integral(k,energies,z,w,order)
    return total


def full_s_disk_integral(original_data,swapped_data,energies,epsilon,order,radial_order,angular_order,radial_power=2.5):
    o=s_disk_integral(original_data,energies,epsilon,order,radial_order,angular_order,radial_power)
    se=[energies[0],energies[2],energies[1],energies[3]]
    s=s_disk_integral(swapped_data,se,epsilon,order,radial_order,angular_order,radial_power)
    return np.array([o[0]+s[1],o[1]+s[0],o[2]+s[2]])


def one_ordering_split(energies:Sequence[complex],*,q_order=5,p_nodes=60,p_max=5.0,
                       epsilon0=.08,epsilon1=.06,theta_orders=(28,28,72),radial_order=28,
                       disk_radial_order=24,disk_angular_order=64,disk_power=2.5,
                       lens_radial_order=24,lens_angular_order=64,lens_power=3.0,
                       p_cut=.03,reference_p_max=.18,cancellation_limit=1.0e7,
                       diagnostics=None,structure_cache=None,block_cache=None,
                       elliptic_cache=None,direct_series_cache=None,
                       momentum_scheme="threshold_weighted",momentum_threshold_options=None,
                       series_parameter="elliptic_nome"):
    swapped=[energies[0],energies[2],energies[1],energies[3]]
    backend_options=dict(
        series_parameter=series_parameter,
        momentum_scheme=momentum_scheme,momentum_threshold_options=momentum_threshold_options,
        p_cut=p_cut,reference_p_max=reference_p_max,
        cancellation_limit=cancellation_limit,diagnostics=diagnostics,
        structure_cache=structure_cache,block_cache=block_cache,
    )
    s=build_s_channel_data(
        energies,q_order,p_nodes,p_max,
        elliptic_cache=elliptic_cache,**backend_options,
    )
    ss=build_s_channel_data(
        swapped,q_order,p_nodes,p_max,
        elliptic_cache=elliptic_cache,**backend_options,
    )
    t=build_t_channel_data(
        energies,q_order,p_nodes,p_max,
        direct_series_cache=direct_series_cache,**backend_options,
    )
    ts=build_t_channel_data(
        swapped,q_order,p_nodes,p_max,
        direct_series_cache=direct_series_cache,**backend_options,
    )
    ann=full_plane_annulus(s,ss,energies,epsilon0,epsilon1,theta_orders,radial_order,t,ts,q_order)
    disk=full_s_disk_integral(s,ss,energies,epsilon0,q_order,disk_radial_order,disk_angular_order,disk_power)
    lens=full_lens_integral(t,ts,energies,epsilon1,q_order,lens_radial_order,lens_angular_order,lens_power)
    return ann+disk+lens


def equal_outgoing_coefficient_split(omega0:complex,**kwargs)->complex:
    E=[omega0/3]*3+[omega0]
    return one_ordering_split(E,**kwargs)[0]

# ---------------------------------------------------------------------------
# Analytic small-disk OPE integration (also supplies local continuation)
# ---------------------------------------------------------------------------

def _block_poly_and_exponent(block,h_internal:complex,parity:str,order:int):
    h=complex(h_internal); h1=complex(block.h1); h2=complex(block.h2)
    h2a=h2+(0.5 if block.star2 else 0.0)
    base=h-h1-h2a
    if parity=='e':
        coeff=np.array([1+0j]+[complex(block.coefficient(2*n,h)) for n in range(1,order+1)],dtype=complex)
        return base,coeff
    sign=-1 if block.star2 and block.star3 else 1
    coeff=np.array([sign*complex(block.coefficient(2*n+1,h)) for n in range(order+1)],dtype=complex)
    return base+0.5,coeff


def _binomial_minus_power(a:complex,order:int)->np.ndarray:
    """Coefficients of (1-z)^(-a) through z^order."""
    out=np.empty(order+1,dtype=complex); out[0]=1
    for n in range(order): out[n+1]=out[n]*(a+n)/(n+1)
    return out


def _conv_trunc(a:np.ndarray,b:np.ndarray,order:int)->np.ndarray:
    return np.convolve(a,b)[:order+1]


def _integrate_series_disk(hol:np.ndarray,anti:np.ndarray,ah:complex,ab:complex,epsilon:float)->complex:
    """Meromorphically integrate two local series over a full disk.

    At an exactly logarithmic radial power the continued integral means its
    finite part.  Thus ``epsilon**rho/rho`` is replaced by
    ``log(epsilon)`` at ``rho=0``.  Keeping this branch here (rather than in
    callers) makes direct collision disks safe at exact resonant kinematics
    as well as at the generic complex momenta used by the production scan.
    """
    delta=ah-ab
    k=int(round(delta.real))
    le=math.log(epsilon)

    def radial_integral(power:complex)->complex:
        power=complex(power)
        if abs(power)<=1e-13:
            return complex(le)
        return cmath.exp(power*le)/power

    if abs(delta-k)>2e-8:
        # For a non-integer spin, use the exact angular integral.  This branch
        # should not be needed for the single-valued heterotic integrand.
        total=0j
        for n,cn in enumerate(hol):
            for m,cm in enumerate(anti):
                spin=delta+n-m
                radial=ah+ab+n+m+2
                angular=(cmath.exp(2j*math.pi*spin)-1)/(1j*spin) if abs(spin)>1e-13 else 2*math.pi
                total += cn*cm*angular*radial_integral(radial)
        return total
    total=0j
    # ah+n = ab+m => m=n+k
    for n,cn in enumerate(hol):
        m=n+k
        if 0<=m<len(anti):
            radial=ah+ab+n+m+2
            total += 2*math.pi*cn*anti[m]*radial_integral(radial)
    return total


def _analytic_s_disk_one_kernel(kernel:PKernel,energies,epsilon:float,block_order:int,total_order:int)->np.ndarray:
    w1,w2,w3,_=map(complex,energies); a=w2*w3; t=w1*w2
    ee,pe=kernel.primary.local_data('e',block_order,total_order)
    eo,po=kernel.primary.local_data('o',block_order,total_order)
    ese,pse=kernel.starstar.local_data('e',block_order,total_order)
    eso,pso=kernel.starstar.local_data('o',block_order,total_order)

    # component data: (overall tensor sign, anti z shift, extra anti (1-z)^-1)
    comps=[(1.0,-1,0),(-1.0,0,0),(1.0,0,1)]
    out=np.zeros(3,dtype=complex)
    for j,(tsign,anti_shift,anti_extra) in enumerate(comps):
        # L, even structure: star-star odd x primary even
        bh=_binomial_minus_power(a,total_order)
        ba=_binomial_minus_power(a+anti_extra,total_order)
        hol=_conv_trunc(pso,bh,total_order); anti=_conv_trunc(pe,ba,total_order)
        out[j]+=tsign*kernel.even_structure*_integrate_series_disk(hol,anti,eso-t,ee-t+anti_shift,epsilon)
        # L, odd structure: star-star even x primary odd
        hol=_conv_trunc(pse,bh,total_order); anti=_conv_trunc(po,ba,total_order)
        out[j]+=tsign*kernel.odd_structure*_integrate_series_disk(hol,anti,ese-t,eo-t+anti_shift,epsilon)
        # timelike/primary term, with holomorphic (1-z)^-1
        bhp=_binomial_minus_power(a+1,total_order)
        hol=_conv_trunc(pe,bhp,total_order); anti=_conv_trunc(pe,ba,total_order)
        out[j]+=tsign*kernel.even_structure*(w2*w3)*_integrate_series_disk(hol,anti,ee-t,ee-t+anti_shift,epsilon)
        hol=_conv_trunc(po,bhp,total_order); anti=_conv_trunc(po,ba,total_order)
        out[j]+=tsign*kernel.odd_structure*(w2*w3)*_integrate_series_disk(hol,anti,eo-t,eo-t+anti_shift,epsilon)
    return out


def analytic_s_disk_integral(data,energies,epsilon,block_order,total_order=None):
    if total_order is None: total_order=block_order+8
    total=np.zeros(3,dtype=complex)
    for k in data:
        total += k.weight*_analytic_s_disk_one_kernel(k,energies,epsilon,block_order,total_order)
    return total


def full_analytic_s_disk(original_data,swapped_data,energies,epsilon,block_order,total_order=None):
    o=analytic_s_disk_integral(original_data,energies,epsilon,block_order,total_order)
    se=[energies[0],energies[2],energies[1],energies[3]]
    s=analytic_s_disk_integral(swapped_data,se,epsilon,block_order,total_order)
    return np.array([o[0]+s[1],o[1]+s[0],o[2]+s[2]])


def one_ordering_regularized(energies:Sequence[complex],*,q_order=5,p_nodes=60,p_max=5.0,
                       epsilon0=.08,epsilon1=.06,theta_orders=(28,28,72),radial_order=28,
                       disk_total_order=14,
                       lens_radial_order=24,lens_angular_order=64,lens_power=3.0,
                       p_cut=.03,reference_p_max=.18,cancellation_limit=1.0e7,
                       diagnostics=None,structure_cache=None,block_cache=None,
                       elliptic_cache=None,direct_series_cache=None,
                       momentum_scheme="threshold_weighted",momentum_threshold_options=None,
                       series_parameter="elliptic_nome"):
    """Crossing-patched amplitude with the z=0/infinity disks integrated as OPE series.

    The disk formula is meromorphic in energies and therefore implements the local
    analytic-continuation/counterterm prescription for all terms represented in the
    truncated OPE.  The z=1 lens is integrated directly; the Figure-3-like slices
    used in the fitting notebook are convergent there.

    By default q_order=N truncates H_even at qhat**N and H_odd at
    qhat**(N+1/2), after c recursion and algebraic conversion. Local disk
    series back-expand that same approximation to disk_total_order.
    Explicit series_parameter='sewing' retains the unresummed diagnostic.
    """
    swapped=[energies[0],energies[2],energies[1],energies[3]]
    backend_options=dict(
        series_parameter=series_parameter,
        momentum_scheme=momentum_scheme,momentum_threshold_options=momentum_threshold_options,
        p_cut=p_cut,reference_p_max=reference_p_max,
        cancellation_limit=cancellation_limit,diagnostics=diagnostics,
        structure_cache=structure_cache,block_cache=block_cache,
    )
    s=build_s_channel_data(
        energies,q_order,p_nodes,p_max,
        elliptic_cache=elliptic_cache,**backend_options,
    )
    ss=build_s_channel_data(
        swapped,q_order,p_nodes,p_max,
        elliptic_cache=elliptic_cache,**backend_options,
    )
    t=build_t_channel_data(
        energies,q_order,p_nodes,p_max,
        direct_series_cache=direct_series_cache,**backend_options,
    )
    ts=build_t_channel_data(
        swapped,q_order,p_nodes,p_max,
        direct_series_cache=direct_series_cache,**backend_options,
    )
    ann=full_plane_annulus(s,ss,energies,epsilon0,epsilon1,theta_orders,radial_order,t,ts,q_order)
    disk=full_analytic_s_disk(s,ss,energies,epsilon0,q_order,disk_total_order)
    lens=full_lens_integral(t,ts,energies,epsilon1,q_order,lens_radial_order,lens_angular_order,lens_power)
    return ann+disk+lens


def equal_outgoing_coefficient_regularized(omega0:complex,**kwargs)->complex:
    E=[omega0/3]*3+[omega0]
    return one_ordering_regularized(E,**kwargs)[0]

def equal_outgoing_coefficient_regularized_fast(omega0:complex,*,q_order=5,p_nodes=40,p_max=4.0,
                       epsilon0=.08,epsilon1=.06,theta_orders=(24,24,60),radial_order=24,
                       disk_total_order=14,lens_radial_order=20,lens_angular_order=56,lens_power=3.0,
                       p_cut=.03,reference_p_max=.18,cancellation_limit=1.0e7,
                       diagnostics=None,momentum_scheme="threshold_weighted",momentum_threshold_options=None,
                       series_parameter="elliptic_nome"):
    """Specialized equal-outgoing evaluation; avoids rebuilding identical permutations."""
    E=[omega0/3]*3+[omega0]
    structure_cache={}
    block_cache={}
    elliptic_cache={}
    direct_series_cache={}
    s=build_s_channel_data(
        E,q_order,p_nodes,p_max,
        series_parameter=series_parameter,
        momentum_scheme=momentum_scheme,momentum_threshold_options=momentum_threshold_options,
        p_cut=p_cut,reference_p_max=reference_p_max,
        cancellation_limit=cancellation_limit,diagnostics=diagnostics,
        structure_cache=structure_cache,block_cache=block_cache,
        elliptic_cache=elliptic_cache,
    )
    t=build_t_channel_data(
        E,q_order,p_nodes,p_max,
        series_parameter=series_parameter,
        momentum_scheme=momentum_scheme,momentum_threshold_options=momentum_threshold_options,
        p_cut=p_cut,reference_p_max=reference_p_max,
        cancellation_limit=cancellation_limit,diagnostics=diagnostics,
        structure_cache=structure_cache,block_cache=block_cache,
        direct_series_cache=direct_series_cache,
    )
    ann=full_plane_annulus(s,s,E,epsilon0,epsilon1,theta_orders,radial_order,t,t,q_order)
    disk=full_analytic_s_disk(s,s,E,epsilon0,q_order,disk_total_order)
    lens=full_lens_integral(t,t,E,epsilon1,q_order,lens_radial_order,lens_angular_order,lens_power)
    return (ann+disk+lens)[0]


def vector_amplitude_coefficients_regularized(energies:Sequence[complex],**kwargs):
    """Regularized tensor coefficients (A,B,C) for energy ordering (w1,w2,w3,w0)."""
    if abs(complex(energies[3]-sum(energies[:3])))>1e-10:
        raise ValueError('energy conservation required')
    kwargs=dict(kwargs)
    kwargs.setdefault("structure_cache",{})
    kwargs.setdefault("block_cache",{})
    kwargs.setdefault("elliptic_cache",{})
    kwargs.setdefault("direct_series_cache",{})
    o=one_ordering_regularized(energies,**kwargs)
    cyc=[energies[1],energies[2],energies[0],energies[3]]
    c=one_ordering_regularized(cyc,**kwargs)
    return np.array([o[0],o[1],c[0]])


def universal_pair_ansatz(energies:Sequence[complex])->np.ndarray:
    """Candidate inferred from the equal-energy scan and exact resonance.

    Returns coefficients (A,B,C) in the tensor basis
      A delta(a0,a3)delta(a1,a2) + B delta(a0,a2)delta(a1,a3)
        + C delta(a0,a1)delta(a2,a3).
    """
    w1,w2,w3,w0=map(complex,energies)
    F=-w0*w0/(3+2j*w0)
    return math.pi*F*np.array([w1*w2,w1*w3,w2*w3],dtype=complex)


def pair_channel_ansatz(energies:Sequence[complex])->np.ndarray:
    """Simple channel-dependent rational candidate inferred from numerical data.

    For ordering (w1,w2,w3,w0), the coefficients multiply
      (delta03 delta12, delta02 delta13, delta01 delta23).
    """
    w1,w2,w3,w0=map(complex,energies)
    common=-math.pi*w0*w1*w2*w3
    return common*np.array([
        1/(1+1j*(w1+w2)),
        1/(1+1j*(w1+w3)),
        1/(1+1j*(w2+w3)),
    ],dtype=complex)
