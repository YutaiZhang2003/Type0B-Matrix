"""Genuine two-edge ordinary Virasoro five-point c-recursion (CCY).

The comb positions are (0,q1*q2,q2,1,infinity). Coefficients exclude the
two physical leading powers. The SL(2) seed retains the central trinion;
it is not a product of hypergeometric four-point seeds. Generic Verma
weights only: unresolved pole collisions raise, never use a PBW fallback.
"""
from __future__ import annotations

import cmath
from functools import lru_cache
import math

from spin23_virasoro_torus_recursion import (
    _b_square_rs_from_h, _c_rs_from_h, _fusion_polynomial,
    _minus_dc_dh_times_a_rs,
)
from virasoro_sphere_c_recursion import _rising


def global_middle(m, n, h_out, d, h_in):
    """<h_out| L1**m V_d(1) L-1**n |h_in>, with primary tensor one."""
    @lru_cache(maxsize=None)
    def value(a, b):
        if a == 0:
            return _rising(h_in + d - h_out, b)
        result = (h_out + a - 1 - h_in - b + d) * value(a - 1, b)
        if b:
            result += b * (2 * h_in + b - 1) * value(a - 1, b - 1)
        return result
    return value(m, n)


def fivepoint_global_coefficient(n1, n2, h1, h2, external_weights):
    d1, d2, d3, d4, d5 = external_weights
    denominator = (math.factorial(n1) * _rising(2 * h1, n1)
                   * math.factorial(n2) * _rising(2 * h2, n2))
    if denominator == 0:
        raise ZeroDivisionError("singular five-point global-block weight")
    return (_rising(h1 + d2 - d1, n1)
            * global_middle(n2, n1, h2, d3, h1)
            * _rising(h2 + d4 - d5, n2) / denominator)


def fivepoint_c_coefficients(*, c, internal_weights, external_weights,
                            orders, pole_tolerance=1e-12):
    """Return {(n1,n2): coefficient}, with integer Virasoro levels."""
    cutoffs = (orders, orders) if isinstance(orders, int) else tuple(orders)
    if len(cutoffs) != 2 or any(type(x) is not int or x < 0 for x in cutoffs):
        raise ValueError("orders must be two nonnegative integers")
    internal, external = tuple(map(complex, internal_weights)), tuple(map(complex, external_weights))
    if len(internal) != 2 or len(external) != 5:
        raise ValueError("two internal and five external weights required")
    if not math.isfinite(pole_tolerance) or pole_tolerance <= 0:
        raise ValueError("pole_tolerance must be finite and positive")
    if any(not (math.isfinite(v.real) and math.isfinite(v.imag))
           for v in (complex(c), *internal, *external)):
        raise ValueError("all weights and the central charge must be finite")
    d1, d2, d3, d4, d5 = external

    @lru_cache(maxsize=None)
    def coefficient(n1, n2, current_c, h1, h2):
        total = fivepoint_global_coefficient(n1, n2, h1, h2, external)
        for edge, level, h in ((0, n1, h1), (1, n2, h2)):
            for r in range(2, level + 1):
                for s in range(1, level // r + 1):
                    null = r * s
                    pole = _c_rs_from_h(r, s, h)
                    denominator = current_c - pole
                    if abs(denominator) < pole_tolerance:
                        raise ZeroDivisionError(
                            f"five-point c-pole collision: edge={edge}, (r,s)={(r,s)}, "
                            f"c={current_c}, weights={(h1,h2)}")
                    b = cmath.sqrt(_b_square_rs_from_h(r, s, h))
                    pairs = ((d1, d2), (h2, d3)) if edge == 0 else ((h1, d3), (d5, d4))
                    residue = _minus_dc_dh_times_a_rs(r, s, h)
                    for x, y in pairs:
                        residue *= _fusion_polynomial(r, s, b, x, y)
                    if residue == 0:
                        continue
                    if edge == 0:
                        sub = coefficient(n1-null, n2, pole, h1+null, h2)
                    else:
                        sub = coefficient(n1, n2-null, pole, h1, h2+null)
                    total += residue / denominator * sub
        if not (math.isfinite(total.real) and math.isfinite(total.imag)):
            raise ArithmeticError("nonfinite five-point c-recursive coefficient")
        return complex(total)

    return {(i, j): coefficient(i, j, complex(c), *internal)
            for i in range(cutoffs[0]+1) for j in range(cutoffs[1]+1)}


def regular_fivepoint_c_coefficients(*,c,internal_weights,external_weights,orders,
                                    tolerance=1e-8):
    """Resolve coincident recursion poles by a two-radius weight limit.

    Coalescing c poles at equal edge weights are not necessarily physical
    singularities. Average the FULL coefficient over h2+epsilon*exp(i theta),
    never the individual residues. Both radii must agree. A real Verma pole
    is not removed by this prescription; its nonzero Laurent coefficients
    are explicitly tested. There is no PBW or node-dropping fallback.
    """
    options=dict(c=c,external_weights=external_weights,orders=orders)
    h1,h2=tuple(internal_weights)
    try:
        return fivepoint_c_coefficients(internal_weights=(h1,h2),**options)
    except ZeroDivisionError as exc:
        if "c-pole collision" not in str(exc):
            raise
    radius=1e-3*max(1.,abs(h2))
    tables=[]
    for r,n in ((radius,16),(radius*1.3,24)):
        samples=[]
        phases=[]
        for k in range(n):
            phase=cmath.exp(2j*math.pi*(k+.5)/n)
            samples.append(fivepoint_c_coefficients(internal_weights=(h1,h2+r*phase),**options))
            phases.append(phase)
        table={key:sum(x[key] for x in samples)/n for key in samples[0]}
        # At the supported level range a physical Verma singularity would
        # show up in a low negative Laurent mode. Refuse such a finite part.
        for key,value in table.items():
            scale=max(1.,abs(value))
            for mode in range(1,4):
                negative=sum(x[key]*phase**mode for x,phase in zip(samples,phases))/n
                if abs(negative)>tolerance*scale:
                    raise ArithmeticError("nonremovable internal-weight pole in five-point block")
        tables.append(table)
    worst=max(abs(tables[0][k]-tables[1][k])/max(1.,abs(tables[0][k]),abs(tables[1][k]))
              for k in tables[0])
    if worst>tolerance:
        raise ArithmeticError(f"coincident-pole weight-limit radii disagree: {worst}")
    return tables[1]
