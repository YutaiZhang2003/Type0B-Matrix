#!/usr/bin/env python3
"""Exact checks of SO(d-1)-invariant symmetric-traceless spin tensors.

These are finite-dimensional spin identities. They do not construct a
matrix reduction, an inverted-oscillator sea, or heterotic amplitudes.
"""

from collections import defaultdict
from itertools import permutations
import json
from pathlib import Path

import sympy as s


RESULTS = []


def equal(label, expression, expected=0):
    difference = s.factor(expression - expected)
    assert difference == 0, (label, difference)
    RESULTS.append({"check": label, "passed": True})


def zonal_polynomial(degree, dimension, t, radius_squared):
    """Axis-normalized harmonic t^degree + lower-axis-degree terms."""
    coefficient = s.Integer(1)
    polynomial = s.Integer(0)
    for k in range(degree // 2 + 1):
        polynomial += coefficient * t ** (degree - 2*k) * radius_squared**k
        coefficient *= -s.Rational(
            (degree - 2*k)*(degree - 2*k - 1),
            2*(k + 1)*(2*k + dimension - 1),
        )
    return s.expand(polynomial)


def radial_checks():
    t, r2, r, u = s.symbols("t r2 r u")
    for dimension in (3, 4, 24):
        alpha = s.Rational(dimension - 2, 2)
        for degree in range(13):
            h = zonal_polynomial(degree, dimension, t, r2)
            laplace = (s.diff(h,t,2) + 4*r2*s.diff(h,r2,2)
                       + 2*(dimension-1)*s.diff(h,r2))
            equal(f"d={dimension}, N={degree}: harmonic", laplace)
            equal(f"d={dimension}, N={degree}: nonzero axis normalization",
                  h.subs(r2,0), t**degree)
            equal(f"d={dimension}, N={degree}: homogeneous degree",
                  t*s.diff(h,t)+2*r2*s.diff(h,r2),degree*h)
            gegenbauer = s.gegenbauer(degree,alpha,u)
            equal(f"d={dimension}, N={degree}: Gegenbauer expression",
                  h.subs({t:r*u,r2:r*r*(1-u*u)}),
                  r**degree*gegenbauer/gegenbauer.subs(u,1))


def tensor_checks(degree, dimension):
    coordinates = s.symbols(f"z0:{dimension}")
    h = zonal_polynomial(degree,dimension,coordinates[0],
                         sum(z*z for z in coordinates[1:]))
    tensor = {}
    for powers, coefficient in s.Poly(h,*coordinates).terms():
        indices = tuple(i for i,power in enumerate(powers) for _ in range(power))
        component = coefficient*s.prod(s.factorial(p) for p in powers)/s.factorial(degree)
        for permuted in set(permutations(indices)):
            tensor[permuted] = component
    for i in range(degree):
        for j in range(i+1,degree):
            for indices, value in tensor.items():
                exchanged = list(indices)
                exchanged[i],exchanged[j] = exchanged[j],exchanged[i]
                assert tensor.get(tuple(exchanged),0) == value
            traces = defaultdict(lambda:s.Integer(0))
            for indices,value in tensor.items():
                if indices[i] == indices[j]:
                    remaining = tuple(index for place,index in enumerate(indices)
                                      if place not in (i,j))
                    traces[remaining] += value
            assert all(s.factor(value) == 0 for value in traces.values())
            RESULTS.append({"check":f"d={dimension}, N={degree}: P{i}{j}=1 and Q{i}{j}=0",
                            "passed":True})
    for a,b in sorted({(1,2),(1,dimension-1),(dimension-2,dimension-1)}):
        equal(f"d={dimension}, N={degree}: SO(d-1) rotation {a},{b}",
              coordinates[a]*s.diff(h,coordinates[b])
              -coordinates[b]*s.diff(h,coordinates[a]))


if __name__ == "__main__":
    radial_checks()
    for d,N in ((4,2),(4,3),(4,4),(24,2),(24,3),(24,4)):
        tensor_checks(N,d)
    report={"scope":"Exact zonal spin identities; no matrix origin or continuum sea is established",
            "count":len(RESULTS),"checks":RESULTS}
    destination=(Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_zonal_state_results.json')
    destination.write_text(json.dumps(report,indent=2)+"\n")
    print(f"{len(RESULTS)} exact zonal-state checks passed; {destination.name}")
