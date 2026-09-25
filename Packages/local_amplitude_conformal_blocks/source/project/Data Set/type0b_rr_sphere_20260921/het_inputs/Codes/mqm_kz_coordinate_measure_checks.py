#!/usr/bin/env python3
"""Independent Wick-Gram test of the real-coordinate current-block norm.

The flavor contraction is computed by matching loops, without invoking
the doubled-current proof or constructing exponentially many components.
"""
import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import sympy as s


@lru_cache(None)
def matchings(indices):
    if not indices:
        return ((1, ()),)
    i, tail = indices[0], indices[1:]
    out = []
    for pos, j in enumerate(tail):
        remaining = tuple(k for k in tail if k != j)
        for sign, pairs in matchings(remaining):
            out.append(((-1)**pos * sign, ((i, j),) + pairs))
    return tuple(out)


def loops(left, right, N):
    parent = list(range(N))
    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i
    for i, j in left + right:
        parent[root(i)] = root(j)
    return len({root(i) for i in range(N)})


def main():
    checks = {}
    samples = {}
    for N in (2, 4, 6, 8):
        for tag, xs in (
            ("uniform", list(range(N))),
            ("nonuniform", [i*i + i for i in range(N)]),
        ):
            entries = matchings(tuple(range(N)))
            coeffs = [sign * s.prod(s.Rational(1, xs[i]-xs[j])
                                   for i, j in pairs)
                      for sign, pairs in entries]
            gram_by_power = defaultdict(lambda: s.Integer(0))
            for a, (_, left) in enumerate(entries):
                for b, (_, right) in enumerate(entries):
                    gram_by_power[loops(left, right, N)] += coeffs[a]*coeffs[b]
            haf = sum(c*c for c in coeffs)
            pf = sum(coeffs)
            assert pf*pf == haf
            for power, value in gram_by_power.items():
                assert value == (haf if power == N//2 else 0)
            key = f"N{N}_{tag}"
            checks[key] = True
            samples[key] = {"matchings": len(entries),
                            "norm": f"n**{N//2} * ({haf})",
                            "cancelled_lower_powers": list(range(1, N//2))}

    x = s.symbols("x0:4", real=True)
    n = s.symbols("n", positive=True)
    entries = matchings((0, 1, 2, 3))
    coeff = s.Matrix([sign*s.prod(1/(x[i]-x[j]) for i,j in pairs)
                      for sign,pairs in entries])
    delta = s.prod(x[j]-x[i] for i in range(4) for j in range(i+1,4))
    poly = (delta*coeff).applyfunc(s.cancel)
    gram = s.Matrix(3,3,lambda i,j:n*n if i==j else n)
    assert s.factor((poly.T*gram*poly)[0]-n*n*sum(poly)**2) == 0
    checks["N4_symbolic_coordinates_and_flavor"] = True

    # Scalar multiplication of the block is a form compression, generally
    # not an invariant dynamical subspace. H(f Phi)-f H(Phi) includes
    # -sum_i (partial_i f)(partial_i Phi); its transverse spin part is
    # independent of the Gaussian trap and the null-penalty coefficient.
    f = sum(t**4 for t in x)
    cross = -sum((s.diff(f,t)*poly.diff(t) for t in x), s.zeros(3,1))
    sample = dict(zip(x,(0,1,3,6)))
    u, v = poly.subs(sample), cross.subs(sample)
    minor = u[0]*v[1]-u[1]*v[0]
    assert minor != 0
    checks["scalar_multiplier_space_not_invariant"] = True
    result = {"all_checks_passed": True, "checks": checks,
              "samples": samples,
              "noninvariance_witness": {"x": [0,1,3,6],
                 "f": "sum_i x_i**4", "Phi_polynomial": list(map(str,u)),
                 "cross_derivative": list(map(str,v)), "minor_01": str(minor)},
              "scope": "Real-coordinate ground-state measure; not equality of spin Hamiltonians or scattering."}
    target = (Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_kz_coordinate_measure_results.json')
    target.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"all_checks_passed": True, "check_count": len(checks),
                      "noninvariance_minor": str(minor), "output": str(target)}, indent=2))


if __name__ == "__main__":
    main()
