"""Exact checks of all-rank singlet covariants and low-rank generalized exponents.

No amplitude data enter these checks. The Pfaffian covariance tests use exact
integer/rational matrices. The independent D2/D3 computation is the full Weyl
alternating q-partition formula, rather than a check of the claimed exponents.
"""
from __future__ import annotations

from functools import lru_cache
from itertools import combinations, permutations, product
from math import factorial
import json
from pathlib import Path
import random

import sympy as s

checks = []


def check(name, condition):
    if not bool(condition):
        raise AssertionError(name)
    checks.append(name)


def pf(M):
    n = M.rows
    if n == 0:
        return s.Integer(1)
    if n % 2:
        return s.Integer(0)
    return sum((-1) ** (j + 1) * M[0, j]
               * pf(M.extract([i for i in range(1, n) if i != j],
                              [i for i in range(1, n) if i != j]))
               for j in range(1, n))


def cof(C):
    r = C.rows
    return s.Matrix([(-1) ** i * pf(C.extract([j for j in range(r) if j != i],
                                             [j for j in range(r) if j != i]))
                     for i in range(r)])


def scalar_p(C, A):
    z = cof(C)
    return ((A*z).T * C * A*A*z)[0]


def form_component(C, A, indices):
    """i_(Az) C^((t+1)/2)/(normalization), using ordered Pfaffians."""
    z = A * cof(C)
    value = 0
    for j in range(C.rows):
        if j in indices:
            continue
        order = (j,) + tuple(indices)
        value += z[j] * pf(C.extract(order, order))
    return s.expand(value)


def random_matrix(r, rng):
    return s.Matrix(r, r, [rng.randint(-2, 2) for _ in range(r*r)])


rng = random.Random(912036)
for r in (3, 5, 7):
    for trial in range(3):
        C0 = random_matrix(r, rng)
        C = C0-C0.T
        A = random_matrix(r, rng)
        u0 = random_matrix(r, rng)
        u = u0-u0.T
        z = cof(C)
        check(f"r{r}_{trial}_cofactor_kernel", C*z == s.zeros(r, 1))
        check(f"r{r}_{trial}_p_unipotent", scalar_p(C, A+u*C) == scalar_p(C, A))
        for t in range(1, r-1, 2):
            ind = tuple(range(t))
            check(f"r{r}_{trial}_v{t}_unipotent",
                  form_component(C, A+u*C, ind) == form_component(C, A, ind))

        # Full GL covariance, using a non-diagonal unimodular matrix.
        g = s.eye(r)
        g[0, 1] = 2
        g[r-1, 0] = 1
        gi = g.inv()
        Cg, Ag = gi.T*C*gi, g*A*gi
        check(f"r{r}_{trial}_cofactor_GL", cof(Cg) == g*z/g.det())
        check(f"r{r}_{trial}_p_GL", scalar_p(Cg, Ag) == scalar_p(C, A)/g.det()**2)
        # Torus covariance determines every highest-weight component.
        d = s.diag(*range(2, r+2))
        di = d.inv()
        Cd, Ad = di.T*C*di, d*A*di
        check(f"r{r}_{trial}_p_torus", scalar_p(Cd, Ad) == scalar_p(C, A)/d.det()**2)
        for t in range(1, r-1, 2):
            ind = tuple(range(t))
            factor = d.det() * s.prod(d[i, i] for i in ind)
            check(f"r{r}_{trial}_v{t}_torus",
                  form_component(Cd, Ad, ind)*factor == form_component(C, A, ind))

    # A single explicit nonzero configuration works for every required t.
    C = s.zeros(r)
    for i in range(0, r-1, 2):
        C[i, i+1], C[i+1, i] = 1, -1
    A = s.zeros(r)
    A[0, r-1], A[1, 0] = 1, 1
    check(f"r{r}_explicit_p", scalar_p(C, A) == 1)
    for t in range(1, r-1, 2):
        check(f"r{r}_explicit_v{t}", form_component(C, A, tuple(range(1, t+1))) != 0)

# Infinitesimal/full upper-Levi tests for the actual highest component.
for r in (3, 5):
    C0 = random_matrix(r, rng)
    C = C0-C0.T
    A = random_matrix(r, rng)
    for i in range(r-1):
        g = s.eye(r)
        g[i, i+1] = 3
        gi = g.inv()
        for t in range(1, r-1, 2):
            check(f"r{r}_v{t}_Levi_simple{i}",
                  form_component(gi.T*C*gi, g*A*gi, tuple(range(t)))
                  == form_component(C, A, tuple(range(t))))

for r in (2, 4, 6, 8):
    C0 = random_matrix(r, rng)
    C = C0-C0.T
    for ell in range(0, r+1, 2):
        t = r-ell
        p = pf(C)**11 * pf(C[:t, :t])
        degree = 6*r-ell//2
        weights = (12,)*t + (11,)*ell
        check(f"r{r}_ell{ell}_degree_bound_saturated", 2*degree == sum(weights))
        if p:
            d = s.diag(*range(2, r+2))
            di = d.inv()
            Cd = di.T*C*di
            pd = pf(Cd)**11 * pf(Cd[:t, :t])
            check(f"r{r}_ell{ell}_Pf_weight",
                  pd*s.prod(d[i,i]**weights[i] for i in range(r)) == p)

for r in range(2, 41):
    vals = []
    for ell in range(r+1):
        kap = (12,)*(r-ell) + (11,)*ell
        root_lattice = sum(kap) % 2 == 0
        check(f"r{r}_ell{ell}_root_lattice", root_lattice == (ell % 2 == 0))
        cas = sum(kap[i]*(kap[i]+2*(r-i-1)) for i in range(r))
        expected = 11*r*(r+10)+(r-ell)*(r+ell+22)
        check(f"r{r}_ell{ell}_Casimir", cas == expected)
        if root_lattice:
            vals.append(cas)
    expected_min = 11*r*(r+10) + (2*r+21 if r % 2 else 0)
    check(f"r{r}_Casimir_min", min(vals) == expected_min)
    if r % 2:
        for ell in range(2, r+1, 2):
            t = r-ell
            degree = 5*(r+3)+(r+t+2)//2
            check(f"r{r}_ell{ell}_odd_upper_degree", degree == 6*r-ell//2+16)
        check(f"r{r}_odd_gap16", (11*r+33)//2-(11*r+1)//2 == 16)


def simple_coords(v):
    r = len(v)
    a = list(v[:r-2])
    out = [sum(a[:i+1]) for i in range(r-2)]
    left = sum(v[:-1])
    if (left-v[-1]) % 2:
        return None
    return tuple(out + [(left-v[-1])//2, (left+v[-1])//2])


def perm_sign(p):
    return (-1)**sum(p[i] > p[j] for i in range(len(p)) for j in range(i+1, len(p)))


def generalized_exponents(kap):
    """Independent exact Weyl q-partition calculation, practical here for D2/D3."""
    r = len(kap)
    roots = []
    for i,j in combinations(range(r), 2):
        for sg in (-1, 1):
            v = [0]*r
            v[i], v[j] = 1, sg
            roots.append(simple_coords(v))
    roots = tuple(sorted(roots, key=lambda z: (sum(z), z)))

    @lru_cache(None)
    def part(idx, target):
        if min(target) < 0:
            return ()
        if idx < 0:
            return (1,) if not any(target) else ()
        a = roots[idx]
        nmax = min(target[i]//a[i] for i in range(r) if a[i])
        res = []
        for n in range(nmax+1):
            v = tuple(target[i]-n*a[i] for i in range(r))
            pol = part(idx-1, v)
            if len(res) < len(pol)+n:
                res += [0]*(len(pol)+n-len(res))
            for k,c in enumerate(pol):
                res[k+n] += c
        while res and res[-1] == 0:
            res.pop()
        return tuple(res)

    rho = tuple(range(r-1, -1, -1))
    shifted = tuple(kap[i]+rho[i] for i in range(r))
    result = []
    for pm in permutations(range(r)):
        for signs in product((-1,1), repeat=r):
            if s.prod(signs) != 1:
                continue
            v = tuple(signs[i]*shifted[pm[i]]-rho[i] for i in range(r))
            target = simple_coords(v)
            if target is None or min(target) < 0:
                continue
            pol = part(len(roots)-1, target)
            if len(result) < len(pol):
                result += [0]*(len(pol)-len(result))
            for i,c in enumerate(pol):
                result[i] += perm_sign(pm)*c
    return {i:c for i,c in enumerate(result) if c}


expected = {
    (12,12): {12:1},
    (11,11): {11:1},
    (12,11): {},
    (12,12,12): {36:1},
    (12,11,11): {33:1,34:1,35:1},
    (12,12,11): {},
    (11,11,11): {},
    (2,2,2): {6:1},
    (2,1,1): {3:1,4:1,5:1},
}
polynomials = {}
for kap, pred in expected.items():
    actual = generalized_exponents(kap)
    check(f"Kostant_{kap}", actual == pred)
    polynomials[str(kap)] = actual


def dominant(v):
    a = sorted((abs(x) for x in v), reverse=True)
    if all(a) and sum(x < 0 for x in v) % 2:
        a[-1] *= -1
    return tuple(a)


def descending_tuples(rank, maximum):
    if rank == 0:
        yield ()
    else:
        for first in range(maximum, -1, -1):
            for tail in descending_tuples(rank-1, first):
                yield (first,)+tail


def zero_weight_freudenthal(kap):
    """Independent multiplicity recurrence on dominant weights, no Weyl q-sum."""
    r, maximum = len(kap), kap[0]
    norm = lambda v: sum(x*x for x in v)
    rho = tuple(range(r-1,-1,-1))
    possible = []
    for mu0 in descending_tuples(r, maximum):
        signs = (1,-1) if mu0[-1] else (1,)
        for sign in signs:
            mu = mu0[:-1]+(sign*mu0[-1],)
            difference = simple_coords(tuple(kap[i]-mu[i] for i in range(r)))
            if (difference is not None and min(difference) >= 0
                    and norm(mu) <= norm(kap)):
                possible.append(mu)
    possible.sort(key=norm, reverse=True)
    multiplicities = {kap:1}
    norm_lr = norm(tuple(kap[i]+rho[i] for i in range(r)))
    for mu in possible:
        if mu == kap:
            continue
        numerator = 0
        for i,j in combinations(range(r), 2):
            for sg in (-1,1):
                for k in range(1, maximum-mu[i]+1):
                    nu = list(mu)
                    nu[i] += k
                    nu[j] += sg*k
                    if abs(nu[j]) > maximum:
                        continue
                    mult = multiplicities.get(dominant(nu), 0)
                    numerator += 2*(mu[i]+sg*mu[j]+2*k)*mult
        denominator = norm_lr-norm(tuple(mu[i]+rho[i] for i in range(r)))
        if numerator % denominator:
            raise AssertionError(("Freudenthal_nonintegral", kap, mu, numerator, denominator))
        multiplicities[mu] = numerator//denominator

    dimension = 0
    for mu,mult in multiplicities.items():
        absolute = tuple(abs(x) for x in mu)
        denominator = 1
        for x in set(absolute):
            denominator *= factorial(absolute.count(x))
        nonzero = sum(x != 0 for x in mu)
        orbit = factorial(r)*2**(nonzero if nonzero < r else r-1)//denominator
        dimension += orbit*mult
    shifted = tuple(kap[i]+rho[i] for i in range(r))
    weyl_dim = s.prod(s.Rational(shifted[i]**2-shifted[j]**2,
                                rho[i]**2-rho[j]**2)
                      for i,j in combinations(range(r),2))
    check(f"Freudenthal_dimension_{kap}", dimension == weyl_dim)
    check(f"Freudenthal_nonnegative_{kap}", all(m>=0 for m in multiplicities.values()))
    return multiplicities.get((0,)*r,0), int(dimension)


zero_weights = {}
for r in (2,3,4,5,6):
    for ell in range(0,r+1,2):
        kap = (12,)*(r-ell)+(11,)*ell
        m0, dim = zero_weight_freudenthal(kap)
        if str(kap) in polynomials:
            check(f"two_methods_zero_{kap}", m0 == sum(polynomials[str(kap)].values()))
        zero_weights[str(kap)] = {"zero_weight_multiplicity":m0,"dimension":dim}

results = {
    "status": "passed", "number_of_checks": len(checks),
    "generalized_exponents": polynomials,
    "zero_weights": zero_weights,
    "singlet_thresholds": [
        {"N":2*r, "lower":11*r//2 if r%2==0 else (11*r+1)//2,
         "upper":11*r//2 if r%2==0 else (11*r+33)//2,
         "exact": r%2==0 or r==3}
        for r in range(2,13)
    ],
    "checks": checks,
}
# D3 independently sharpens the generic odd-r lower bound to the upper value.
results["singlet_thresholds"][1].update(lower=33, upper=33)
(Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_majorana_singlet_large_rank_results.json').write_text(
    json.dumps(results, indent=2)+"\n")
print(json.dumps({"status":"passed", "checks":len(checks),
                  "generalized_exponents":polynomials,"zero_weights":zero_weights}, indent=2))
