"""Independent two-color matrix reduction and KZ collision-domain checks.

Uses polynomial Schwinger bosons and exact symbolic radial operators.
It does not import the existing KZ-check implementations.
"""
import json
from collections import defaultdict
from pathlib import Path

import sympy as s


def E(poly, target, source, n):
    out = defaultdict(int)
    for occ, coef in poly.items():
        for a in range(n):
            i, j = target*n+a, source*n+a
            if occ[j]:
                nxt = list(occ)
                nxt[j] -= 1
                nxt[i] += 1
                out[tuple(nxt)] += coef*occ[j]
    return {k: v for k, v in out.items() if v}


def one_per_color(a, b, n):
    occ = [0]*(2*n)
    occ[a] += 1
    occ[n+b] += 1
    return tuple(occ)


def check_spin(n):
    for a in range(n):
        for b in range(n):
            state = {one_per_color(a, b, n): 1}
            actual = E(E(state, 1, 0, n), 0, 1, n)
            expected = defaultdict(int)
            expected[one_per_color(a, b, n)] += 1
            expected[one_per_color(b, a, n)] += 1
            assert actual == dict(expected)
    tr = {one_per_color(a, a, n): 1 for a in range(n)}
    assert E(E(tr, 1, 0, n), 0, 1, n) == {k: 2*v for k,v in tr.items()}
    return {"n": n, "basis_states_checked": n*n,
            "E12_E21": "I+P", "trace_eigenvalue": 2}


def main():
    r = s.symbols('r', positive=True)
    f = s.Function('f')(r)
    u = s.Function('u')(r)
    psi = u/r
    ordinary = -s.diff(psi, r, 2)-2*s.diff(psi, r)/r+2*psi/r**2
    flattened = s.simplify(r*ordinary)
    assert s.simplify(flattened+s.diff(u,r,2)-2*u/r**2) == 0
    weighted_kz = -s.diff(psi,r,2)-2*s.diff(psi,r)/r
    assert s.simplify(r*weighted_kz+s.diff(u,r,2)) == 0
    D = lambda f: s.diff(f,r)+f/r
    Ddag = lambda f: -s.diff(f,r)-f/r
    assert s.simplify(Ddag(D(f))+s.diff(f,r,2)+2*s.diff(f,r)/r) == 0
    assert D(1/r) == 0
    assert s.simplify(r*D(psi)-s.diff(u,r)) == 0
    p = s.symbols('p')
    ordinary_roots = s.solve(p*(p-1)-2,p)
    assert ordinary_roots == [-1,2]
    # Exact angular metric from M=R I+(r/2)n.sigma.
    # Tr(sigma_a sigma_b)=2 delta_ab and n.dn=0 give these entries.
    R,theta,phi = s.symbols('R theta phi', real=True)
    eye = s.eye(2)
    pauli = [s.Matrix([[0,1],[1,0]]),s.Matrix([[0,-s.I],[s.I,0]]),
             s.Matrix([[1,0],[0,-1]])]
    nv = [s.sin(theta)*s.cos(phi),s.sin(theta)*s.sin(phi),s.cos(theta)]
    mat = R*eye+r*sum((nv[a]*pauli[a] for a in range(3)),s.zeros(2))/2
    coords = [R,r,theta,phi]
    metric = s.Matrix([[s.trigsimp(s.trace(s.diff(mat,x)*s.diff(mat,y)))
                        for y in coords] for x in coords])
    assert metric == s.diag(2,s.Rational(1,2),r*r/2,r*r*s.sin(theta)**2/2)
    eps = s.symbols('epsilon',positive=True)
    raw_divergence = s.integrate(r*r*s.diff(1/r,r)**2+2/r**2,(r,eps,1))
    assert s.simplify(raw_divergence-3*(1/eps-1)) == 0
    results = {
        "spin_checks": [check_spin(n) for n in [3,4,23,24]],
        "matrix_metric": str(metric),
        "ordinary_flattened_trace_operator": str(flattened),
        "ordinary_indicial_roots": [str(x) for x in ordinary_roots],
        "weighted_KZ_flattened_trace_operator": "-u''",
        "raw_matrix_gradient_norm_on_epsilon_to_1_for_psi_1_over_r": str(raw_divergence),
        "scope": "Exact differential/form-domain diagnostic; no all-N matrix-domain construction or string scattering claim"
    }
    (Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_kz_matrix_collision_results.json').write_text(
        json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__ == '__main__':
    main()
