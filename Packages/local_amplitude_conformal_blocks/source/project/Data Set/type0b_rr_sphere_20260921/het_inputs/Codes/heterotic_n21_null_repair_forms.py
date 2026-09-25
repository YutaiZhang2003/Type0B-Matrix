"""Exterior-form check of the specified Hermitian/null-current lift.

This is a conditional geometric obstruction, NOT a string-theory no-go.
Lawrence's displayed geometric current constraints are leading in alpha';
Kutasov--Martinec's induced metric is anomaly shifted. Their identification
at the next order requires a separate derivation including current corrections.
No repository files are modified.
"""

import itertools
import json
import sympy as s

u, v, a, b, q = s.symbols('u v a b q', real=True, nonzero=True)
coords = [u, v, a, b]


def simplify(form):
    return {k: s.simplify(z) for k, z in form.items() if s.simplify(z) != 0}


def add(*forms):
    result = {}
    for form in forms:
        for k, z in form.items():
            result[k] = result.get(k, 0) + z
    return simplify(result)


def scale(form, c):
    return simplify({k: c*z for k, z in form.items()})


def canonical(indices):
    if len(set(indices)) < len(indices):
        return 0, ()
    inversions = sum(indices[i] > indices[j]
                     for i in range(len(indices))
                     for j in range(i+1, len(indices)))
    return (-1)**inversions, tuple(sorted(indices))


def wedge(first, second):
    terms = []
    for k, z in first.items():
        for ell, w in second.items():
            sign, inds = canonical(k+ell)
            if sign:
                terms.append({inds: sign*z*w})
    return add(*terms)


def d(form):
    return add(*(wedge({(i,): s.diff(z, x)}, {k: 1})
                 for k, z in form.items() for i, x in enumerate(coords)))


def interior(form, vector):
    return add(*({k[:j]+k[j+1:]: (-1)**j * z * vector[k[j]]}
                 for k, z in form.items() for j in range(len(k))))


def Jpull(form):
    # J^*du=-da, J^*dv=-db, J^*da=du, J^*db=dv.
    transform = {0: (2, -1), 1: (3, -1), 2: (0, 1), 3: (1, 1)}
    terms = []
    for k, z in form.items():
        indices = tuple(transform[i][0] for i in k)
        sign, inds = canonical(indices)
        coeff = s.prod(transform[i][1] for i in k)
        terms.append({inds: sign*coeff*z})
    return add(*terms)


C = s.Function('C')(u, v)
F = s.Function('F')(u, v)
G = s.Function('G')(u, v)
# Real metric: C(du^2+da^2)+2F(du dv+da db)+G(dv^2+db^2).
omega = {(0, 2): C, (0, 3): F, (1, 2): F, (1, 3): G}
H = Jpull(d(omega))
dH = d(H)
expected_density = s.diff(G, u, 2) + s.diff(C, v, 2) - 2*s.diff(F, u, v)
assert simplify(add(dH, {(0, 1, 2, 3): -expected_density})) == {}

# v_current=partial_a. Its norm is C, hence nullity means C=0.
omega_null = {(0, 3): F, (1, 2): F, (1, 3): G}
H_null = Jpull(d(omega_null))
dual_current = {(3,): F}
current_defect = add(d(dual_current), interior(H_null, [0, 0, 1, 0]))
expected_defect = {(1, 3): 2*s.diff(F, v)-s.diff(G, u)}
assert add(current_defect, scale(expected_defect, -1)) == {}
# dH is minus the u derivative of the remaining current defect.
assert s.simplify(d(H_null)[(0, 1, 2, 3)]
                  + s.diff(expected_defect[(1, 3)], u)) == 0

Y = q*s.log(u/v)/2
gauge_A = {(2,): s.diff(Y, u), (3,): s.diff(Y, v)}
gauge_F = d(gauge_A)
gauge_F_squared = wedge(gauge_F, gauge_F)
source = q**2/(2*u**2*v**2)
assert add(gauge_F_squared, {(0, 1, 2, 3): -source}) == {}

induced_omega = {
    (0, 2): s.diff(Y, u)**2,
    (0, 3): s.Rational(1, 2)+s.diff(Y, u)*s.diff(Y, v),
    (1, 2): s.Rational(1, 2)+s.diff(Y, u)*s.diff(Y, v),
    (1, 3): s.diff(Y, v)**2,
}
induced_dH = d(Jpull(d(induced_omega)))
assert add(induced_dH, scale(gauge_F_squared, -1)) == {}

# A scalar Hessian perturbation preserves this four-form source.
h = s.Function('h')(u, v)
hessian_omega = {(0, 2): s.diff(h, u, 2),
                 (0, 3): s.diff(h, u, v),
                 (1, 2): s.diff(h, u, v),
                 (1, 3): s.diff(h, v, 2)}
assert d(Jpull(d(hessian_omega))) == {}

print(json.dumps({
    'scope': 'Conditional specified geometric lift; anomaly-shift/current-order issues remain.',
    'all_checks_pass': True,
    'current_defect_at_C_zero': str(expected_defect[(1, 3)])+' times dv wedge db',
    'dH_density': str(expected_density),
    'gauge_F_wedge_F_density': str(source),
    'induced_dH_equals_F_wedge_F': True,
    'Hessian_completion_leaves_dH_unchanged': True,
    'conclusion': 'Null torsion-parallel current forces dH=0 in this invariant four-dimensional Hermitian geometry, inconsistent with the retained nonzero abelian four-form source unless the geometric current conditions or other source sectors change.',
}, indent=2))
