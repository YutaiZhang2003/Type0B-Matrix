"""Bounded algebra checks for the wall-on SO(23) gauge-charge memo.

No quantum matrix scattering state or gauge-source normalization is assumed.
"""
import json
from pathlib import Path
import numpy as np
import sympy as s

checks = {}

def require(name, condition):
    assert bool(condition), name
    checks[name] = True

def gen(n, a, b):
    q = np.zeros((n, n), dtype=complex)
    if a == b:
        return q
    q[a, b] = -1j
    q[b, a] = 1j
    return q

n = 23
gens = [gen(n, a, b) for a in range(n) for b in range(a+1, n)]
require("so23_dimension_253", len(gens) == 253)
require("vector_Casimir_22", np.array_equal(sum(t @ t for t in gens),
                                          22 * np.eye(n)))
require("unit_current_generator_trace_2",
        all(np.trace(t @ t) == 2 for t in gens))
require("sugawara_c_23_over_2", s.Rational(253, 1+21) == s.Rational(23, 2))
require("vector_weight_half", s.Rational(22, 2*(1+21)) == s.Rational(1, 2))

def d(a, b):
    return int(a == b)

for abcd in [(0,1,1,2), (0,1,0,2), (0,1,2,3), (0,1,0,1),
             (7,19,19,22)]:
    a,b,c,e = abcd
    lhs = gen(n,a,b) @ gen(n,c,e) - gen(n,c,e) @ gen(n,a,b)
    rhs = 1j * (d(a,c)*gen(n,b,e) - d(a,e)*gen(n,b,c)
                - d(b,c)*gen(n,a,e) + d(b,e)*gen(n,a,c))
    require("commutator_"+"_".join(map(str, abcd)), np.array_equal(lhs, rhs))

# The three independent four-vector invariant tensors obey the Ward identity
# separately, so arbitrary energy-dependent channel coefficients preserve it.
m = 5
delta = np.eye(m)
tensors = [
    np.einsum("ab,cd->abcd", delta, delta),
    np.einsum("ac,bd->abcd", delta, delta),
    np.einsum("ad,bc->abcd", delta, delta),
]
for idx, tensor in enumerate(tensors):
    t = gen(m, 1, 3)
    variation = (np.einsum("ae,ebcd->abcd", t, tensor)
                 + np.einsum("be,aecd->abcd", t, tensor)
                 + np.einsum("ce,abed->abcd", t, tensor)
                 + np.einsum("de,abce->abcd", t, tensor))
    require(f"four_vector_Ward_tensor_{idx}", np.all(variation == 0))

omega = s.symbols("omega", positive=True)
require("source_energy_cancels_two_LSZ_factors",
        s.simplify(omega / s.sqrt(omega**2)) == 1)

# Free target-space scalar rotation currents are not the worldsheet affine
# fermion currents. The separated-point log term cannot be a constant level.
z, w, zb, wb, kappa = s.symbols("z w zb wb kappa")
G = -kappa * (s.log(z-w) + s.log(zb-wb))
rotation_two_point = 2*(G*s.diff(G,z,w) - s.diff(G,w)*s.diff(G,z))
expected = 2*kappa**2*(s.log(z-w)+s.log(zb-wb)+1)/(z-w)**2
require("rotation_current_log_two_point",
        s.simplify(rotation_two_point-expected) == 0)
require("rotation_current_not_holomorphic",
        s.simplify(s.diff(rotation_two_point, zb)) != 0)

eps, x, y = s.symbols("eps x y", positive=True)
E = x+y+eps
F3 = (x+eps)*x*eps / s.sqrt(1+(x+eps)**2)
F4v = E*x*y*eps/(1+s.I*(x+y))
F4mix = (E*x*y*eps / (s.sqrt(1+E**2)*s.sqrt(1+x**2))
         * (1+s.I*E)*(1+s.I*(2*E-x))/(1+s.I*(y+eps)))
for name, F in [('local_archives/root_cleanup/SVV',F3),("VVVV_channel",F4v),("SSVV",F4mix)]:
    require(name+"_soft_vector_linear_zero", s.limit(F,eps,0) == 0)
    coefficient = s.simplify(s.limit(F/eps,eps,0))
    require(name+"_soft_vector_coefficient_nonzero", coefficient != 0)

out = {
    "checks": checks,
    "count": len(checks),
    "scope": "Algebra, source energy power, and supplied soft limits only.",
    "rotation_current_two_point": str(expected),
}
(Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_gauge_charge_dictionary_results.json').write_text(
    json.dumps(out, indent=2)+"\n"
)
print(f"Passed {len(checks)} bounded checks.")
