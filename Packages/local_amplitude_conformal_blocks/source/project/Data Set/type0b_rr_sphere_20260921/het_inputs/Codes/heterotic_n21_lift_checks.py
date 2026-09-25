"""Illustrative N=(2,1) Hermitian-lift checks; not a full sigma-model proof.

Run with a Python environment containing SymPy.  No files are modified.
The action's anomaly-shifted metric is not assumed identical to the bare metric
used in leading-order null-supercurrent constraints.  Consequently failures
below reject the specified geometric ansatz with its stated current only.
"""

import json
import sympy as s


def zero(value):
    if isinstance(value, s.MatrixBase):
        return all(s.simplify(e) == 0 for e in value)
    return s.simplify(value) == 0


T, X, q = s.symbols("T X q", real=True, nonzero=True)
xs = [T, X]
eta = s.diag(-1, 1)
r2 = X**2 - T**2
Y = q * s.atanh(T / X)
p = s.Matrix([s.diff(Y, a) for a in xs])
M = eta + p * p.T
D = 1 - q**2 / r2
P = s.sqrt(D) * M.inv()
checks = {}
checks["determinant"] = zero(-M.det() - D)
checks["linear_wave"] = zero(-s.diff(Y, T, 2) + s.diff(Y, X, 2))
checks["DBI_scalar"] = zero(sum(s.diff((P * p)[i], xs[i]) for i in range(2)))
checks["DBI_longitudinal"] = all(
    zero(sum(s.diff(P[i, j], xs[i]) for i in range(2)))
    for j in range(2)
)
checks["primitive_abelian_curvature"] = zero(
    s.trace(M.inv() * s.hessian(Y, xs))
)
checks["Hermitian_connection_trace"] = all(
    zero(
        2 * sum(M.inv()[j, k] * s.diff(M[i, k], xs[j])
                for j in range(2) for k in range(2))
        - s.diff(s.log(D), xs[i])
    )
    for i in range(2)
)
checks["Hessian_determinant"] = zero(
    s.hessian(Y, xs).det() + q**2 / r2**2
)

# The doubled real Hermitian metric, in radial coordinates.
r, tau, a, b = s.symbols("r tau a b", real=True, nonzero=True)
coords = [r, tau, a, b]
n = s.Matrix([s.cosh(tau), -s.sinh(tau)])
fiber = eta + q**2 / r**2 * n * n.T
G = s.diag(1, -(r**2 - q**2), fiber)

# Compute curvature using the metric's two-jet at tau=0. The boost isometry
# checked below guarantees that the scalar result holds for every tau.
at0 = lambda m: m.subs(tau, 0)
g = at0(G)
gi = s.simplify(g.inv())
dg = [at0(G.diff(x)) for x in coords]
ddg = [[at0(G.diff(x, y)) for y in coords] for x in coords]
dgi = [-gi * h * gi for h in dg]
Gamma = [[[
    s.simplify(sum(gi[k, l] * (dg[m][l, nn] + dg[nn][l, m]
                              - dg[l][m, nn]) for l in range(4)) / 2)
    for nn in range(4)] for m in range(4)] for k in range(4)]


def dgamma(p0, k, m, nn):
    return s.simplify(sum(
        dgi[p0][k, l] * (dg[m][l, nn] + dg[nn][l, m] - dg[l][m, nn])
        + gi[k, l] * (ddg[p0][m][l, nn] + ddg[p0][nn][l, m]
                       - ddg[p0][l][m, nn])
        for l in range(4)) / 2)


Ric = s.Matrix(4, 4, lambda m, nn: s.factor(sum(
    dgamma(k, k, m, nn) - dgamma(nn, k, m, k)
    + sum(Gamma[k][k][l] * Gamma[l][m][nn]
          - Gamma[k][nn][l] * Gamma[l][m][k] for l in range(4))
    for k in range(4))))
R = s.factor(s.trace(gi * Ric))
expected_R = 3*q**2*(4*r**2-3*q**2)/(2*r**2*(r**2-q**2)**2)
checks["four_dimensional_Ricci_scalar"] = zero(R - expected_R)
checks["Ricci_monotonic_derivative"] = zero(
    s.diff(R, r)
    + 3*q**2*(8*r**4-9*q**2*r**2+3*q**4)/(r**3*(r**2-q**2)**3)
)
# Numerator positivity: 8u^2-9u+3 has negative discriminant and positive lead.
checks["derivative_numerator_positive"] = 81 - 4*8*3 < 0


def lie_metric(v):
    return s.Matrix(4, 4, lambda i, j: s.simplify(
        sum(v[k] * s.diff(G[i, j], coords[k])
            + G[k, j] * s.diff(v[k], coords[i])
            + G[i, k] * s.diff(v[k], coords[j]) for k in range(4))))


for name, vector in {
    "spectator_translation_1": s.Matrix([0, 0, 1, 0]),
    "spectator_translation_2": s.Matrix([0, 0, 0, 1]),
    "simultaneous_boost": s.Matrix([0, 1, b, a]),
}.items():
    checks["Killing_" + name] = zero(lie_metric(vector))

# Completeness argument: R'(r)!=0 implies v^r=0; the rA Killing equations
# imply d_r v^A=0. Coefficients of r^2,r^0,r^-2 in the remaining equations
# give v=c*d_tau+(A+c*b)*d_a+(B+c*a)*d_b, with constant A,B,c.
A, B, c = s.symbols("A B c", real=True)
v = s.Matrix([0, c, A+c*b, B+c*a])
norm = s.expand((v.T * G * v)[0])
expected_norm = (
    -(r**2-q**2)*c**2 - (A+c*b)**2 + (B+c*a)**2
    + q**2/r**2 * (s.cosh(tau)*(A+c*b)
                   - s.sinh(tau)*(B+c*a))**2
)
checks["general_Killing_norm"] = zero(norm - expected_norm)
checks["old_null_plus_norm"] = zero(
    (s.Matrix([1, 1]).T * M * s.Matrix([1, 1]))[0]
    - q**2/(X+T)**2
)
checks["old_null_minus_norm"] = zero(
    (s.Matrix([1, -1]).T * M * s.Matrix([1, -1]))[0]
    - q**2/(X-T)**2
)

# Proposed metric repair: add a Hessian that cancels the same-null components.
h = q**2/4 * s.log(r2)
Fconf = 1 - q**2/(2*r2)
checks["Hessian_repair_conformal_metric"] = zero(
    M + s.hessian(h, xs) - Fconf*eta
)
# For g_4=Fconf*eta_4 and Bismut torsion H=+/- *_flat dFconf,
# dv_flat = dFconf wedge ell and i_v H = +/- dual_1+1(dFconf) wedge ell.
# Thus parallel v demands one of F_T+F_X=0 or F_T-F_X=0.
null_derivative_plus = s.factor(s.diff(Fconf, T) + s.diff(Fconf, X))
null_derivative_minus = s.factor(s.diff(Fconf, T) - s.diff(Fconf, X))
checks["Hessian_repair_fails_parallel_current_plus"] = not zero(null_derivative_plus)
checks["Hessian_repair_fails_parallel_current_minus"] = not zero(null_derivative_minus)

assert all(checks.values()), checks
print(json.dumps({
    "scope": "Specified Hermitian geometric ansatz only; not an exclusion of the N=(2,1) candidate. Anomaly-shifted and bare sigma-model metrics/current improvements are not identified here.",
    "checks": checks,
    "Ricci_scalar": str(R),
    "Ricci_derivative": str(s.factor(s.diff(R, r))),
    "old_null_plus_norm": "q**2/(X+T)**2",
    "old_null_minus_norm": "q**2/(X-T)**2",
    "Killing_classification": "v=c*d_tau+(A+c*b)*d_a+(B+c*a)*d_b; only v=0 has constant norm for q!=0 on r>abs(q)",
    "repaired_conformal_factor": str(Fconf),
    "repair_null_derivative_plus": str(null_derivative_plus),
    "repair_null_derivative_minus": str(null_derivative_minus),
}, indent=2))
