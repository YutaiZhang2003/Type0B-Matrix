"""Independent oscillator-matrix and spectral tests; no string amplitudes enter.

The exact Laguerre overlaps derive the Jacobi representation used in the
variational argument. Finite truncations are a numerical convergence check,
not the proof of the inequalities or of any matrix/string dictionary.
"""
from pathlib import Path
import json
import sympy as s
import numpy as np
from scipy.linalg import eigh_tridiagonal

checks = []


def check(name, condition):
    ok = bool(condition)
    checks.append({"name": name, "passed": ok})
    if not ok:
        raise AssertionError(name)


# Direct normalized radial integrals, independent of the claimed ladder rule.
x = s.symbols("x", positive=True)
for L in [1, 2, 12]:
    a = s.Rational(2 * L + 1, 2)
    for n in range(4):
        for m in range(4):
            poly = s.Poly(s.assoc_laguerre(n, a-1, x)
                          * s.assoc_laguerre(m, a, x), x)
            integral = sum(coef * s.gamma(a+power[0]+1)
                           for power, coef in poly.terms())
            value = s.simplify(integral * s.sqrt(
                s.factorial(n)*s.factorial(m)
                / (s.gamma(n+a)*s.gamma(m+a+1))))
            expected = ((s.sqrt(n+a) if m == n else 0)
                        - (s.sqrt(n) if m == n-1 else 0))
            check(f"L={L},n={n},m={m}: normalized r overlap",
                  s.simplify(value-expected) == 0)

# Fourth-order ground energy from the three-state secular determinant.
# A path returning to state zero in four steps reaches at most state two,
# so this truncation is exact for that perturbative coefficient.
a, g, t = s.symbols("a g t", positive=True)
e2, e4 = s.symbols("e2 e4", real=True)
e = e2*t + e4*t*t
mat = s.Matrix([[-e, -g*s.sqrt(a), 0],
                [-g*s.sqrt(a), 1-e, -g],
                [0, -g, 2-e]])
det = s.expand(mat.det()).subs(g*g, t)
eq2 = s.expand(det).coeff(t, 1)
sol2 = s.solve(eq2, e2)[0]
sol4 = s.solve(s.expand(det.subs(e2, sol2)).coeff(t, 2), e4)[0]
check("exact second-order coefficient", s.simplify(sol2+a) == 0)
check("exact fourth-order coefficient", s.simplify(sol4-a*(a-s.Rational(1,2))) == 0)

# The finite matrices represent the full radial oscillator basis successively.
# Phases have been chosen so all nearest-neighbor matrix elements are negative.
def ground(L, coupling, size):
    aa = L + .5
    diagonal = aa + np.arange(size, dtype=float)
    edge_index = np.arange(size-1)
    squared = np.where(edge_index % 2 == 0, edge_index//2 + aa,
                       (edge_index+1)//2)
    offdiag = -abs(coupling)*np.sqrt(squared)
    return eigh_tridiagonal(diagonal, offdiag, select='i',
                            select_range=(0, 0), eigvals_only=True,
                            tol=1e-13)[0]


records = []
for lam in [.03, .1, .3, .7, 1., 2., 4.]:
    coupling = np.sqrt(2)*lam
    e1a, e12a = ground(1,coupling,96), ground(12,coupling,96)
    e1b, e12b = ground(1,coupling,160), ground(12,coupling,160)
    check(f"lambda={lam}: oscillator-basis convergence",
          max(abs(e1a-e1b),abs(e12a-e12b)) < 2e-9)
    check(f"lambda={lam}: strict radial and Jacobi bounds",
          0 < e12b-e1b < 11)
    rec = {"lambda_over_Omega_3halves":lam,
           "singlet_ground_over_Omega":e12b+1.5,
           "vector_ground_over_Omega":e1b+12.5,
           "Cartan_flavor_ground_over_Omega":e1b+1.5,
           "S_minus_V_over_Omega":e12b-e1b-11,
           "truncation_difference":max(abs(e1a-e1b),abs(e12a-e12b))}
    records.append(rec)

# Compare the independently integrated oscillator basis to the perturbation.
for L in [1,12]:
    errors = []
    for gg in [.08,.04,.02]:
        aa = L+.5
        exact = ground(L,gg,64)
        approx = aa-aa*gg*gg+aa*L*gg**4
        errors.append(abs(exact-approx))
    check(f"L={L}: fourth-order expansion has sixth-order remainder",
          all(40 < errors[i]/errors[i+1] < 85 for i in range(2)))

# Spectral difference in the physical rank-four sectors (g=sqrt(2) lambda).
L = s.symbols("L", integer=True, positive=True)
rank4_e2 = -2*(L+s.Rational(1,2))
rank4_e4 = 4*L*(L+s.Rational(1,2))
check("rank4 S/V second-order splitting is -22 lambda squared",
      rank4_e2.subs(L,12)-rank4_e2.subs(L,1) == -22)
check("rank4 S/V fourth-order splitting is 594 lambda fourth",
      rank4_e4.subs(L,12)-rank4_e4.subs(L,1) == 594)

result = {"passed":True, "number_of_checks":len(checks), "checks":checks,
          "finite_truncation_spectra":records,
          "scope":"Exact radial oscillator overlaps and perturbative coefficients; "
                  "converging stable-trap spectra. No inverted sea or string amplitude."}
(Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_majorana_rank4_spectral_results.json').write_text(
    json.dumps(result,indent=2)+'\n')
print(json.dumps({"passed":True,"checks":len(checks),"spectra":records},indent=2))
