"""Exact independent tests of the action-derived Laguerre matrix regulator.

The all-rank spectral statement is proved by a determinant lemma in the
companion memo. These calculations test its matrix realization, boundary
weights, variational normalization, and finite Poisson constraints.
"""
from pathlib import Path
import json
import sympy as s

checks = []


def equal(name, lhs, rhs=0):
    residual = s.factor(s.together(lhs-rhs))
    if residual != 0:
        raise AssertionError((name, residual))
    checks.append(name)


def truth(name, value):
    if not value:
        raise AssertionError(name)
    checks.append(name)


def matrices(N):
    A = s.Matrix(N, N, lambda n, m:
                 -s.Rational(1, 2) if n == m else -1 if n < m else 0)
    b = s.ones(N, 1)
    return A, b, A.T*A


x, z = s.symbols("x z", positive=True)
laguerre = [s.exp(-x/2)*s.laguerre(n, x) for n in range(5)]
for m in range(5):
    equal(f"derivative closure m={m}",
          s.diff(laguerre[m], x),
          -laguerre[m]/2-sum(laguerre[:m]))
    equal(f"boundary value m={m}", laguerre[m].subs(x, 0), 1)
    for n in range(m+1):
        equal(f"orthonormality {n},{m}",
              s.integrate(s.expand(laguerre[n]*laguerre[m]), (x, 0, s.oo)),
              int(n == m))

for N in range(1, 9):
    A, b, B = matrices(N)
    truth(f"boundary rank-one identity N={N}", A+A.T == -b*b.T)
    equal(f"positive stiffness determinant N={N}", B.det(), s.Rational(1, 4)**N)
    determinant = (B+z*z*s.eye(N)).det(method="domain-ge")
    expected = ((z+s.Rational(1, 2))**(2*N)
                +(z-s.Rational(1, 2))**(2*N))/2
    equal(f"all-frequency characteristic polynomial N={N}", determinant, expected)
    for value in (s.Rational(1, 7), s.Rational(1, 2), s.Rational(5, 3)):
        M = B+value**2*s.eye(N)
        actual = (b.T*M.inv()*b)[0]
        ratio = ((2*value-1)/(2*value+1))**(2*N)
        closed = (1-ratio)/(value*(1+ratio))
        equal(f"boundary resolvent N={N}, z={value}", actual, closed)
        equal(f"exact resolvent error N={N}, z={value}",
              1/value-actual, 2*ratio/(value*(1+ratio)))

# Direct eigenvectors numerically test residues independently of the
# determinant-lemma derivation.
import numpy as np
max_frequency_error = 0.0
max_weight_error = 0.0
for N in (1, 2, 3, 5, 12, 32):
    A, b, B = matrices(N)
    eigenvalues, vectors = np.linalg.eigh(np.array(B, dtype=float))
    freqs = np.sqrt(eigenvalues)
    j = np.arange(N, 0, -1)
    predicted = 0.5/np.tan((2*j-1)*np.pi/(4*N))
    weights = (np.ones(N)@vectors)**2
    predicted_weights = (1+4*predicted**2)/(2*N)
    frequency_error = float(np.max(np.abs(freqs-predicted)))
    weight_error = float(np.max(np.abs(weights-predicted_weights)))
    max_frequency_error = max(max_frequency_error, frequency_error)
    max_weight_error = max(max_weight_error, weight_error)
    truth(f"independent bath frequencies N={N}", frequency_error < 2e-11)
    truth(f"independent boundary spectral weights N={N}", weight_error < 2e-11)
    equal(f"total boundary spectral weight N={N}", (b.T*b)[0], N)

# Shape variation and bulk/boundary energy transfer use independent
# incoming and outgoing profiles with the moving Neumann condition.
velocity, Tin, kappa, K = s.symbols("velocity Tin kappa K", real=True)
hprime = (1-velocity)/(1+velocity)
Tout = Tin/hprime**2
bulk_rate = (1-velocity)*Tout-(1+velocity)*Tin
equal("bulk energy rate with moving Neumann", bulk_rate, 2*velocity*Tin/hprime)
equal("boundary energy cancels bulk transfer",
      bulk_rate.subs(Tin, kappa*K*hprime)-2*velocity*kappa*K, 0)

# Direct canonical brackets at finite N, including the background trace
# term. Exponential e^(2R) is represented by a variable E with derivative 2E.
for N, species in ((1, 1), (2, 1), (2, 2), (3, 2)):
    A, b, B = matrices(N)
    Q = [s.Matrix(s.symbols(f"q{a}_0:{N}")) for a in range(species)]
    P = [s.Matrix(s.symbols(f"p{a}_0:{N}")) for a in range(species)]
    bg = s.symbols(f"bkg0:{species}")
    E, kap = s.symbols("E kap")
    D = sum((P[a].T*A*Q[a])[0]-bg[a]*(b.T*Q[a])[0]
            for a in range(species))
    HQ = sum((P[a].T*P[a])[0]+(Q[a].T*B*Q[a])[0]/4
             for a in range(species))
    bracket = 2*kap*E-sum(t*t for t in bg)
    for a in range(species):
        for n in range(N):
            bracket += (s.diff(D,Q[a][n])*s.diff(HQ,P[a][n])
                        -s.diff(D,P[a][n])*s.diff(HQ,Q[a][n]))
    expected = 2*kap*E-sum((bg[a]+(b.T*P[a])[0])**2
                          for a in range(species))
    expected += sum((b.T*A*Q[a])[0]**2/4 for a in range(species))
    equal(f"primary constraint gives boundary pressure N={N}, d={species}",
          bracket, expected)

result = {
    "passed": True, "checks": len(checks), "named_checks": checks,
    "max_numerical_frequency_error": max_frequency_error,
    "max_numerical_boundary_weight_error": max_weight_error,
    "scope": "Finite autonomous mirror regulator and bath boundary resolvent; "
             "no coupled quantum scattering limit or heterotic duality claim."
}
(Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_clock_laguerre_matrix_results.json').write_text(
    json.dumps(result, indent=2)+"\n")
print(json.dumps(result, indent=2))
