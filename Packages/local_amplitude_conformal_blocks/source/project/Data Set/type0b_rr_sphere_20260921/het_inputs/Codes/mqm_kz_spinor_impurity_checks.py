"""Exact KZ and level-one null checks with a fixed Spin(n) spinor impurity."""
import json
from pathlib import Path
import sympy as s

passed = []


def check(name, value):
    if isinstance(value, s.MatrixBase):
        ok = all(s.simplify(z) == 0 for z in value)
    elif isinstance(value, bool):
        ok = value
    else:
        ok = s.simplify(value) == 0
    assert ok, name
    passed.append(name)


n = s.symbols("n", integer=True, positive=True)
x, y = s.symbols("x y", positive=True)
alpha = 1 / (n - 1)
O12 = s.diag(-(n - 1), -1)
O1S = s.Matrix([[0, n - 1], [1, -(n - 2)]]) / 2
O2S = s.Matrix([[0, -(n - 1)], [-1, -(n - 2)]]) / 2
U = (x + y) / (2 * s.sqrt(x * y) * (x - y))
V = 1 / (2 * s.sqrt(x * y))
F = s.Matrix([U, V])
check("two-vector impurity KZ x", F.diff(x) - alpha * (O12 / (x-y) + O1S / x) * F)
check("two-vector impurity KZ y", F.diff(y) - alpha * (O12 / (y-x) + O2S / y) * F)
check("two-vector homogeneity", x * F.diff(x) + y * F.diff(y) + F)
check("two-vector flattened polynomial",
      (x-y) * s.sqrt(x*y) * F - s.Matrix([(x+y)/2, (x-y)/2]))

omega_S = -(n - 1) / 2
omega_R = s.Rational(1, 2)
CS = n * (n - 1) / 8
omega = s.symbols("omega")
null_poly = 2 * CS / 3 + (n - 2) * omega / 3 + (n - 4) * omega**2 / (3*(n - 1))
uR = n * (n*n - 4) / (12*(n-1))
check("spinor null eigenvalue", null_poly.subs(omega, omega_S))
check("gamma-traceless null eigenvalue", null_poly.subs(omega, omega_R) - uR)
check("one-vector KZ exponent", alpha * omega_S + s.Rational(1,2))
check("one-vector Casimir minimum", CS + (n-1) + 2*omega_S - CS)
check("gamma-traceless Casimir", CS + (n-1) + 2*omega_R - CS - n)

I2 = s.eye(2)
X = s.Matrix([[0, 1], [1, 0]])
Y = s.Matrix([[0, -s.I], [s.I, 0]])
Z = s.diag(1, -1)


def kron(factors):
    out = s.ones(1, 1)
    for factor in factors:
        out = s.kronecker_product(out, factor)
    return s.SparseMatrix(out)


for nn in (3, 5, 7):
    ell = (nn - 1) // 2
    gammas = []
    for j in range(ell):
        for local in (X, Y):
            gammas.append(kron([Z] * j + [local] + [I2] * (ell - j - 1)))
    gammas.append(kron([Z] * ell))
    dd = 2**ell
    IV, IS = s.eye(nn), s.eye(dd)
    Ls, Ts = [], []
    for a in range(nn):
        for b in range(a+1, nn):
            L = s.zeros(nn)
            L[a,b], L[b,a] = -s.I, s.I
            T = -s.I * gammas[a] * gammas[b] / 2
            Ls.append(s.SparseMatrix(s.kronecker_product(L, IS)))
            Ts.append(s.SparseMatrix(s.kronecker_product(IV, T)))
    identity = s.eye(nn*dd)
    zero = s.zeros(nn*dd)
    O = sum((L*T for L,T in zip(Ls,Ts)), zero)
    embedding = s.Matrix.vstack(*gammas)
    check(f"n{nn} gamma embedding eigenvalue", O*embedding + s.Rational(nn-1,2)*embedding)
    check(f"n{nn} two eigenvalue polynomial",
          O*O + s.Rational(nn-2,2)*O - s.Rational(nn-1,4)*identity)
    PS = (identity - 2*O)/nn
    check(f"n{nn} gamma projector idempotent", PS*PS - PS)
    check(f"n{nn} gamma projector rank", s.trace(PS) == dd)
    norm = s.zeros(nn*dd)
    for j,(L,T) in enumerate(zip(Ls,Ts)):
        ward = s.Rational(2,3)*T - L*O/(nn-1) + O*L/3
        check(f"n{nn} null annihilation generator {j}", ward*embedding)
        norm += ward.conjugate().T*ward
    expected = s.Rational(nn*(nn*nn-4),12*(nn-1))*(identity - PS)
    check(f"n{nn} full null-square matrix", norm - expected)

results = {
    "checks_passed": len(passed),
    "N2_KZ": "Both derivatives exact for symbolic n,x,y",
    "N1_null": "Full vector-spinor matrices verified at n=3,5,7",
    "n23_gamma_traceless_null_eigenvalue": str(uR.subs(n,23)),
    "n24_gamma_traceless_null_eigenvalue": str(uR.subs(n,24)),
    "boundary_flat_exponents_g0_half": {
        "spinor": "0",
        "gamma_traceless_n23": str(s.Rational(1,2)+s.Rational(1,44)),
    },
}
(Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_kz_spinor_impurity_results.json').write_text(
    json.dumps(results, indent=2)+"\n")
print(json.dumps(results, indent=2))
