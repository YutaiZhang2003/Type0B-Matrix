"""Exact representation/radial tests and an unfitted two-channel reflection test."""
from pathlib import Path
from math import comb
import json
import sympy as s
import numpy as np
from scipy.integrate import solve_ivp

checks = []


def check(name, condition):
    ok = bool(condition)
    checks.append({"name": name, "passed": ok})
    if not ok:
        raise AssertionError(name)


def zero(matrix):
    return all(s.simplify(x) == 0 for x in matrix)


I2 = s.eye(2)
sx = s.Matrix([[0, 1], [1, 0]])
sy = s.Matrix([[0, -s.I], [s.I, 0]])
sz = s.diag(1, -1)


def kron(*matrices):
    out = s.ones(1, 1)
    for matrix in matrices:
        out = s.kronecker_product(out, matrix)
    return out


def b_dimension(weight):
    ell = len(weight)
    rho = [s.Rational(2 * (ell - i) - 1, 2) for i in range(ell)]
    z = [weight[i] + rho[i] for i in range(ell)]
    ans = s.Integer(1)
    for i in range(ell):
        ans *= z[i] / rho[i]
        for j in range(i + 1, ell):
            ans *= (z[i] ** 2 - z[j] ** 2) / (rho[i] ** 2 - rho[j] ** 2)
    return s.simplify(ans)


branching = []
for n in [3, 5, 7, 23]:
    ell, ds = (n - 1) // 2, 2 ** ((n - 1) // 2)
    dimension_sum = 0
    multiplicity_sum = 0
    for k in range(ell + 1):
        dim = ds * (comb(n, k) - (comb(n, k - 1) if k else 0))
        weight = [s.Rational(3 if i < k else 1, 2) for i in range(ell)]
        check(f"n={n},k={k}: Weyl dimension of primitive spinor form",
              b_dimension(weight) == dim)
        j = s.Rational(n, 2) - k
        dimension_sum += (2 * j + 1) * dim
        multiplicity_sum += dim
        if n == 23:
            branching.append({"k": k, "L": 12 - k, "spectator_j": str(j),
                              "flavor_dimension": dim,
                              "color_J": [11 - k, 12 - k]})
    check(f"n={n}: full spectator Clifford dimension", dimension_sum == 2 ** ((3 * n - 1) // 2))
    check(f"n={n}: full neutral fiber in one parity",
          2 * multiplicity_sum == comb(n + 1, (n + 1) // 2) * ds)

# Independent n=3 spectator construction in the complete 16-dimensional Cl9 module.
n = 3
g1 = [kron(*([sz] * a + [sx] + [I2] * (n - a - 1)), I2) for a in range(n)]
g2 = [kron(*([sz] * a + [sy] + [I2] * (n - a - 1)), I2) for a in range(n)]
pf = kron(sz, sz, sz)
g3 = [kron(pf, matrix) for matrix in [sx, sy, sz]]
gammas = [g1, g2, g3]
dim = 16
spins = [s.zeros(dim) for _ in range(3)]
for a in range(n):
    spins[0] += -s.I * g2[a] * g3[a] / 2
    spins[1] += -s.I * g3[a] * g1[a] / 2
    spins[2] += -s.I * g1[a] * g2[a] / 2
check("complete Cl9 color commutator", zero(spins[0] * spins[1] - spins[1] * spins[0] - s.I * spins[2]))
cs = sum((x * x for x in spins), s.zeros(dim))
check("complete Cl9 spectator color spectrum", cs.eigenvals() == {s.Rational(15, 4): 8, s.Rational(3, 4): 8})
flavor_cs = s.zeros(dim)
for a in range(n):
    for b in range(a + 1, n):
        jab = sum((s.I * gc[a] * gc[b] / 2 for gc in gammas), s.zeros(dim))
        flavor_cs += jab * jab
        check(f"complete Cl9 commuting flavor action {a},{b}", zero(jab * spins[0] - spins[0] * jab))
check("complete Cl9 color/flavor Casimir pairing", zero(cs + flavor_cs - s.Rational(9, 2) * s.eye(dim)))

total = [kron(spins[a], I2) + kron(s.eye(dim), matrix / 2)
         for a, matrix in enumerate([sx, sy, sz])]
ct = sum((x * x for x in total), s.zeros(2 * dim))
neutral = [i for i in range(2 * dim) if total[2][i, i] == 0]
check("n=3 exact fixed-parity neutral dimension", len(neutral) == 12)
ct0 = ct.extract(neutral, neutral)
cs0 = kron(cs, I2).extract(neutral, neutral)
zh0 = kron(s.eye(dim), sz).extract(neutral, neutral)
id0 = s.eye(len(neutral))
for L in [1, 2]:
    wanted = s.Rational(4 * L * L - 1, 4)
    other = s.Rational(15, 4) if L == 1 else s.Rational(3, 4)
    projector = (cs0 - other * id0) / (wanted - other)
    shift = ct0 - L * L * id0
    check(f"n=3,L={L}: exact centrifugal square", zero((shift * shift - L * L * id0) * projector))
    check(f"n=3,L={L}: exact heavy/centrifugal anticommutation",
          zero((shift * zh0 + zh0 * shift) * projector))

r, omega, lam, energy = s.symbols("r omega lam energy", positive=True)
u = s.Function("u")(r)
check("three-dimensional radial measure flattening",
      s.simplify(-r * (s.diff(u / r, r, 2) + 2 * s.diff(u / r, r) / r) / 2 + s.diff(u, r, 2) / 2) == 0)
for L in [1, 2, 12]:
    c = L * L * I2 + L * sx
    check(f"L={L}: centrifugal eigenvalues", set(c.eigenvals()) == {L * (L - 1), L * (L + 1)})
    check(f"L={L}: nonzero heavy-sector commutator", not zero(c * sz - sz * c))
    for j in [L - 1, L]:
        for m in [0, 1, 2]:
            state = r ** (j + 1) * s.exp(-r * r / 2) * s.assoc_laguerre(m, j + s.Rational(1, 2), r * r)
            hstate = -s.diff(state, r, 2) / 2 + (s.Rational(j * (j + 1), 2) / r ** 2 + r ** 2 / 2) * state
            check(f"L={L},J={j},m={m}: exact stable oscillator tower",
                  s.simplify(hstate - (2 * m + j + s.Rational(3, 2)) * state) == 0)

# The forms have a common H_0^1 domain (Hardy's inequality); these exact
# eigenvalues certify strict stable-trap form ordering for every Yukawa coupling.
for L in range(2, 13):
    difference = (L * L - 1) * I2 + (L - 1) * sx
    check(f"L={L}: strict common-domain stable-trap comparison",
          set(difference.eigenvals()) == {L * (L - 1), (L - 1) * (L + 2)})

# At infinity both channels have real momenta at every fixed energy.
sigma = s.symbols("sigma", integer=True)
for sign in [-1, 1]:
    kap = omega * r + sign * lam / omega + (energy / omega - lam ** 2 / (2 * omega ** 3)) / r
    residual = s.expand(kap ** 2 - (omega ** 2 * r ** 2 + 2 * sign * lam * r + 2 * energy))
    check(f"asymptotic momentum coefficients sign={sign}",
          s.limit(residual, r, s.oo) == 0)


def origin_basis(L, E, wavelength, omega_value, r0, terms=14):
    """Exact regular Frobenius recurrence in the color-J basis, numerically evaluated."""
    cdiag = np.array([L * (L - 1), L * (L + 1)], dtype=float)
    out, deriv = np.zeros((2, 2)), np.zeros((2, 2))
    flip = np.array([[0., 1.], [1., 0.]])
    for col, power in enumerate([L, L + 1]):
        coeffs = [np.eye(2)[:, col]]
        for m in range(1, terms + 1):
            rhs = np.zeros(2)
            if m >= 2:
                rhs -= 2 * E * coeffs[m - 2]
            if m >= 3:
                rhs -= 2 * wavelength * flip @ coeffs[m - 3]
            if m >= 4:
                rhs -= omega_value ** 2 * coeffs[m - 4]
            denominator = (power + m) * (power + m - 1) - cdiag
            current = np.zeros(2)
            for a in range(2):
                if abs(denominator[a]) < 1e-14:
                    if abs(rhs[a]) > 1e-12:
                        raise ValueError("Unexpected resonant logarithm")
                else:
                    current[a] = rhs[a] / denominator[a]
            coeffs.append(current)
        # Divide each entire solution by r0^power; harmless conditioning change.
        out[:, col] = sum(cj * r0 ** m for m, cj in enumerate(coeffs))
        deriv[:, col] = sum((power + m) * cj * r0 ** (m - 1)
                           for m, cj in enumerate(coeffs))
    return out, deriv


def reflection(L, E, wavelength, R, r0=0.04, tol=2e-11):
    U, V = origin_basis(L, E, wavelength, 1., r0)
    cdiag = np.diag([L * (L - 1), L * (L + 1)])
    flip = np.array([[0., 1.], [1., 0.]])

    def ode(radius, values):
        mat, dmat = values[:4].reshape(2, 2), values[4:].reshape(2, 2)
        potential = cdiag / radius ** 2 - (radius ** 2 + 2 * E) * np.eye(2) - 2 * wavelength * radius * flip
        return np.concatenate([dmat.ravel(), (potential @ mat).ravel()])

    sol = solve_ivp(ode, (r0, R), np.concatenate([U.ravel(), V.ravel()]),
                    method="DOP853", rtol=tol, atol=tol * 0.01)
    if not sol.success:
        raise RuntimeError(sol.message)
    # Columns (-,+) of this matrix are the sigma-x centrifugal eigenvectors.
    change = np.array([[1., 1.], [-1., 1.]]) / np.sqrt(2)
    U = change @ sol.y[:4, -1].reshape(2, 2)
    V = change @ sol.y[4:, -1].reshape(2, 2)
    signs = np.array([1., -1.])
    k2 = R ** 2 + 2 * signs * wavelength * R + 2 * E - L * L / R ** 2
    if np.any(k2 <= 0):
        raise ValueError("Matching radius lies in a forbidden region")
    kval = np.sqrt(k2)
    kprime = (2 * R + 2 * signs * wavelength + 2 * L * L / R ** 3) / (2 * kval)
    W = np.sqrt(kval)[:, None] * U
    Z = (V + (kprime / (2 * kval))[:, None] * U) / (1j * np.sqrt(kval)[:, None])
    outgoing, incoming = (W + Z) / 2, (W - Z) / 2
    smat = np.linalg.solve(incoming.T, outgoing.T).T
    return {"L": L, "E": E, "lambda": wavelength, "R": R, "r0": r0,
            "rtol": tol, "transition_probability": float(abs(smat[1, 0]) ** 2),
            "unitarity_error": float(np.linalg.norm(smat.conj().T @ smat - np.eye(2))),
            "reciprocity_error": float(np.linalg.norm(smat - smat.T)),
            "steps": len(sol.t)}


scattering = []
for L, E in [(1, 0.), (12, 0.), (12, 2.)]:
    for R, r0, tol in [(25., .04, 2e-11), (40., .04, 2e-11), (60., .02, 3e-12)]:
        result = reflection(L, E, .7, R, r0, tol)
        scattering.append(result)
        check(f"L={L},E={E},R={R}: flux unitarity", result["unitarity_error"] < 2e-7)
        check(f"L={L},E={E},R={R}: time-reversal reciprocity", result["reciprocity_error"] < 2e-7)
    last = scattering[-3:]
    check(f"L={L},E={E}: reflection convergence", abs(last[-1]["transition_probability"] - last[-2]["transition_probability"]) < 2e-4)
    check(f"L={L},E={E}: nonzero band conversion", min(z["transition_probability"] for z in last) > 1e-4)

results = {"passed": all(c["passed"] for c in checks), "number_of_checks": len(checks),
           "checks": checks, "Spin23_branching": branching, "radial_reflection": scattering,
           "scattering_scope": "Finite rank3 one-coordinate radial reflection; local flux WKB matching, not string amplitudes."}
(Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_majorana_rank3_dynamics_results.json').write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps({"passed": results["passed"], "checks": len(checks), "radial_reflection": scattering}, indent=2))
