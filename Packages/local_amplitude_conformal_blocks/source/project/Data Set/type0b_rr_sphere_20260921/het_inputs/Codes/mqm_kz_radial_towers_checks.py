#!/usr/bin/env python3
"""Exact scalar towers and low-site density data of the KZ/null ground."""
import json
from pathlib import Path

import sympy as s


def gaussian_expectation(poly, variables):
    """Integral with normalized exp(-sum x²), keeping other symbols."""
    out = s.Integer(0)
    for powers, coefficient in s.Poly(s.expand(poly), *variables).terms():
        if any(power % 2 for power in powers):
            continue
        out += coefficient * s.prod(
            s.factorial2(power - 1) / s.Integer(2)**(power // 2)
            if power else 1 for power in powers
        )
    return s.factor(out)


def gamma_expectation(poly, z, a):
    return s.factor(sum(coef * s.rf(a, powers[0])
                        for powers, coef in s.Poly(s.expand(poly), z).terms()))


def main():
    a = s.symbols("a", positive=True)
    z, y = s.symbols("z y", real=True)
    checks = {}
    laguerres = [s.assoc_laguerre(m, a - 1, z) for m in range(6)]
    for m, L in enumerate(laguerres):
        ode = s.factor(z * s.diff(L, z, 2) + (a - z) * s.diff(L, z) + m * L)
        assert ode == 0
        f = s.exp(-z / 2) * L
        Hf = -2 * z * s.diff(f, z, 2) - 2 * a * s.diff(f, z) + z * f / 2
        assert s.simplify((Hf - (2 * m + a) * f) * s.exp(z / 2)) == 0
        checks[f"radial_eigenfunction_m{m}"] = True
        for ell, Lell in enumerate(laguerres):
            norm = gamma_expectation(L * Lell, z, a)
            expected = s.rf(a, m) / s.factorial(m) if m == ell else 0
            assert s.factor(norm - expected) == 0
            checks[f"radial_norm_m{m}_ell{ell}"] = True
    hermites = [s.hermite(k, y) for k in range(6)]
    for k, H in enumerate(hermites):
        f = s.exp(-y**2 / 2) * H
        op = -s.diff(f, y, 2) / 2 + y**2 * f / 2
        assert s.simplify((op - (k + s.Rational(1, 2)) * f) * s.exp(y**2 / 2)) == 0
        checks[f"COM_eigenfunction_k{k}"] = True
        for ell, Hell in enumerate(hermites):
            norm = gaussian_expectation(H * Hell, (y,))
            expected = 2**k * s.factorial(k) if k == ell else 0
            assert s.factor(norm - expected) == 0
            checks[f"COM_norm_k{k}_ell{ell}"] = True

    n = s.symbols("n", integer=True, positive=True)
    x = s.symbols("x0:4", real=True)
    delta = s.prod(x[j] - x[i] for i in range(4) for j in range(i + 1, 4))
    phi = s.Matrix([
        s.cancel(delta / ((x[0] - x[1]) * (x[2] - x[3]))),
        s.cancel(-delta / ((x[0] - x[2]) * (x[1] - x[3]))),
        s.cancel(delta / ((x[0] - x[3]) * (x[1] - x[2]))),
    ])
    G = s.Matrix(3, 3, lambda i, j: n*n if i == j else n)
    weight = s.expand((phi.T * G * phi)[0])
    norm = gaussian_expectation(weight, x)
    assert norm == 27*n*n/2
    R2 = sum(t*t for t in x)
    X = sum(x)/2
    r2 = R2-X**2
    A = s.Integer(6)
    ar = A-s.Rational(1, 2)
    for q in range(5):
        assert s.factor(gaussian_expectation(weight*R2**q, x)/norm-s.rf(A, q)) == 0
        assert s.factor(gaussian_expectation(weight*r2**q, x)/norm-s.rf(ar, q)) == 0
        assert s.factor(gaussian_expectation(weight*X**(2*q), x)/norm-s.rf(s.Rational(1, 2), q)) == 0
        checks[f"N4_radius_moment_q{q}"] = True
    # Nontrivial joint moment tests translation-invariant angular polynomial.
    joint = gaussian_expectation(weight*X**2*r2**2, x)/norm
    assert s.factor(joint-s.Rational(1, 2)*s.rf(ar, 2)) == 0
    checks["N4_COM_relative_independence_moment"] = True
    density_poly = s.factor(4*gaussian_expectation(weight, x[1:])/norm)
    # rho(x)=exp(-x²)/sqrt(pi) times this polynomial, at Omega=1.
    assert gaussian_expectation(density_poly, (x[0],)) == 4
    density_moments = {}
    for q in range(1, 5):
        value = s.factor(gaussian_expectation(weight*sum(t**(2*q) for t in x), x)/norm)
        direct = gaussian_expectation(density_poly*x[0]**(2*q), (x[0],))
        assert s.factor(value-direct) == 0
        density_moments[f"M{2*q}"] = str(value)
    result = {
        "all_checks_passed": True,
        "check_count": len(checks),
        "checks": checks,
        "general_even_N": {
            "homogeneity": "ell=N(N-2)/2",
            "A": "N(N-1)/2",
            "relative_gamma_shape": "a=A-1/2",
            "ordinary_trap_energies": "Omega*(A+k+2m)",
            "factorized_singlet_energies": "Omega*(k+2m)",
            "normalized_tower_multiplier":
                "H_k(sqrt(Omega)*X)/sqrt(2^k*k!) * sqrt(m!/(a)_m) L_m^(a-1)(Omega*r^2)",
        },
        "N4_norm_over_pi_squared": str(norm),
        "N4_density_rho_over_exp_minus_x2_over_sqrtpi": str(density_poly),
        "N4_density_moments_Omega1": density_moments,
        "N4_joint_X2_r4": str(s.factor(joint)),
        "scope": "Exact known harmonic ground branch and reducing scalar towers; not a Fermi sea or wall/string-coupling dictionary.",
    }
    path = (Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_kz_radial_towers_results.json')
    path.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"all_checks_passed": True, "check_count": len(checks),
                      "N4_density_polynomial": str(density_poly),
                      "N4_density_moments": density_moments,
                      "output": str(path)}, indent=2))


if __name__ == "__main__":
    main()
