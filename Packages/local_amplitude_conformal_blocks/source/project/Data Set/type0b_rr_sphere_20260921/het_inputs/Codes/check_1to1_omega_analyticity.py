"""Numerical checks for the 1->1 resummation and its complex-energy limits.

Run from the repository root with Python, NumPy, SciPy and mpmath installed.
Outputs go to Machine Note/data_exports/heterotic_1to1_resummation_20260916.
The checks compare refinements; they are not interval error certificates.
"""

import json
from pathlib import Path

import mpmath as mp
import numpy as np
from numpy.polynomial.chebyshev import chebval
from scipy.fft import dct
from scipy.linalg import eigh_tridiagonal
from scipy.special import roots_legendre


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Machine Note/data_exports/heterotic_1to1_resummation_20260916"


def pair(value):
    value = complex(value)
    return [value.real, value.imag]


def outer_rule(omega, lam, order):
    """Tricomi moment rule; the ray is selected locally inside the energy slit."""
    a = mp.j * omega
    theta = -mp.arg(1 - mp.j * lam * omega) / 2
    ray = mp.exp(mp.j * theta)
    decay = min(mp.re(ray), mp.re((1 - mp.j * lam * omega) * ray))
    cutoff = 70 / float(decay)
    nodes, weights = roots_legendre(order)
    top = cutoff ** (1 / 8)
    u = (nodes + 1) * top / 2
    radial = u ** 8
    jacobian = weights * top / 2 * 8 * u ** 7
    moments = []
    for radius, jac in zip(radial, jacobian):
        t = ray * float(radius)
        moments.append(complex(ray * float(jac) * t ** (a - 1)
                               * mp.exp(-t) * mp.hyperu(1 - a, 1, t)
                               * mp.rgamma(a) ** 2))
    return complex(lam * ray) * radial, np.asarray(moments)


def integral_grid(omega, mesh, highest):
    """Compute the first few momentum integrals on the straight endpoint path."""
    x = complex(omega) * np.linspace(0, 1, mesh + 1)
    step = complex(omega) / mesh
    kernel = 1 / (x - 1j)
    coefficients = [x, (x - 1j) * np.log1p(1j * x) - x]
    for r in range(2, highest + 1):
        previous = coefficients[-1]
        values = np.zeros(mesh + 1, dtype=complex)
        for j in range(1, mesh + 1):
            values[j] = step * (np.dot(previous[1:j], kernel[j - 1:0:-1])
                                + 0.5j * previous[j])
        coefficients.append(values)
    return x, step, kernel, coefficients


def subtracted_amplitude(omega, z, subtraction, mesh, rule):
    """Solve directly for the remainder, avoiding cancellation at tiny t."""
    xi, weights = rule
    lam = z / (1 + z / 2)
    a = 1j * complex(omega)
    _, step, kernel, coefficients = integral_grid(omega, mesh, subtraction + 1)
    remainder = np.zeros((mesh + 1, len(xi)), dtype=complex)
    denominator = 1 - 0.5j * step * xi
    source_factor = xi ** (subtraction + 1)
    for j in range(1, mesh + 1):
        memory = np.einsum('ij,i->j', remainder[1:j], kernel[j - 1:0:-1],
                           optimize=False)
        remainder[j] = (source_factor * coefficients[subtraction + 1][j]
                        + step * xi * memory) / denominator
    finite_sum = complex(omega)
    prefactor = 1 + 0j
    for r in range(1, subtraction + 1):
        prefactor *= (a + r - 1) ** 2 * lam / r
        finite_sum += prefactor * coefficients[r][-1]
    return (1 + z / 2) ** (-a) * (finite_sum + weights @ remainder[-1])


def checked_amplitude(omega, z, subtraction, order=128):
    lam = z / (1 + z / 2)
    rule = outer_rule(mp.mpc(omega), mp.mpf(lam), order)
    values = {n: subtracted_amplitude(omega, z, subtraction, n, rule)
              for n in (256, 512, 1024)}
    extrapolated = (4 * values[1024] - values[512]) / 3
    coarse = (4 * values[512] - values[256]) / 3
    return extrapolated, {
        "omega": pair(omega), "g_squared": z, "subtraction_M": subtraction,
        "quadrature_order": order,
        "mesh_values": {str(n): pair(v) for n, v in values.items()},
        "richardson": pair(extrapolated),
        "richardson_refinement_change": abs(extrapolated - coarse),
    }


def chebyshev_coefficients(values):
    coefficients = dct(values, type=1) / (len(values) - 1)
    coefficients[[0, -1]] /= 2
    return coefficients


def beta_rule(r, order):
    """Normalized Beta(1,r+1) rule, avoiding large Jacobi weight overflow."""
    k = np.arange(order, dtype=float)
    diagonal = -r ** 2 / ((2 * k + r) * (2 * k + r + 2))
    k = np.arange(1, order, dtype=float)
    off_diagonal = (2 * k * (k + r) / ((2 * k + r)
                    * np.sqrt((2 * k + r - 1) * (2 * k + r + 1))))
    nodes, vectors = eigh_tridiagonal(diagonal, off_diagonal)
    return (nodes + 1) / 2, vectors[0] ** 2


def lower_axis_sum(xs, lam, terms, grid_size=41, quadrature=24):
    """Positive simplex recursion, with two known large-r terms subtracted.

    I_r(-ix)=-i J_r(x), D_r=(r+1)! J_r/x^(r+1).
    D_r(x)=E[D_(r-1)(x(1-T))/(1+x*T)], T~Beta(1,r+1).
    Polylogarithms sum the first two asymptotic sequences exactly.
    """
    xs = np.asarray(xs)
    xmax = max(xs) * 1.001
    nodes = np.cos(np.pi * np.arange(grid_size) / (grid_size - 1))
    grid = xmax * (1 + nodes) / 2
    previous = np.ones(grid_size)
    leading = np.array([float(x * mp.exp(-x) / mp.gamma(x) ** 2) for x in xs])
    correction = 2 * xs ** 2 + xs - 1
    y = lam * xs
    total = xs.copy()
    for j, x in enumerate(xs):
        total[j] += leading[j] * float(mp.polylog(3 - 2 * x, y[j])
                                      + correction[j] * mp.polylog(4 - 2 * x, y[j]))
    prefactor = xs.copy()
    for r in range(1, terms + 1):
        t, weights = beta_rule(r, quadrature)
        argument = grid[:, None] * (1 - t)
        coefficients = chebyshev_coefficients(previous)
        sampled = (chebval(2 * argument / xmax - 1, coefficients)
                   / (1 + grid[:, None] * t))
        previous = np.einsum('ij,j->i', sampled, weights, optimize=False)
        if not np.all(np.isfinite(previous)):
            raise ArithmeticError(f"Nonfinite simplex recursion at r={r}")
        current = chebval(2 * xs / xmax - 1, chebyshev_coefficients(previous))
        prefactor *= (xs + r - 1) ** 2 / (r * (r + 1)) * y
        total += (prefactor * current
                  - leading * y ** r * r ** (2 * xs - 3) * (1 + correction / r))
    return total


def main():
    mp.mp.dps = 35
    records = {"complex_energy": [], "first_upper_threshold": [],
               "lower_threshold": []}
    # Different subtraction counts represent the same function in overlaps,
    # including Im(omega)>1, where the original M=0 formula is not integrable.
    cases = [(0.6 + 0.3j, 4., (0, 1)),
             (0.45 + 1.2j, 4., (1, 2)),
             (-0.6 + 0.3j, 4., (1,))]
    for omega, z, subtractions in cases:
        for subtraction in subtractions:
            value, row = checked_amplitude(omega, z, subtraction)
            records["complex_energy"].append(row)
            print("complex energy", json.dumps(row), flush=True)
    # An independent refinement of the outer moment integral.
    _, row = checked_amplitude(0.45 + 1.2j, 4., 1, order=192)
    records["complex_energy"].append(row)
    print("outer refinement", json.dumps(row), flush=True)

    z = 20.
    B = 1 + z / 2
    lam = z / B
    limit = 1j * (1 - z / 2)
    linear = B * (1 - 3 * lam + (1 - lam) * np.log(B))
    for epsilon in (0.02, 0.005, 0.001, 0.0002, 0.00004):
        delta = -1j * epsilon
        value, row = checked_amplitude(1j + delta, z, 1)
        row["epsilon"] = epsilon
        row["threshold_limit"] = pair(limit)
        row["logarithm_coefficient_estimate"] = pair(
            (value - limit - linear * delta) / (delta * np.log(epsilon)))
        row["predicted_logarithm_coefficient"] = z
        records["first_upper_threshold"].append(row)
        print("upper threshold", json.dumps(row), flush=True)

    for rho in (0.75, 1., 1.25, 1.5):
        lam = 1 / rho
        z = 1 / (rho - 0.5)
        B = 1 + z / 2
        epsilons = np.array([0.01, 0.002, 0.0004, 0.00008, 0.000016, 0.0000032])
        if rho < 1:
            epsilons = np.append(epsilons, 0.)
        xs = rho * (1 - epsilons)
        coarse_terms, fine_terms = (1024, 2048) if rho == 1.5 else (512, 1024)
        coarse = B ** (-xs) * lower_axis_sum(xs, lam, coarse_terms)
        fine = B ** (-xs) * lower_axis_sum(xs, lam, fine_terms, 49, 32)
        row = {"rho": rho, "g_squared": z,
               "coarse_terms": coarse_terms, "fine_terms": fine_terms,
               "epsilon": epsilons.tolist(), "i_times_amplitude": fine.tolist(),
               "max_refinement_change": float(max(abs(fine - coarse)))}
        if rho == 1:
            # A=(i/(2e))*log(epsilon)+regular; S=iA=-(1/(2e))*log(epsilon).
            row["log_coefficient_ratios"] = (
                np.diff(fine) / (-np.diff(np.log(epsilons)) / (2 * np.e))).tolist()
        else:
            coefficient = float(rho * mp.exp(-rho) * B ** (-rho)
                                * mp.gamma(2 * rho - 2) / mp.gamma(rho) ** 2)
            if rho < 1:
                ratios = ((fine[:-1] - fine[-1])
                          / (coefficient * epsilons[:-1] ** (2 - 2 * rho)))
            else:
                ratios = fine / (coefficient * epsilons ** (2 - 2 * rho))
            row["predicted_leading_coefficient_for_iA"] = coefficient
            row["leading_asymptotic_ratios"] = ratios.tolist()
            if rho == 1.5:
                pole_subtracted = fine - coefficient / epsilons
                row["subleading_log_coefficient_ratios"] = (
                    np.diff(pole_subtracted)
                    / (-2 * coefficient * np.diff(np.log(epsilons)))).tolist()
        records["lower_threshold"].append(row)
        print("lower threshold", json.dumps(row), flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "omega_checks.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
