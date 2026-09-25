"""Controlled one-coordinate tests of the full positive boundary quantization.

These test a constant-coefficient Fourier multiplier, not the coupled MQM
continuum limit. The limiting Gaussian convolution is an ordering choice.
"""
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import gamma


def velocity(p, mass, quartic):
    if mass == 0:
        return np.cbrt(p / (4 * quartic))
    return 2 * np.sqrt(mass / (12 * quartic)) * np.sinh(
        np.arcsinh(3 * p / (2 * mass) * np.sqrt(12 * quartic / mass)) / 3
    )


def symbol(p, mass, quartic):
    v = velocity(p, mass, quartic)
    return mass * v * v / 2 + 3 * quartic * v**4


def smoothed(p, mass, quartic, h):
    # Integrate in the unsmoothed momentum s; split at s=0, where T_0
    # is C^1 but not C^2. This is independent of spectral grid sampling.
    f = lambda s: np.exp(-(p - s) ** 2 / h) * symbol(s, mass, quartic)
    value = quad(f, -np.inf, 0, epsabs=2e-11, epsrel=2e-11)[0]
    value += quad(f, 0, np.inf, epsabs=2e-11, epsrel=2e-11)[0]
    return value / np.sqrt(np.pi * h)


def main():
    c, h = 0.7, 0.3
    coefficient = 3 / (4 ** (4 / 3) * c ** (1 / 3))
    masses = [1.0, 0.1, 0.01, 0.001, 0.00001, 0.0]
    grid = np.linspace(-4, 4, 161)
    rows = []
    previous = None
    limit = np.array([smoothed(p, 0, c, h) for p in grid])
    # Unit L2 Gaussian wave packet in momentum, up to negligible grid tails.
    weight = np.exp(-grid**2) / np.sqrt(np.pi)
    errors = []
    for mass in masses:
        values = np.array([smoothed(p, mass, c, h) for p in grid])
        if previous is not None:
            assert np.min(values - previous) > -2e-10
        assert np.max(values - limit) < 2e-10
        difference = 1 / (values + 1j) - 1 / (limit + 1j)
        error = float(np.sqrt(np.trapezoid(weight * abs(difference) ** 2, grid)))
        errors.append(error)
        rows.append({"mass": mass, "multiplier_at_zero": float(values[len(grid)//2]),
                     "resolvent_wavepacket_L2_error": error})
        previous = values
        for p in [-3.1, -0.2, 0, 0.7, 4.2]:
            v = velocity(p, mass, c)
            assert abs(mass * v + 4*c*v**3 - p) < 1e-11
            assert abs(p*v - mass*v*v/2 - c*v**4 - symbol(p,mass,c)) < 1e-11
    assert all(a > b for a, b in zip(errors[:-1], errors[1:]))
    expected_zero = coefficient * h ** (2 / 3) * gamma(7 / 6) / np.sqrt(np.pi)
    assert abs(limit[len(grid)//2] - expected_zero) < 1e-11
    # A vacuum subtraction does not remove the full momentum-dependent
    # smoothing. Check curvature and the non-polynomial crossover scale.
    curvature = (8 / 3) * expected_zero / h
    step = 0.0004
    curvature_numeric = 2*(smoothed(step, 0, c, h)-expected_zero)/step**2
    assert abs(curvature_numeric / curvature - 1) < 2e-6
    scaled = []
    for trial_h in [0.003, 0.03, 0.3, 3.0]:
        p = 0.8*np.sqrt(trial_h)
        value = smoothed(p, 0, c, trial_h) / trial_h**(2/3)
        scaled.append(float(value))
    assert max(scaled)-min(scaled) < 1e-10
    output = {
        "scope": "constant-coefficient full Legendre symbol, Gaussian ordering; no coupled limit",
        "quartic": c, "effective_hbar": h, "rows": rows,
        "massless_zero_exact": float(expected_zero),
        "massless_curvature_exact": float(curvature),
        "massless_curvature_numeric": float(curvature_numeric),
        "h_scaled_crossover_values": scaled,
        "all_checks_passed": True,
    }
    (Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_positive_boundary_quantization_results.json').write_text(
        json.dumps(output, indent=2) + "\n"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
