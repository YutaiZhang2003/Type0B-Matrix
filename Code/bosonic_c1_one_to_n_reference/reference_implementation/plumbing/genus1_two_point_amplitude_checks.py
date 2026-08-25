#!/usr/bin/env python3
"""Exact algebra checks for the genus-one c=1 two-point note."""

from __future__ import annotations

import sympy as sp
import mpmath as mp

try:
    from audit_genus1_two_point_sewing_normalization import build_audit
except ImportError:  # pragma: no cover - supports package-style execution
    from plumbing.audit_genus1_two_point_sewing_normalization import build_audit


def exact_phase_ratio(mu: mp.mpf, omega: mp.mpf, x: mp.mpf) -> mp.mpc:
    """Perturbative particle-hole reflection ratio on the branch K -> 1."""
    particle_energy = mu + omega - x
    hole_energy = mu - x
    log_ratio = 1j * omega * mp.log(mu) + mp.mpf("0.5") * (
        mp.loggamma(mp.mpf("0.5") + 1j * hole_energy)
        - mp.loggamma(mp.mpf("0.5") - 1j * hole_energy)
        + mp.loggamma(mp.mpf("0.5") - 1j * particle_energy)
        - mp.loggamma(mp.mpf("0.5") + 1j * particle_energy)
    )
    return mp.exp(log_ratio)


def asymptotic_phase_ratio(mu: mp.mpf, omega: mp.mpf, x: mp.mpf) -> mp.mpc:
    return (
        1
        - 1j * omega * (omega - 2 * x) / (2 * mu)
        - omega
        * (omega - 1j)
        * (1 - 1j * omega + 3 * (omega - 2 * x) ** 2)
        / (24 * mu**2)
    )


def main() -> None:
    mp.mp.dps = 50
    omega, x, mu, g_s, g_s_bry = sp.symbols(
        "omega x mu g_s g_s_bry", nonzero=True
    )
    imaginary_unit = sp.I

    # Freeze the worldsheet prefactor from target-free sewing before any
    # matrix-model or BRY comparison below.
    sewing_audit = build_audit(alpha_prime=sp.Rational(17, 10), g_s=1.0)
    assert sewing_audit["passed"]
    assert not any(sewing_audit["external_target_controls"].values())
    sewing_geometric_coefficient = sp.nsimplify(
        sewing_audit["result"]["geometric_coefficient"],
        [sp.pi],
        full=True,
        tolerance=1.0e-12,
    )
    assert sewing_geometric_coefficient == 8 * sp.pi**2

    phase_order_one = -imaginary_unit * omega * (omega - 2 * x) / 2
    assert sp.integrate(phase_order_one, (x, 0, omega)) == 0

    phase_order_two = (
        -omega
        * (omega - imaginary_unit)
        * (1 - imaginary_unit * omega + 3 * (omega - 2 * x) ** 2)
        / 24
    )
    integrated = sp.expand(sp.integrate(phase_order_two, (x, 0, omega)))
    expected = (
        imaginary_unit * omega**2
        + 2 * imaginary_unit * omega**4
        - omega**5
    ) / 24
    assert sp.simplify(integrated - expected) == 0

    resonance_polynomial = sp.simplify(
        (imaginary_unit * omega**2 + 2 * imaginary_unit * omega**4 - omega**5).subs(
            omega, 2 * imaginary_unit
        )
    )
    assert resonance_polynomial == -4 * imaginary_unit

    modular_volume = sp.pi / 3
    liouville_modulus_product = -1 / sp.pi
    reduced_worldsheet_integral = sp.simplify(
        modular_volume * liouville_modulus_product
    )
    assert reduced_worldsheet_integral == -sp.Rational(1, 3)

    worldsheet_resonance = sp.simplify(
        sewing_geometric_coefficient
        * imaginary_unit
        * g_s**2
        * reduced_worldsheet_integral
    )
    expected_resonance = -8 * sp.pi**2 * imaginary_unit * g_s**2 / 3
    assert sp.simplify(worldsheet_resonance - expected_resonance) == 0

    mu_dictionary = {mu: 1 / (4 * sp.pi * g_s)}
    matrix_resonance = -imaginary_unit / (6 * mu**2)
    assert sp.simplify(matrix_resonance.subs(mu_dictionary) - expected_resonance) == 0

    bry_dictionary = {g_s_bry: 2 * g_s}
    bry_prefactor = 2 * sp.pi**2 * g_s_bry**2
    our_prefactor = 8 * sp.pi**2 * g_s**2
    assert sp.simplify(bry_prefactor.subs(bry_dictionary) - our_prefactor) == 0

    numeric_omega = mp.mpf("0.7")
    numeric_x = mp.mpf("0.2")
    errors = []
    for numeric_mu in (mp.mpf(100), mp.mpf(200)):
        errors.append(
            abs(
                exact_phase_ratio(numeric_mu, numeric_omega, numeric_x)
                - asymptotic_phase_ratio(numeric_mu, numeric_omega, numeric_x)
            )
        )
    # Truncation after mu^-2 must leave an error proportional to mu^-3.
    assert mp.mpf("7.8") < errors[0] / errors[1] < mp.mpf("8.2")

    print("genus-one two-point algebra checks: PASS")
    print("target-free sewing prefactor = 8*pi^2*i*g_s^2")
    print(f"integrated one-loop coefficient = {expected}")
    print(f"resonance polynomial = {resonance_polynomial}")
    print(f"reduced worldsheet integral = {reduced_worldsheet_integral}")
    print(f"our-convention resonance = {expected_resonance}")
    print(f"Gamma-ratio truncation errors (mu=100,200) = {errors}")


if __name__ == "__main__":
    main()
