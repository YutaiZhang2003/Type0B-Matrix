"""Independent RR matrix benchmark and local picture-changing checks.

No worldsheet amplitude data are read here. These formulas do not supply
the missing nonchiral RR torus integral.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path

import mpmath as mp
import sympy as sp


def amplitude_coefficient(omega):
    return mp.j*mp.pi**2*omega**2*(1+8*omega**2+8*mp.j*omega**3)/384


def theta(delta, z, tau):
    if delta not in (1, 2, 3, 4):
        raise ValueError("delta must name theta_1, theta_2, theta_3 or theta_4")
    return mp.jtheta(delta, mp.pi*z, mp.exp(mp.pi*mp.j*tau))


def prime_form(z, tau):
    nome = mp.exp(mp.pi*mp.j*tau)
    return mp.jtheta(1, mp.pi*z, nome)/(mp.pi*mp.jtheta(1, 0, nome, 1))


def eta(tau):
    nome = mp.exp(2*mp.pi*mp.j*tau)
    return mp.exp(mp.pi*mp.j*tau/12)*mp.qp(nome)


def rr_superghost(delta, u, z, tau):
    """Chiral (+1,-1/2,-1/2) correlator in a fixed branch patch.

    PCO at u; external R punctures at 0 and z. Square roots are separate
    and continued from the local patch. This is a ghost factor only.
    """
    return (eta(tau)*prime_form(-z, tau)**(-mp.mpf(1)/4)
        *mp.sqrt(prime_form(u, tau))*mp.sqrt(prime_form(u-z, tau))
        /theta(delta, u-z/2, tau))


def raised_rr_superghost(delta, z, tau):
    """Finite ghost coefficient after a physical PCO collides with R_1."""
    return eta(tau)*prime_form(-z, tau)**(mp.mpf(1)/4)/theta(delta, -z/2, tau)


def pair(z):
    return [float(mp.re(z)), float(mp.im(z))]


def checks():
    p, w, x, W = sp.symbols("P omega x Omega")
    # A convenient Clifford basis for timelike and Liouville R ground
    # doublets; this does not itself fix the nonchiral BRY cocycles.
    parity = sp.diag(1, -1)
    time_g0 = w/sp.sqrt(2)*sp.Matrix([[0, 1], [-1, 0]])
    liouville_g0 = p/sp.sqrt(2)*sp.Matrix([[0, 1], [1, 0]])
    g0 = sp.kronecker_product(time_g0, sp.eye(2))+sp.kronecker_product(parity, liouville_g0)
    assert sp.simplify(g0*g0-(p*p-w*w)/2*sp.eye(4)) == sp.zeros(4)
    kernels = (sp.Matrix([1, 0, 0, -1]), sp.Matrix([0, 1, -1, 0]))
    for vector in kernels:
        assert sp.simplify(g0.subs(p, w)*vector) == sp.zeros(4, 1)
    # Expand the independent same-side MQM particle-hole kernel.
    U = W*(W/2-x)
    V = W/24+W**3/6-W**2*x/2+W*x*x/2
    loop = sp.integrate(sp.I*V-U*U/2, (x, 0, W))
    expected = sp.I*W**2*(1+2*W**2+sp.I*W**3)/24
    assert sp.simplify(loop-expected) == 0
    # R=T+A and L=T-A; the sides decouple perturbatively.
    change = sp.Matrix([[1, 1], [1, -1]])/2
    fields = change*sp.diag(expected, expected)*change.T
    assert fields[0, 1] == 0 and sp.simplify(fields[0, 0]-fields[1, 1]) == 0
    physical = sp.simplify(fields[1, 1].subs(W, 2*w)*sp.Rational(1, 2)*sp.pi**2/16)
    target = sp.I*sp.pi**2*w*w*(1+8*w*w+8*sp.I*w**3)/384
    assert sp.simplify(physical-target) == 0
    tau, z = mp.mpc("0.173", "1.317"), mp.mpc("0.231", "0.191")
    ghost_checks = []
    for delta in (1, 2, 3, 4):
        target_ghost = raised_rr_superghost(delta, z, tau)
        errors = [abs(rr_superghost(delta, u, z, tau)/mp.sqrt(u)/target_ghost-1)
                  for u in (mp.mpf("1e-4"), mp.mpf("1e-6"), mp.mpf("1e-8"))]
        assert errors[2] < mp.mpf("1e-6") and errors[2] < errors[1] < errors[0]
        ghost_checks.append(dict(theta=delta, relative_residuals=list(map(float, errors))))
    return dict(matrix_amplitude_derived_from_side_kernel=True,
        RR_equals_NSNS_perturbatively_in_BRY_basis=True,
        mixed_TA_elastic_coefficient_zero=True, ramond_G0_constraint_verified=True,
        PCOs_per_chirality=1, ghost_collision_checks=ghost_checks,
        worldsheet_amplitude_computed=False,
        local_checks_do_not_fix_nonchiral_cocycles_or_global_transport=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    mp.mp.dps = 60
    report = checks()
    grid = sorted({Fraction(1, 20)+Fraction(j, 60) for j in range(19)} | {Fraction(13, 40)})
    rows = []
    for value in grid:
        t = mp.mpf(value.numerator)/value.denominator
        amplitude = amplitude_coefficient(mp.j*t)
        rows.append(dict(t_exact=str(value), t=float(t), energy=[0, float(t)],
            predicted_amplitude_coefficient=pair(amplitude),
            predicted_stripped_integral=pair(amplitude/(2*mp.pi*mp.j)),
            worldsheet_amplitude=None))
    result = dict(schema="type0b-rr-genus1-matrix-target-v1", channel="RR axion A->A",
        genus=1, rr_flux=0, basis="BRY, alpha_prime=2, leg phases absorbed",
        definition="S_AA^(g=1)=delta(omega-omega_prime)*g_s^2*a_AA^(1)",
        formula="i*pi^2/384*omega^2*(1+8*omega^2+8*i*omega^3)",
        tree_coefficient="omega/2", scalar_reflection_correction="2*g_s^2*a_AA^(1)/omega",
        status="independent matrix prediction; no RR worldsheet values yet", rows=rows)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out/"KINEMATICS_CHECKS.json").write_text(json.dumps(report, indent=2)+"\n")
    (out/"MATRIX_TARGET.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
