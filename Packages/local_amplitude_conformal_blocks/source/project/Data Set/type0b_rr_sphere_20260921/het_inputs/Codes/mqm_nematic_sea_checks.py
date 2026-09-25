#!/usr/bin/env python3
"""Finite-spin and frozen-chain tests of the nematic MQM candidate.

This does not establish a many-body sea or string scattering.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import sympy as sp

from mqm_anisotropic_spin_checks import compositions


def oscillator_bounds():
    rows = []
    for d in (3, 4):
        for kappa in (1, 2, 3):
            basis = list(compositions(kappa, d))
            lookup = {n: i for i, n in enumerate(basis)}
            dim = len(basis)
            E = {}
            for a in range(d):
                for b in range(d):
                    mat = np.zeros((dim, dim))
                    for col, occ in enumerate(basis):
                        if not occ[b]:
                            continue
                        nxt = list(occ)
                        nxt[b] -= 1
                        coeff = np.sqrt(occ[b] * (nxt[a] + 1))
                        nxt[a] += 1
                        mat[lookup[tuple(nxt)], col] = coeff
                    E[a, b] = mat
            P = sum(np.kron(E[a, b], E[b, a])
                    for a in range(d) for b in range(d))
            Q = sum(np.kron(E[a, b], E[a, b])
                    for a in range(d) for b in range(d))
            q_eigs = np.linalg.eigvalsh(Q)
            expected = [kappa*(kappa+d-1)-l*(l+d-1)
                        for l in range(kappa+1)]
            assert all(min(abs(e-x) for x in expected) < 1e-9 for e in q_eigs)
            assert abs(q_eigs[-1]-kappa*(kappa+d-1)) < 1e-9
            C = 2*kappa*kappa+kappa*(d-1)
            B = C*np.eye(dim*dim)-P-Q
            minB = np.linalg.eigvalsh(B)[0]
            assert minB > -1e-9
            rows.append({"d": d, "kappa": kappa,
                         "max_Q": float(q_eigs[-1]),
                         "min_C_minus_P_minus_Q": float(minB)})
    gamma, J, d, kap = sp.symbols("gamma J d kap")
    p, q = sp.symbols("P Q")
    eta_pair_c = gamma*(gamma-1)-J*(d-1)
    target = gamma*(gamma-1)+J/kap*(2*kap**2-p-q)
    rewritten = eta_pair_c+J/kap*(2*kap**2+kap*(d-1)-p-q)
    assert sp.expand(target-rewritten) == 0
    return rows


def hessian_and_director():
    qi, qj, pi, pj, t = sp.symbols("qi qj pi pj t", real=True)
    wi, wj = (qi+sp.I*pi)/sp.sqrt(2), (qj+sp.I*pj)/sp.sqrt(2)
    zi0 = 1-t*t*wi*sp.conjugate(wi)/2
    zj0 = 1-t*t*wj*sp.conjugate(wj)/2
    h = zi0*zj0+t*t*sp.conjugate(wi)*wj
    b = zi0*zj0+t*t*wi*wj
    penalty = sp.expand(2-h*sp.conjugate(h)-b*sp.conjugate(b)).coeff(t, 2)
    expected = (qi-qj)**2+pi*pi+pj*pj
    assert sp.simplify(penalty-expected) == 0
    # Hermitian SO(d) generators have zero expectation on a real spinor.
    z = sp.Matrix(sp.symbols("z0:4", real=True))
    for a in range(4):
        for b in range(a+1, 4):
            L = sp.zeros(4)
            L[a, b], L[b, a] = sp.I, -sp.I
            assert sp.expand((z.T*L*z)[0]) == 0
    return {"coherent_pair_penalty_quadratic": str(sp.factor(penalty)),
            "SOd_charge_of_real_director": "zero"}


def three_site_pair_vertex():
    J, w1, w2 = sp.symbols("J w1 w2", positive=True)
    A = sp.Matrix([[0, w1, 0], [w1, 0, w2], [0, w2, 0]])
    S = sp.diag(w1, w1+w2, w2)
    L = S-A
    # Fix total linearized rotation charge and quotient its zero coordinate.
    U = sp.Matrix([[1/sp.sqrt(2), 1/sp.sqrt(6)],
                   [0, -2/sp.sqrt(6)],
                   [-1/sp.sqrt(2), 1/sp.sqrt(6)]])
    assert U.T*U == sp.eye(2)
    assert U.T*sp.ones(3, 1) == sp.zeros(2, 1)
    Kq, Kp = 2*J*U.T*L*U, 2*J*U.T*S*U
    at_equal = {w1: 1, w2: 1}
    assert Kq.subs(at_equal) == sp.diag(2*J, 6*J)
    assert Kp.subs(at_equal) == sp.diag(2*J, 10*J/3)
    rq = sp.diag(1, sp.root(sp.Rational(5, 9), 4))
    # q=rq(c+c†)/sqrt2, p=rq^-1(c-c†)/(i sqrt2).
    # Off-diagonal coefficient multiplies c1†c2† (not 1/2 times it).
    dq, dp = Kq.diff(w1), Kp.diff(w1)
    vertex_matrix = (rq*dq*rq-rq.inv()*dp*rq.inv())/2
    vertex = sp.simplify(vertex_matrix[0, 1])
    target = J*(sp.sqrt(5)-1)/(2*sp.root(5, 4))
    assert sp.simplify(vertex-target) == 0
    assert vertex_matrix[0, 0] == 0
    assert sp.simplify(vertex_matrix[1, 1]) == 0
    # Uniform dilation rescales both matrices, so pair production cancels.
    uniform = sp.simplify(
        (rq*(Kq.diff(w1)+Kq.diff(w2))*rq
         -rq.inv()*(Kp.diff(w1)+Kp.diff(w2))*rq.inv())/2)
    assert uniform == sp.zeros(2)
    return {"relative_frequencies_at_unit_weights": ["2J", "2sqrt(5)J"],
            "edge1_pair_creation_vertex": str(vertex),
            "coefficient_at_J1": float(vertex.subs(J, 1)),
            "uniform_weight_variation_vertex": "zero"}


def uniform_chain_and_continuum():
    J, rho, k, gamma = sp.symbols("J rho k gamma", positive=True)
    theta = sp.symbols("theta", real=True)
    S = 2*rho*rho
    L = 4*rho*rho*sp.sin(theta/2)**2
    omega2 = 4*J*J*S*L
    assert sp.simplify(omega2-32*J*J*rho**4*sp.sin(theta/2)**2) == 0
    # Dimensionless lattice momentum theta=k_x/rho.
    limit = sp.limit(omega2.subs(theta, k/rho)/k**2, k, 0)
    assert limit == 8*J*J*rho*rho
    zpe = sp.integrate(
        2*sp.sqrt(2)*J*rho*rho*sp.sin(theta/2)-2*J*rho*rho,
        (theta, 0, 2*sp.pi))/(2*sp.pi)
    assert sp.simplify(zpe-(4*sp.sqrt(2)/sp.pi-2)*J*rho*rho) == 0
    # Legendre elimination of p, including material rather than partial time.
    p, Dtq, qx = sp.symbols("p Dtq qx", real=True)
    lag = rho*p*Dtq-2*J*rho**3*p*p-J*rho*qx*qx
    pstar = sp.solve(sp.diff(lag, p), p)[0]
    reduced = sp.factor(lag.subs(p, pstar))
    expected = Dtq**2/(8*J*rho)-J*rho*qx*qx
    assert sp.simplify(reduced-expected) == 0
    # Direct finite-chain frequencies and depletion; zero mode removed.
    depletion = []
    for N in (64, 128, 256, 512, 1024, 2048):
        th = 2*np.pi*np.arange(1, N)/N
        Kq = 8*np.sin(th/2)**2
        Kp = np.full_like(th, 4.0)
        occupation = (np.sqrt(Kp/Kq)+np.sqrt(Kq/Kp)-2)/4
        depletion.append((N, float(np.mean(occupation))))
    slope = np.polyfit(np.log([x[0] for x in depletion]),
                       [x[1] for x in depletion], 1)[0]
    expected_slope = 1/(2*np.sqrt(2)*np.pi)
    assert abs(slope-expected_slope) < 0.006
    return {"omega_squared": str(omega2),
            "speed_in_x": "2sqrt(2) J rho",
            "speed_in_tau_if_vF_pi_gamma_rho": "2sqrt(2)J/(pi gamma)",
            "spin_ground_energy_per_site_per_flavor": str(zpe),
            "material_derivative_action_density": str(reduced),
            "depletion_per_flavor": depletion,
            "log_depletion_fit": float(slope),
            "asymptotic_log_depletion_coefficient": float(expected_slope)}


def main():
    result = {"scope": "Exact finite-spin bounds and frozen-chain Gaussian tests; fluid/time-of-flight dictionary conditional.",
              "oscillator_bounds": oscillator_bounds(),
              "director_hessian": hessian_and_director(),
              "three_site": three_site_pair_vertex(),
              "uniform_chain": uniform_chain_and_continuum()}
    path = (Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_nematic_sea_results.json')
    path.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
