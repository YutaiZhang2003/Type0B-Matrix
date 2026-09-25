#!/usr/bin/env python3
"""Scoped checks of an SO(23)-invariant spin-MQM deformation.

These are exact finite-spin and frozen-coordinate large-spin checks.
They do not calculate collective string scattering or construct a sea.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import sympy as sp


def compositions(total: int, slots: int):
    if slots == 1:
        yield (total,)
    else:
        for n in range(total + 1):
            for rest in compositions(total - n, slots - 1):
                yield (n,) + rest


def pair_singlet_checks():
    d = 24
    m = d - 1
    size = d * d
    P = sp.SparseMatrix(size, size, {
        (b * d + a, a * d + b): 1 for a in range(d) for b in range(d)
    })
    Q = sp.SparseMatrix(size, size, {
        (a * d + a, b * d + b): 1 for a in range(d) for b in range(d)
    })
    NV = sp.diag(*[int(a > 0) + int(b > 0)
                   for a in range(d) for b in range(d)])
    U = sp.SparseMatrix(size, 2, {
        (0, 0): 1, **{(a * d + a, 1): 1 / sp.sqrt(m) for a in range(1, d)}
    })
    J, h = sp.symbols("J h", positive=True)
    H = J * (Q - P) + h * NV
    block = sp.Matrix([[0, sp.sqrt(m) * J],
                       [sp.sqrt(m) * J, (m - 1) * J + 2 * h]])
    assert U.T * U == sp.eye(2)
    assert (H * U - U * block).applyfunc(sp.simplify) == sp.zeros(size, 2)
    gap2 = ((m - 1) * J + 2 * h)**2 + 4 * m * J**2
    C = block * block.diff(J) - block.diff(J) * block
    # In a two-level system, Tr([H,V]^dagger[H,V])/(2 gap^2)=|V_ge|^2.
    matrix_element_sq = sp.simplify(sp.trace(C.T * C) / (2 * gap2))
    expected_sq = 4 * m * h**2 / gap2
    assert sp.simplify(matrix_element_sq - expected_sq) == 0
    pair_weight = (1 - ((m - 1) * J + 2 * h) / sp.sqrt(gap2)) / 2
    assert sp.limit(pair_weight, J, sp.oo) == sp.Rational(1, d)
    assert sp.limit(pair_weight * h**2 / J**2, h, sp.oo) == sp.Rational(m, 4)
    # The SO(24) symmetric-traceless singlet is mixed with its trace by NV.
    st = sp.Matrix([sp.sqrt(m), -1]) / sp.sqrt(d)
    trace = sp.Matrix([1, sp.sqrt(m)]) / sp.sqrt(d)
    trace_mixing = (trace.T * sp.diag(0, 2) * st)[0]
    assert trace_mixing == -2 * sp.sqrt(m) / d
    samples = []
    for Jn, hn in [(0.01, 1.0), (1.0, 1.0), (1000.0, 1.0)]:
        bn = np.array(block.subs({J: Jn, h: hn}), dtype=float)
        ev, vec = np.linalg.eigh(bn)
        vn = np.array(block.diff(J), dtype=float)
        direct = abs(vec[:, 1] @ vn @ vec[:, 0])**2
        target = float(expected_sq.subs({J: Jn, h: hn}))
        assert np.isclose(direct, target, atol=1e-12)
        samples.append({"J": Jn, "h": hn, "ground_pair_weight": float(vec[1, 0]**2),
                        "density_transition_squared": direct})
    return {"exact_d": d, "block": str(block),
            "transition_squared": str(matrix_element_sq),
            "close_pair_weight": "1/24", "trace_mixing": str(trace_mixing),
            "samples": samples}


def large_spin_pair_checks():
    """Independent finite oscillator matrices check positivity and collapse."""
    d = 3
    rows = []
    for kappa in range(1, 5):
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
        ident = np.eye(dim * dim)
        B = kappa**2 * ident - P + Q
        pe = np.linalg.eigvalsh(P)
        qe = np.linalg.eigvalsh(Q)
        be = np.linalg.eigvalsh(B)
        spe = np.linalg.eigvalsh(Q - P)
        assert pe[-1] <= kappa**2 + 1e-10
        assert qe[0] > -1e-10 and be[0] > -1e-10
        assert abs(spe[0] + kappa**2) < 1e-10
        lam = 0.1
        naive = lam**2 + lam * spe[0] / kappa
        repaired = lam * (lam - 1) + lam * be[0] / kappa
        assert np.isclose(repaired, -0.09)
        rows.append({"kappa": kappa, "local_dimension_at_d3": dim,
                     "min_Q_minus_P": float(spe[0]),
                     "min_positive_B": float(be[0]),
                     "naive_min_pair_c_at_lambda_point1": float(naive),
                     "repaired_min_pair_c": float(repaired)})
    assert rows[2]["naive_min_pair_c_at_lambda_point1"] < -0.25
    return rows


def quadratic_checks():
    # One transverse flavor suffices: SO(23) makes 23 identical copies.
    qi, qj, pi, pj, h, gP, gQ = sp.symbols("qi qj pi pj h gP gQ", real=True)
    t = sp.symbols("t", real=True)
    wi, wj = (qi + sp.I*pi)/sp.sqrt(2), (qj + sp.I*pj)/sp.sqrt(2)
    zi0 = 1 - t**2 * sp.expand_complex(wi*sp.conjugate(wi))/2
    zj0 = 1 - t**2 * sp.expand_complex(wj*sp.conjugate(wj))/2
    overlap = zi0*zj0 + t**2*sp.conjugate(wi)*wj
    contraction = zi0*zj0 + t**2*wi*wj
    Pclass = sp.expand(overlap * sp.conjugate(overlap)).coeff(t, 2)
    Qclass = sp.expand(contraction * sp.conjugate(contraction)).coeff(t, 2)
    targetP = -((qi-qj)**2 + (pi-pj)**2)/2
    targetQ = -((qi-qj)**2 + (pi+pj)**2)/2
    assert sp.simplify(Pclass - targetP) == 0
    assert sp.simplify(Qclass - targetQ) == 0
    assert sp.simplify(Qclass - Pclass + 2*pi*pj) == 0
    A, S = sp.symbols("A S", real=True)
    Kq = h - (gP + gQ)*S + (gP + gQ)*A
    Kp = h - (gP + gQ)*S + (gP - gQ)*A
    assert sp.simplify(Kq.subs(A, S)) == h
    assert sp.simplify(Kp.subs(A, S)) == h - 2*gQ*S
    lam = sp.symbols("lambda", positive=True)
    assert sp.simplify(Kq.subs({gP: -lam, gQ: lam})) == h
    assert sp.simplify(Kp.subs({gP: -lam, gQ: lam})) == h - 2*lam*A
    # Escape: onsite normal h_i=gQ S_i and onsite anomalous B_ii=-gQ S_i.
    L = S - A
    M = h - (gP + gQ)*S + gP*A
    B = gQ*A - gQ*S
    assert sp.simplify(M.subs(h, gQ*S) + gP*L) == 0
    assert sp.simplify(B + gQ*L) == 0
    return {"P_quadratic_coherent_energy": str(sp.factor(Pclass)),
            "Q_quadratic_coherent_energy": str(sp.factor(Qclass)),
            "Kq": str(Kq), "Kp": str(Kp)}


def uniform_chain_checks():
    lam, ell = 0.1, 1.0
    n = np.arange(1, 200001, dtype=float)
    fourier_errors = []
    for k in (0.13, 0.51, 1.6):
        direct = 2*np.sum(np.cos(n*k)/n**2)/ell**2
        closed = (np.pi**2/3 - np.pi*abs(k) + k*k/2)/ell**2
        err = abs(direct - closed)
        assert err < 2/len(n)
        fourier_errors.append(err)
    ks = np.array([0.002, 0.004, 0.008, 0.016])
    hc = 2*lam*np.pi**2/(3*ell**2)
    deltaA = np.array([2*np.sum((1-np.cos(n*k))/n**2)/ell**2 for k in ks])
    omega = np.sqrt(hc*2*lam*deltaA)
    slope = np.polyfit(np.log(ks), np.log(omega), 1)[0]
    assert abs(slope - 0.5) < 0.003
    x = np.arange(8, dtype=float)
    diff = x[:, None] - x[None, :]
    np.fill_diagonal(diff, np.inf)
    Af = 1/diff**2
    eig = np.linalg.eigvalsh(Af)
    assert eig[-1] >= np.max(Af)
    return {"critical_h_at_unit_spacing": float(hc),
            "independent_Fourier_sum_errors": fourier_errors,
            "small_k_fitted_dispersion_exponent": float(slope),
            "finite_8_site_critical_h": float(2*lam*eig[-1])}


def common_bogoliubov_checks():
    # Arbitrary, noncommuting spatial kernels share the same INTERNAL map
    # when the normal and anomalous matrices are alpha*L and beta*L.
    rng = np.random.default_rng(7329)
    alpha, beta = 2.0, 0.7
    theta = 0.5*np.arctanh(-beta/alpha)
    c, s = np.cosh(theta), np.sinh(theta)
    residuals = []
    for _ in range(5):
        raw = rng.normal(size=(4, 4))
        L = raw.T @ raw
        M, B = alpha*L, beta*L
        transformed_M = (c*c+s*s)*M + 2*c*s*B
        transformed_B = (c*c+s*s)*B + 2*c*s*M
        residuals.append(float(np.linalg.norm(transformed_B)))
        assert np.allclose(transformed_B, 0, atol=1e-12)
        assert np.allclose(transformed_M, np.sqrt(alpha**2-beta**2)*L)
    M, B, dM, dB = sp.symbols("M B dM dB", real=True)
    anomalous = (M*dB - B*dM)/sp.sqrt(M*M-B*B)
    assert sp.simplify(anomalous.subs({dM: M, dB: B})) == 0
    return {"anomalous_residuals_for_five_distinct_kernels": residuals,
            "density_anomalous_vertex": str(anomalous)}


def main():
    result = {
        "scope": "Finite-spin algebra and controlled frozen-coordinate large-spin diagnostics; no sea or string S-matrix.",
        "exact_two_site_singlet": pair_singlet_checks(),
        "large_spin_regulator": large_spin_pair_checks(),
        "quadratic_algebra": quadratic_checks(),
        "uniform_chain": uniform_chain_checks(),
        "common_Bogoliubov_map": common_bogoliubov_checks(),
    }
    out = (Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_anisotropic_spin_results.json')
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
