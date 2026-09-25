"""Free factors and the local RR picture-changing differential operator.

The input Liouville components must ALREADY be in the same flat-torus
spin frame as delta. This module does not supply that interacting sewing
map, spectral integrals, collision subtraction or a string amplitude.
"""
from __future__ import annotations

import mpmath as mp
import sympy as sp

from kinematics import theta, prime_form, eta


def theta_log_derivative(delta, z, tau):
    nome = mp.exp(mp.pi*mp.j*tau)
    return mp.pi*mp.jtheta(delta, mp.pi*z, nome, 1)/mp.jtheta(delta, mp.pi*z, nome)


def time_green(z, tau):
    """Positive-sign timelike Green function, with its zero mode removed."""
    return 2*mp.log(abs(prime_form(z, tau)))-2*mp.pi*mp.im(z)**2/mp.im(tau)


def picture_connection(delta, z, tau):
    """D at the raised puncture 0, the other puncture being at z.

    D=(1/omega^2)*partial_0 log B_time -4*partial_0 log S_delta.
    The derivative of theta_delta is with respect to its FULL argument;
    the factor 1/2 from the argument z/2 has already been included.
    """
    e = theta_log_derivative(1, z, tau)
    t = theta_log_derivative(delta, z/2, tau)
    return -mp.mpf(3)/2*e+t-2*mp.pi*mp.j*mp.im(z)/mp.im(tau)


def time_contact(omega, tau):
    """Connected same-puncture time-boson contraction; omega is analytic."""
    if omega == 0:
        raise ValueError("At zero energy, take the joint limit with the G0 components")
    return -mp.pi/(omega**2*mp.im(tau))


def raised_combination(components, *, omega, delta, z, tau):
    """Contract C_pq with p,q in {0,1}, D_1=G_-1, D_0=G_0.

    Each C_pq is the BRY external-state contraction of the full Liouville
    correlator (D_p Dbar_q R)(0) R(z), with holomorphic mode first.
    No conjugation of a Liouville component or omega is performed.
    """
    if set(components) != {(0, 0), (0, 1), (1, 0), (1, 1)}:
        raise ValueError("Supply all four G_-1/G_0 Liouville components")
    d = picture_connection(delta, z, tau)
    da = mp.conj(d)  # only geometry; D does not contain omega
    return (components[1, 1]+d*components[0, 1]+da*components[1, 0]
            +(d*da+time_contact(omega, tau))*components[0, 0])


def free_outer_factor(*, omega, delta, z, tau):
    """Stripped period-one free factor in the previous NSNS measure.

    Includes omega^2 from RR vertices, two PCO factors (-1/2), GSO 1/2,
    and bc/automorphism 1/2. Uses dx dy, not twice the area. It excludes
    Liouville components and dP_NS dP_R/pi^2, and does not certify their
    normalization or the overall conversion to an S-matrix coefficient.
    """
    return (omega**2*abs(eta(tau))**3*abs(prime_form(z, tau))**(mp.mpf(1)/4)
        *mp.exp(omega**2*time_green(z, tau))
        /(16*mp.sqrt(8*mp.pi**2*mp.im(tau))*abs(theta(delta, z/2, tau))))


def bry_ground_clifford(P, k):
    """G0 matrices in the physical (sigma,mu) x (V_R+,V_R-) basis.

    The timelike holomorphic and antiholomorphic signs implement the BRY
    sigma/mu convention. They are fixed by G0 W_k=G0bar W_k=0 for
    W_{+/-}=(sigma V_R+ +/- mu V_R-)/sqrt(2), P=+/-k.
    """
    u, ub = (1-sp.I)/sp.sqrt(2), (1+sp.I)/sp.sqrt(2)
    mh, ma = sp.Matrix([[0, sp.I], [1, 0]]), sp.Matrix([[0, -sp.I], [1, 0]])
    lh, la = sp.I*P/sp.sqrt(2)*ub*mh, sp.I*P/sp.sqrt(2)*u*ma
    th, ta = -k/sp.sqrt(2)*ub*mh, k/sp.sqrt(2)*u*ma
    parity = sp.diag(1, -1)
    total_h = sp.kronecker_product(th, sp.eye(2))+sp.kronecker_product(parity, lh)
    total_a = sp.kronecker_product(ta, sp.eye(2))+sp.kronecker_product(parity, la)
    return dict(liouville_h=lh, liouville_a=la, time_h=th, time_a=ta,
                total_h=total_h, total_a=total_a)
