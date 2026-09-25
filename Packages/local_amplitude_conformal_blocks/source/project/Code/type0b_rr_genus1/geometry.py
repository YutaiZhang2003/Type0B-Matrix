"""Uniformize the two pants after cutting the third Ramond theta tube.

Pants coordinates are w and v, with punctures at infinity (NS), one
(internal R), and zero (external R). Sewing is (1/w)(1/v)=q_NS and
(w-1)(v-1)=q_R. These maps do not fix spin lifts or the CFT anomaly factor.
"""
from __future__ import annotations

import cmath
from dataclasses import dataclass


@dataclass(frozen=True)
class MarkedTorusCoordinates:
    tau: complex
    z: complex
    torus_nome: complex
    puncture_nome: complex
    q_ns: complex
    q_r: complex
    fixed_a: complex
    fixed_b: complex
    derivative_w: complex
    derivative_v: complex
    second_derivative_w: complex
    second_derivative_v: complex

    def mobius(self, w):
        # Fixed-point form avoids cancellation at the repelling fixed
        # point when the torus nome is exponentially small.
        a, b, Q = self.fixed_a, self.fixed_b, self.torus_nome
        return a+Q*(a-b)*(w-a)/((w-b)-Q*(w-a))

    def mobius_derivative(self, w):
        a, b, Q = self.fixed_a, self.fixed_b, self.torus_nome
        return Q*(a-b)**2/((w-b)-Q*(w-a))**2


def marked_coordinates(tau, z):
    """External points map to z (w=0) and 0 (v=0), modulo periods."""
    tau, z = complex(tau), complex(z)
    if tau.imag <= 0 or not 0 < z.imag < tau.imag:
        raise ValueError("Use a marked strip with Im(tau)>Im(z)>0")
    Q, s = cmath.exp(2j*cmath.pi*tau), cmath.exp(2j*cmath.pi*z)
    q_ns = s*(1-Q)**2/(1-Q*s)**2
    q_r = Q*(1-s)**2/(s*(1-Q)**2)
    a = (1-Q*s)/(1-Q)
    b = a/s
    derivative_w = (a-b)/(2j*cmath.pi*a*b)
    derivative_v = q_ns*(b-a)/(2j*cmath.pi)
    common = q_ns*(a+b)
    return MarkedTorusCoordinates(tau, z, Q, s, q_ns, q_r, a, b,
        derivative_w, derivative_v, derivative_w*common, derivative_v*common)


def ramond_level_one_transport(weight, derivative, second_derivative):
    """Coefficients expressing flat G_-1 R in the local G_-1 R/G_0 R basis.

    Local G_-1 R = f'^(h+1) G_-1 R(flat)
                     + (3/4) f'' f'^(h-1) G_0 R(flat).
    The principal power is local only; callers must continue its phase with
    the spin lift when crossing charts. This function is not global sewing.
    """
    return (derivative**(-weight-1),
            -.75*second_derivative*derivative**(-weight-2))
