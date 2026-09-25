"""Analytic RR collision-disk primitives and local picture changing.

The input C_pq polynomials must be in the flat, physical spin frame. This
module computes the free Taylor factors and finite parts, not that spin
projection. No TT collision Ward coefficient is substituted for an RR one.
"""
from __future__ import annotations

import math

import mpmath as mp
import numpy as np

from kinematics import eta
from picture_changed import free_outer_factor, picture_connection, time_contact


def constant(value, order):
    out = np.zeros((order+1, order+1), complex)
    out[0, 0] = value
    return out


def multiply(a, b, order):
    """Multiply two Taylor polynomials, truncating by TOTAL degree."""
    out = constant(0, order)
    for i, j in np.argwhere(a != 0):
        for k, l in np.argwhere(b != 0):
            if i+j+k+l <= order:
                out[i+k, j+l] += a[i, j]*b[k, l]
    return out


def unit_power(a, exponent, order):
    a = np.asarray(a, complex)
    if a[0] != 1:
        raise ValueError('a unit constant is required')
    out = np.zeros(order+1, complex)
    out[0] = 1
    for n in range(1, order+1):
        for j in range(1, min(n+1, len(a))):
            out[n] += ((exponent+1)*j-n)*a[j]*out[n-j]/n
    return out


def holomorphic(a, order):
    out = constant(0, order)
    out[:len(a), 0] = a
    return out


def evaluate(poly, z):
    powers = complex(z)**np.arange(poly.shape[0])
    return powers@poly@powers.conjugate()


class RRCollisionKernel:
    """The period-one free factor near z=0, for all four spins.

    B_RR = leading * |z|^power * free_polynomial(z,zbar).
    Connection polynomials are z*d and zbar*dbar. The Liouville components
    C_pq have powers z^(a+1-p) zbar^(a+1-q) factored out; hence all terms
    below share |z|^(2a). The same analytic omega enters both chiral factors.
    """

    def __init__(self, *, tau, omega, delta, order=4, dps=45):
        if delta not in (1, 2, 3, 4) or order < 2 or not isinstance(order, int):
            raise ValueError('theta label 1..4 and integer order >=2 required')
        if complex(tau).imag <= 0 or complex(omega) == 0:
            raise ValueError('positive torus height and nonzero energy required')
        self.tau, self.omega, self.delta, self.order = complex(tau), complex(omega), delta, order
        odd = int(delta == 1)
        self.free_power = 2*self.omega**2+.25-odd
        with mp.workdps(dps):
            nome = mp.exp(mp.pi*mp.j*tau)
            th = lambda d, j: mp.pi**j*mp.jtheta(d, 0, nome, j)
            first = th(1, 1)
            e = np.array([complex(th(1, j+1)/(mp.factorial(j+1)*first))
                          for j in range(order+1)])
            tlead = th(delta, odd)/(2**odd)
            t = np.array([complex(th(delta, j+odd)
                /(2**(j+odd)*mp.factorial(j+odd)*tlead)) for j in range(order+1)])
            self.leading = complex(self.omega**2*abs(eta(tau))**3
                /(16*mp.sqrt(8*mp.pi**2*mp.im(tau))*abs(tlead)))
        e[0] = t[0] = 1  # ratios above are identically one
        a = self.omega**2+.125
        # Conjugate geometry FIRST; never conjugate the continued energy.
        h = np.convolve(unit_power(e, a, order), unit_power(t, -.5, order))[:order+1]
        ah = np.convolve(unit_power(e.conjugate(), a, order),
                         unit_power(t.conjugate(), -.5, order))[:order+1]
        self.free = multiply(holomorphic(h, order), holomorphic(ah, order).T, order)
        # exp[pi*omega^2*(z-zbar)^2/(2*tau2)]
        gaussian = constant(1, order)
        x = constant(0, order)
        coefficient = math.pi*self.omega**2/(2*self.tau.imag)
        x[2, 0], x[1, 1], x[0, 2] = coefficient, -2*coefficient, coefficient
        term = constant(1, order)
        for n in range(1, order//2+1):
            term = multiply(term, x, order)/n
            gaussian += term
        self.free = multiply(self.free, gaussian, order)
        # z theta_delta'(z/2)/theta_delta(z/2) = 2*odd+2*z*t'/t.
        log_e = np.convolve(np.arange(order+1)*e, unit_power(e, -1, order))[:order+1]
        log_t = np.convolve(np.arange(order+1)*t, unit_power(t, -1, order))[:order+1]
        d = -1.5*log_e+2*log_t
        d[0] = -1.5+2*odd
        self.d = holomorphic(d, order)
        self.d[2, 0] -= math.pi/self.tau.imag
        self.d[1, 1] += math.pi/self.tau.imag
        self.da = self.d.conjugate().T
        self.contact = constant(0, order)
        self.contact[1, 1] = complex(time_contact(self.omega, self.tau))

    def raise_components(self, components):
        if set(components) != {(0, 0), (0, 1), (1, 0), (1, 1)}:
            raise ValueError('all four RR Liouville components are required')
        shape = (self.order+1, self.order+1)
        if any(np.asarray(v).shape != shape for v in components.values()):
            raise ValueError('component Taylor arrays have an incompatible order')
        c = components
        combined = (c[1, 1]+multiply(self.d, c[0, 1], self.order)
                    +multiply(self.da, c[1, 0], self.order)
                    +multiply(multiply(self.d, self.da, self.order)+self.contact,
                              c[0, 0], self.order))
        return self.leading*multiply(self.free, combined, self.order)

    def density_power(self, bridge, family):
        """Power after on-shell cancellation of the analytic external energy."""
        if family not in (0, 1):
            raise ValueError('bridge family must be 0 or 1')
        return 1+bridge**2+family-4-int(self.delta == 1)


def disk_finite_part(poly, *, power, radius, log_scale=None):
    """Meromorphic radial primitive after the exact angular integral.

    At an exact radial pole the finite part depends on a specified scale.
    Reject the call unless that scale is provided; do not silently choose it.
    The pole residue is with respect to `power`, not the bridge momentum.
    """
    if radius <= 0 or not np.isfinite(radius):
        raise ValueError('positive finite collision radius required')
    value, residue = 0j, 0j
    for m in range(min(poly.shape)):
        coefficient = complex(poly[m, m])
        if coefficient == 0:
            continue
        exponent = complex(power)+2*m+2
        if abs(exponent) < 1e-12:
            residue += 2*math.pi*coefficient
            if log_scale is None or log_scale <= 0:
                raise ValueError('radial pole: specify a positive logarithmic subtraction scale')
            value += 2*math.pi*coefficient*math.log(radius/log_scale)
        else:
            value += 2*math.pi*coefficient*np.exp(exponent*math.log(radius))/exponent
    return dict(value=value, residue=residue)


def annulus(poly, *, power, inner, outer):
    """Convergent radius difference, including exact logarithmic powers."""
    if not 0 < inner < outer:
        raise ValueError('require 0 < inner radius < outer radius')
    value = 0j
    log_ratio = math.log(outer/inner)
    for m in range(min(poly.shape)):
        exponent = complex(power)+2*m+2
        radial = (log_ratio if exponent == 0 else
                  np.exp(exponent*math.log(inner))*np.expm1(exponent*log_ratio)/exponent)
        value += 2*math.pi*poly[m, m]*radial
    return value
