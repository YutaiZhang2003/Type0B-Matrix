"""Algebraic NS plane-block resummation; no h recursion or pillow correlator.

Input: physical even/odd sewing coefficients from descendant c recursion.
Output: H_beta(qhat)=qhat**(beta/2) sum_n B_beta[n] qhat**n,
truncated at a specified nome order. BRY 2201.05621, (3.19)-(3.20).
Local OPE coefficients are back-expanded from that SAME truncated H, not
silently taken from a different finite sewing approximation.
"""

from functools import lru_cache
import math

import numpy as np
from scipy.special import elliprf


DTYPE = np.clongdouble
REPRESENTATIONS = ("elliptic_nome", "sewing")
ALGORITHM_VERSION = "ns_c_recursion_elliptic_v1"


def validate_representation(value):
    if value not in REPRESENTATIONS:
        raise ValueError("series_parameter must be 'elliptic_nome' or 'sewing'")
    return value


def representation_metadata(series_parameter, order):
    """Explicit provenance; a small adjacent-order change is not an error bound."""
    validate_representation(series_parameter)
    elliptic = series_parameter == "elliptic_nome"
    return {
        "series_parameter": series_parameter,
        "numerical_algorithm": ALGORITHM_VERSION if elliptic else "ns_c_recursion_sewing_v1",
        "q_order": int(order),
        "nome_order": int(order) if elliptic else None,
        "sewing_order": None if elliptic else int(order),
        "input_sewing_order": int(order),
        "maximum_twice_level": 2*int(order)+1,
        "q_definition": "qhat=exp(-pi*K(1-z)/K(z))" if elliptic else "q_s=z; q_t=1-z",
        "integration_measure": "plane_d2z",
        "local_ope_policy": "back_expansion_of_truncated_nome" if elliptic else "truncated_sewing",
        "accuracy_status": "requires_independent_convergence_checks",
    }


def multiply(a, b, length):
    result = np.zeros(length, dtype=DTYPE)
    if not length or not len(a) or not len(b):
        return result
    product = np.convolve(np.asarray(a, dtype=DTYPE), np.asarray(b, dtype=DTYPE))[:length]
    result[:len(product)] = product
    return result


def inverse(a, length):
    a = np.asarray(a, dtype=DTYPE)
    result = np.zeros(length, dtype=DTYPE)
    if a[0] == 0:
        raise ValueError("power-series inverse needs a nonzero constant")
    result[0] = 1/a[0]
    for n in range(1, length):
        k = min(n, len(a)-1)
        indices = np.arange(1, k+1)
        result[n] = -np.sum(a[indices]*result[n-indices], dtype=DTYPE)/a[0]
    return result


def exponential(a, length):
    a = np.asarray(a, dtype=DTYPE)
    result = np.zeros(length, dtype=DTYPE)
    result[0] = np.exp(a[0])
    for n in range(1, length):
        k = min(n, len(a)-1)
        result[n] = np.sum(np.arange(1, k+1)*a[1:k+1]*result[n-np.arange(1, k+1)], dtype=DTYPE)/n
    return result


def power(a, exponent, length):
    a = np.asarray(a, dtype=DTYPE)
    derivative = np.arange(1, len(a))*a[1:]
    quotient = multiply(derivative, inverse(a, length), length-1) if length > 1 else []
    log = np.zeros(length, dtype=DTYPE)
    log[0] = np.log(a[0])
    log[1:] = np.asarray(quotient)/np.arange(1, length)
    return exponential(DTYPE(exponent)*log, length)


def compose(a, coordinate, length):
    if coordinate[0] != 0:
        raise ValueError("composition requires a coordinate vanishing at the origin")
    result = np.zeros(length, dtype=DTYPE)
    for coefficient in np.asarray(a, dtype=DTYPE)[::-1]:
        result = multiply(result, coordinate, length)
        result[0] += coefficient
    return result


@lru_cache(maxsize=None)
def lambda_series(order):
    """Return z(q), z/(16q), theta3(q), in INTEGER q powers."""
    length = order+1
    theta = np.zeros(length, dtype=DTYPE)
    theta[0] = 1
    for n in range(1, math.isqrt(order)+1):
        theta[n*n] += 2
    reduced = np.zeros(length, dtype=DTYPE)
    n = 0
    while n*(n+1) <= order:
        reduced[n*(n+1)] += 1
        n += 1
    reduced2 = multiply(reduced, reduced, length)
    theta2 = multiply(theta, theta, length)
    u = multiply(multiply(reduced2, reduced2, length),
                 inverse(multiply(theta2, theta2, length), length), length)
    z = np.zeros(length, dtype=DTYPE)
    z[1:] = 16*u[:-1]
    for array in (z, u, theta):
        array.flags.writeable = False
    return z, u, theta


@lru_cache(maxsize=None)
def inverse_lambda_series(order):
    """q(z), 16q/z, theta3(q(z)) from the elliptic differential identity.

    A(z)=2F1(1/2,1/2;1;z)=theta3**2 and
    d log q/dz = 1/[z(1-z) A(z)**2]. This avoids ill-conditioned numerical
    series reversion of coefficients growing like 16**n.
    """
    length = order+1
    a = np.ones(length, dtype=DTYPE)
    for n in range(1, length):
        a[n] = a[n-1]*((DTYPE(n)-DTYPE(.5))/n)**2
    one_minus_z = np.zeros(length, dtype=DTYPE)
    one_minus_z[0] = 1
    if order:
        one_minus_z[1] = -1
    rational = inverse(multiply(one_minus_z, multiply(a, a, length), length), length)
    log_v = np.zeros(length, dtype=DTYPE)
    log_v[1:] = rational[1:]/np.arange(1, length)
    v = exponential(log_v, length)
    q = np.zeros(length, dtype=DTYPE)
    q[1:] = v[:-1]/16
    theta = power(a, .5, length)
    for array in (q, v, theta):
        array.flags.writeable = False
    return q, v, theta


def nome_geometry(z):
    """Principal plane chart, including conjugate lower/upper cut lips."""
    z = np.asarray(z, dtype=complex)
    if np.any(~np.isfinite(z)) or np.any((z == 0) | (z == 1)):
        raise ValueError("nome evaluation requires finite non-collision coordinates")
    # scipy's Carlson RF rejects an exactly negative real argument. Evaluate
    # the two cut lips by exact modular/continuation identities, not an epsilon
    # regulator. Signed zero chooses the same lip as the plane log(z).
    negative = (z.imag == 0) & (z.real < 0)
    above_one = (z.imag == 0) & (z.real > 1)
    regular = ~(negative | above_one)
    q = np.empty_like(z)
    zz = z[regular]
    q[regular] = np.exp(-np.pi*elliprf(np.zeros_like(zz), zz, np.ones_like(zz)) /
                        elliprf(np.zeros_like(zz), 1-zz, np.ones_like(zz)))
    if np.any(negative):
        x = z.real[negative]
        transformed = x/(x-1)
        positive_nome = np.exp(-np.pi*elliprf(0,transformed,1)/elliprf(0,1-transformed,1))
        cut_nome = np.asarray(-positive_nome,dtype=complex)
        cut_nome.imag = np.copysign(0.,z.imag[negative])
        q[negative] = cut_nome
    if np.any(above_one):
        x = z.real[above_one]
        lip = np.where(np.signbit(z.imag[above_one]),-1.,1.)
        k = (elliprf(0,1-1/x,1)+1j*lip*elliprf(0,1/x,1))/np.sqrt(x)
        q[above_one] = np.exp(-np.pi*elliprf(0,x,1)/k)
    if np.any(~np.isfinite(q)) or np.any((abs(q) >= 1) | (q == 0)):
        raise ArithmeticError("elliptic nome lies outside its open convergence disc")
    theta = np.ones_like(q)
    derivative = np.zeros_like(q)
    for n in range(1, 10001):
        term = 2*q**(n*n)
        theta += term
        derivative += n*n*term/q
        if np.max(abs(term), initial=0) < 2e-17:
            break
    else:
        raise ArithmeticError("theta3 did not converge")
    return q, theta, derivative


class EllipticComponentSeries:
    """A pair of component-specific H series with a universal plane prefactor."""

    series_parameter = "elliptic_nome"

    def __init__(self, h, effective_weights, even, odd, order, *, c=13.5):
        if isinstance(order, bool) or int(order) != order or order < 0:
            raise ValueError("nome order must be a nonnegative integer")
        self.order = int(order)
        self.h = complex(h)
        self.weights = tuple(map(complex, effective_weights))
        if len(self.weights) != 4:
            raise ValueError("four effective component dimensions are required")
        self.kappa = (complex(c)-1.5)/24
        self.theta_exponent = 12*self.kappa-4*sum(self.weights)
        self.exponent = self.h-self.weights[0]-self.weights[1]
        self.input_coefficients = tuple(np.asarray(a, dtype=DTYPE) for a in (even, odd))
        if any(len(a) < self.order+1 for a in self.input_coefficients):
            raise ValueError("insufficient c-recursion coefficients for the requested nome order")
        self._local_cache = {}
        z, u, theta = lambda_series(self.order)
        one_minus_z = -z.copy()
        one_minus_z[0] += 1
        length = self.order+1
        common = multiply(power(one_minus_z, self.weights[1]+self.weights[2]-self.kappa, length),
                          power(theta, -self.theta_exponent, length), length)
        self.coefficients = tuple(
            4**beta * multiply(multiply(common, power(u, self.h-self.kappa+beta/2, length), length),
                               compose(a[:length], z, length), length)
            for beta, a in enumerate(self.input_coefficients)
        )
        if any(np.any(~np.isfinite(a)) for a in self.coefficients):
            raise ArithmeticError("nonfinite elliptic coefficients")
        for a in self.coefficients:
            a.flags.writeable = False

    def _order(self, order):
        order = self.order if order is None else order
        if isinstance(order, bool) or int(order) != order or not 0 <= order <= self.order:
            raise ValueError("requested nome order exceeds the computed coefficients")
        return int(order)

    def value(self, z, beta, order=None, *, geometry=None, derivative=False):
        order = self._order(order)
        if beta not in (0, 1):
            raise ValueError("beta must be 0 or 1")
        z = np.asarray(z, dtype=complex)
        q, theta, theta_prime = nome_geometry(z) if geometry is None else geometry
        d1, d2, d3, _ = self.weights
        one_minus_z = np.asarray(1-z,dtype=complex)
        one_minus_z.imag = -z.imag  # preserve the reflected cut lip at signed zero
        prefactor = np.exp((self.h-self.kappa)*np.log(16*q)
            + (self.kappa-d1-d2)*np.log(z) + (self.kappa-d2-d3)*np.log(one_minus_z)
            + self.theta_exponent*np.log(theta) + (beta/2)*np.log(q))
        coefficients = self.coefficients[beta][:order+1]
        polynomial = np.polynomial.polynomial.polyval(q, coefficients)
        if derivative:
            q_log_prime = 1/(z*(1-z)*theta**4)
            log_prime = ((self.h-self.kappa+beta/2)*q_log_prime
                         + (self.kappa-d1-d2)/z - (self.kappa-d2-d3)/(1-z)
                         + self.theta_exponent*theta_prime/theta*q*q_log_prime)
            polynomial_prime = np.polynomial.polynomial.polyval(q, np.arange(1, len(coefficients))*coefficients[1:]) if order else 0
            polynomial = log_prime*polynomial + polynomial_prime*q*q_log_prime
        return np.asarray(prefactor*polynomial, dtype=complex)

    def local_data(self, beta, order=None, total_order=None):
        """z**exponent times Taylor polynomial of the truncated-nome block."""
        order = self._order(order)
        total_order = order if total_order is None else total_order
        if beta not in (0, 1) or int(total_order) != total_order or total_order < 0:
            raise ValueError("invalid parity or local Taylor order")
        total_order = int(total_order)
        key = beta, order, total_order
        if key not in self._local_cache:
            length = total_order+1
            q, v, theta = inverse_lambda_series(total_order)
            one_minus_z = np.zeros(length, dtype=DTYPE)
            one_minus_z[0] = 1
            if total_order:
                one_minus_z[1] = -1
            normalized = multiply(power(v, self.h-self.kappa+beta/2, length),
                power(one_minus_z, self.kappa-self.weights[1]-self.weights[2], length), length)
            normalized = multiply(normalized, power(theta, self.theta_exponent, length), length)/4**beta
            result = multiply(normalized, compose(self.coefficients[beta][:order+1], q, length), length)
            # The triangular transformation is an identity through the known
            # sewing order. Use those coefficients to avoid round-trip loss.
            known = min(order, total_order)+1
            result[:known] = self.input_coefficients[beta][:known]
            result = np.asarray(result, dtype=complex)
            if np.any(~np.isfinite(result)):
                raise ArithmeticError("nonfinite local elliptic-block expansion")
            result.flags.writeable = False
            self._local_cache[key] = result
        return self.exponent+beta/2, self._local_cache[key]
