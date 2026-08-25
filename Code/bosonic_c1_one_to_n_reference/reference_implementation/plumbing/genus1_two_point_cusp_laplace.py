#!/usr/bin/env python3
"""Watson/Laplace expansion of the genus-one two-point necklace integral.

The expansion is intended for the torus cusp with both necklace propagation
multipliers small.  It keeps the exact b=1 BRY/DOZZ dependence on the
external momentum, expands the two internal momenta about zero, and performs
their Gaussian moments analytically.  Descendant-block corrections are not
included; their omission is separately controlled by the two necklace nomes.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping, Sequence

import mpmath as mp
import numpy as np


Exponent = tuple[int, int]
Polynomial = dict[Exponent, complex]


@dataclass(frozen=True)
class NecklaceLaplaceEstimate:
    x: float
    z: complex
    tau: complex
    max_x_degree: int
    decay_coefficients: tuple[float, float]
    value: complex
    relative_step_from_previous: float | None
    maximum_primary_q_abs: float


def _add(first: Mapping[Exponent, complex], second: Mapping[Exponent, complex]) -> Polynomial:
    result = dict(first)
    for exponent, coefficient in second.items():
        result[exponent] = result.get(exponent, 0.0j) + complex(coefficient)
    return result


def _multiply(
    first: Mapping[Exponent, complex],
    second: Mapping[Exponent, complex],
    *,
    max_degree: int,
) -> Polynomial:
    result: Polynomial = {}
    for left, left_coefficient in first.items():
        for right, right_coefficient in second.items():
            exponent = (left[0] + right[0], left[1] + right[1])
            if sum(exponent) > max_degree:
                continue
            result[exponent] = (
                result.get(exponent, 0.0j)
                + complex(left_coefficient) * complex(right_coefficient)
            )
    return result


def _log_upsilon_one_shifted(momentum: complex) -> mp.mpc:
    value = mp.mpc(momentum)
    return mp.log(mp.barnesg(1 + 1j * value)) + mp.log(
        mp.barnesg(1 - 1j * value)
    )


def _origin_even_coefficient(degree: int) -> float:
    """Coefficient of y^(2*degree) in log Upsilon_1(1+i*y)."""

    if degree == 1:
        return float(1.0 + mp.euler)
    return float(
        (-1) ** (degree + 1)
        * mp.zeta(2 * degree - 1)
        / degree
    )


def _cauchy_taylor_coefficients(
    center: complex,
    *,
    maximum_order: int,
    radius: float,
    node_count: int,
    dps: int,
) -> tuple[complex, ...]:
    """Return Taylor coefficients of log Upsilon around a safe circle."""

    if maximum_order < 0 or node_count <= maximum_order:
        raise ValueError("Cauchy node count must exceed the requested Taylor order")
    if radius <= 0.0:
        raise ValueError("Cauchy radius must be positive")
    mp.mp.dps = int(dps)
    angles = 2.0 * math.pi * np.arange(node_count, dtype=float) / node_count
    values = np.asarray(
        [
            complex(
                _log_upsilon_one_shifted(
                    center + radius * complex(math.cos(angle), math.sin(angle))
                )
            )
            for angle in angles
        ],
        dtype=np.complex128,
    )
    # The logarithm is analytic inside the chosen zero-free circle.  Unwrap
    # its sampled branch before applying the discrete Cauchy transform.
    values = values.real + 1.0j * np.unwrap(values.imag)
    fourier = np.fft.fft(values) / node_count
    return tuple(
        complex(fourier[order] / radius**order)
        for order in range(maximum_order + 1)
    )


@lru_cache(maxsize=None)
def necklace_reduced_dozz_polynomial(
    x: float,
    max_x_degree: int,
    *,
    dps: int = 50,
    cauchy_nodes: int = 128,
) -> Polynomial:
    r"""Expand C(i*x/2,P1,P2)^2/(P1^2 P2^2) in x_i=P_i^2.

    If ``E=i*x/2`` and ``L(y)=log Upsilon_1(1+i*y)``, the four denominator
    terms combine into

    ``sum_{s1,s2=+-1} L(E+s1*P1+s2*P2)``.

    This makes the expansion even in each internal momentum.  Cauchy's
    formula evaluates the derivatives of ``L`` at ``E`` independently of the
    nonzero-node threshold quadrature used by the reference calculation.
    """

    x = float(x)
    max_x_degree = int(max_x_degree)
    if not 0.0 < x < 1.0:
        raise ValueError("the real-contour cusp expansion requires 0<x<1")
    if max_x_degree < 0:
        raise ValueError("max_x_degree must be nonnegative")
    mp.mp.dps = int(dps)
    external = 0.5j * x
    # The closest Barnes-G zero in the momentum plane is at distance
    # 1-x/2.  Staying at 65% of that distance is a stable Cauchy contour for
    # the degrees used in the comparison audit.
    radius = 0.65 * (1.0 - 0.5 * x)
    taylor_at_external = _cauchy_taylor_coefficients(
        external,
        maximum_order=2 * max_x_degree,
        radius=radius,
        node_count=int(cauchy_nodes),
        dps=int(dps),
    )
    reduced_at_origin = complex(
        64.0
        * external**2
        * mp.exp(
            2.0
            * (
                _log_upsilon_one_shifted(2.0 * external)
                - 4.0 * _log_upsilon_one_shifted(external)
            )
        )
    )
    logarithm: Polynomial = {}
    for total_degree in range(1, max_x_degree + 1):
        external_taylor = taylor_at_external[2 * total_degree]
        for first_degree in range(total_degree + 1):
            second_degree = total_degree - first_degree
            coefficient = (
                -8.0
                * external_taylor
                * math.factorial(2 * total_degree)
                / (
                    math.factorial(2 * first_degree)
                    * math.factorial(2 * second_degree)
                )
            )
            if second_degree == 0:
                coefficient += (
                    2.0
                    * 2.0 ** (2 * total_degree)
                    * _origin_even_coefficient(total_degree)
                )
            if first_degree == 0:
                coefficient += (
                    2.0
                    * 2.0 ** (2 * total_degree)
                    * _origin_even_coefficient(total_degree)
                )
            exponent = (first_degree, second_degree)
            logarithm[exponent] = logarithm.get(exponent, 0.0j) + coefficient

    exponential: Polynomial = {(0, 0): 1.0 + 0.0j}
    power: Polynomial = {(0, 0): 1.0 + 0.0j}
    for power_index in range(1, max_x_degree + 1):
        power = _multiply(power, logarithm, max_degree=max_x_degree)
        exponential = _add(
            exponential,
            {
                exponent: coefficient / math.factorial(power_index)
                for exponent, coefficient in power.items()
            },
        )
    return {
        exponent: reduced_at_origin * coefficient
        for exponent, coefficient in exponential.items()
    }


def _necklace_decay(z: complex, tau: complex) -> tuple[float, float, float]:
    z = complex(z)
    tau = complex(tau)
    if tau.imag <= 0.0:
        raise ValueError("tau must lie in the upper half-plane")
    first_log_abs = (1.0j * z).real
    second_log_abs = (1.0j * (2.0 * math.pi * tau - z)).real
    if first_log_abs >= 0.0 or second_log_abs >= 0.0:
        raise ValueError("both necklace propagation multipliers must lie inside the unit disc")
    return (
        -2.0 * first_log_abs,
        -2.0 * second_log_abs,
        max(math.exp(first_log_abs), math.exp(second_log_abs)),
    )


def necklace_laplace_value(
    x: float,
    z: complex,
    tau: complex,
    *,
    max_x_degree: int,
    dps: int = 50,
    cauchy_nodes: int = 128,
) -> complex:
    """Return the block-free two-momentum Watson expansion."""

    first_decay, second_decay, _ = _necklace_decay(z, tau)
    polynomial = necklace_reduced_dozz_polynomial(
        float(x),
        int(max_x_degree),
        dps=int(dps),
        cauchy_nodes=int(cauchy_nodes),
    )
    momentum_integral = 0.0 + 0.0j
    for (first_power, second_power), coefficient in polynomial.items():
        first_moment = (
            0.5
            * math.gamma(first_power + 1.5)
            * first_decay ** (-first_power - 1.5)
        )
        second_moment = (
            0.5
            * math.gamma(second_power + 1.5)
            * second_decay ** (-second_power - 1.5)
        )
        momentum_integral += coefficient * first_moment * second_moment
    first_log_abs = -0.5 * first_decay
    second_log_abs = -0.5 * second_decay
    threshold_casimir = math.exp(
        2.0
        * (1.0 - 25.0 / 24.0)
        * (first_log_abs + second_log_abs)
    )
    return complex(threshold_casimir * momentum_integral / math.pi**2)


def necklace_laplace_estimate(
    x: float,
    z: complex,
    tau: complex,
    *,
    max_x_degree: int,
    dps: int = 50,
    cauchy_nodes: int = 128,
) -> NecklaceLaplaceEstimate:
    """Return one truncation with an adjacent-degree asymptotic diagnostic."""

    first_decay, second_decay, q_max = _necklace_decay(z, tau)
    value = necklace_laplace_value(
        x,
        z,
        tau,
        max_x_degree=max_x_degree,
        dps=dps,
        cauchy_nodes=cauchy_nodes,
    )
    relative_step: float | None = None
    if int(max_x_degree) > 0:
        previous = necklace_laplace_value(
            x,
            z,
            tau,
            max_x_degree=int(max_x_degree) - 1,
            dps=dps,
            cauchy_nodes=cauchy_nodes,
        )
        scale = max(abs(value), abs(previous), 1.0e-300)
        relative_step = abs(value - previous) / scale
    return NecklaceLaplaceEstimate(
        x=float(x),
        z=complex(z),
        tau=complex(tau),
        max_x_degree=int(max_x_degree),
        decay_coefficients=(first_decay, second_decay),
        value=value,
        relative_step_from_previous=relative_step,
        maximum_primary_q_abs=q_max,
    )


__all__ = [
    "NecklaceLaplaceEstimate",
    "necklace_laplace_estimate",
    "necklace_laplace_value",
    "necklace_reduced_dozz_polynomial",
]
