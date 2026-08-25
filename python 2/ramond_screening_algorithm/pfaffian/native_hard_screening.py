#!/usr/bin/env python3
"""Genuine Coulomb-screening values for ``(0,3/4,3/4)``.

This module is the first end-to-end native node evaluator in the Ramond
screening project.  It uses only

* the literal 2016 consecutive ``chi`` strings;
* the canonical auxiliary Ising Pfaffian;
* the physical two-spin Majorana Pfaffian in the free-Fock ground frame;
* the bosonic Coulomb weight on a charge-neutral screening plane; and
* an exact Jack/Kadell Selberg average.

It does not construct a superconformal descendant, import a Ward value, or
read either hard polynomial ``K`` or ``H``.  Ward data are used only by the
separate audit module.

For ``N`` screenings the physical free-Fock ground matrix is

    D Gamma_f^eta D X^N,

where ``D=diag(1,-(1-i)/sqrt(2))`` and ``X`` is the Ramond ground flip.
The factor ``X^N`` is not optional: it is the screening holonomy which makes
the screened primary form equal to the requested SCblock form.  Omitting it
makes the odd-screening hard node vanish in the wrong channel.

The optional right-endpoint ``Z`` applies the exact sign relation between
the *raw chi-path coefficients* of the positive and negative strings.  It is
kept only as a diagnostic.  The 2013 reflected fermion mixes physical boson
and fermion oscillators, so this endpoint sign is not a signed SCA-state or
signed-Coulomb callback.  See
``reflection.audit_signed_state_obstruction`` for the first exact mismatch.
"""

from __future__ import annotations

from functools import lru_cache

import sympy as sp

from python.nsrr_chi_branching.nsrr_chi_formula import ramond_fock_paths

from .core import pfaffian, pfaffian_recursive
from .native_spin_kernel import (
    FLIP,
    _one_coefficient,
    _pair_coefficient,
    canonical_ising_value,
    scblock_fock_ground_matrix,
)
from .screening_pfaffian import (
    _row_pair_with_screening,
    _screen_kernel,
    vandermonde,
)
from .selberg_jack import normalized_selberg_average
from .special_oracle import (
    ordinary_selberg,
    physical_nsrr_selberg,
)


I = sp.I
SQRT2 = sp.sqrt(2)
HARD = sp.Rational(3, 4)


def _external(leg, mode):
    return ("external", int(leg), sp.Rational(mode))


def _screening(coordinate):
    return ("screening", 0, sp.sympify(coordinate))


def _pair(left, right):
    kind_left, leg_left, value_left = left
    kind_right, leg_right, value_right = right
    if kind_left == kind_right == "screening":
        return _screen_kernel(value_left, value_right)
    if kind_left == "external" and kind_right == "screening":
        return _row_pair_with_screening(
            {2: "one", 3: "zero"}[leg_left], value_left, value_right
        )
    if kind_left == "screening" and kind_right == "external":
        return _row_pair_with_screening(
            {2: "one", 3: "zero"}[leg_right], value_right, value_left
        )
    return _pair_coefficient(
        leg_left, value_left, leg_right, value_right
    )


def _pfaffian_of(objects):
    objects = tuple(objects)
    matrix = [[sp.Integer(0) for _ in objects] for _ in objects]
    for left in range(len(objects)):
        for right in range(left + 1, len(objects)):
            value = _pair(objects[left], objects[right])
            matrix[left][right] = value
            matrix[right][left] = -value
    evaluator = pfaffian_recursive if len(objects) <= 10 else pfaffian
    return evaluator(matrix)


def _physical_sector_value(
    second_modes,
    second_ground,
    third_modes,
    third_ground,
    ground_matrix,
    screenings,
):
    """One physical ground sector as an ordinary/bordered Pfaffian."""

    objects = (
        tuple(_external(2, mode) for mode in second_modes)
        + tuple(_screening(value) for value in screenings)
        + tuple(_external(3, mode) for mode in third_modes)
    )
    second_ground = int(second_ground)
    third_ground = int(third_ground)
    ground_matrix = sp.Matrix(ground_matrix)
    if len(objects) % 2 == 0:
        return ground_matrix[second_ground, third_ground] * _pfaffian_of(objects)

    answer = sp.Integer(0)
    for position, (kind, leg, value) in enumerate(objects):
        if kind == "external" and leg == 2:
            source_matrix = FLIP * ground_matrix
            one = _one_coefficient(2, value)
        elif kind == "external":
            source_matrix = ground_matrix * FLIP
            one = _one_coefficient(3, value)
        else:
            # A screening is radially between the one and zero punctures,
            # so its odd source acts on the right Ramond ground index.
            source_matrix = ground_matrix * FLIP
            one = 1 / SQRT2
        remaining = objects[:position] + objects[position + 1 :]
        answer += (
            (-1) ** position
            * one
            * source_matrix[second_ground, third_ground]
            * _pfaffian_of(remaining)
        )
    return answer


@lru_cache(None)
def hard_chi_integrand(
    screenings,
    form_parity,
    eta,
    right_endpoint_z=False,
):
    """Return the exact fermionic hard integrand.

    ``screenings`` is a tuple of coordinates.  The two hard Ramond copies
    are the natural ``epsilon_2=epsilon_3=0`` chains.  The auxiliary form
    parity is consequently equal to ``form_parity``.
    """

    screenings = tuple(map(sp.sympify, screenings))
    form_parity = int(form_parity)
    eta = int(eta)
    if form_parity not in (0, 1) or eta not in (-1, 1):
        raise ValueError("form_parity must be 0 or 1 and eta must be +/-1")
    physical_ground = scblock_fock_ground_matrix(form_parity, eta) * (
        FLIP ** (len(screenings) % 2)
    )
    answer = sp.Integer(0)
    for state2, coefficient2 in ramond_fock_paths(HARD, 0):
        auxiliary2, auxiliary_ground2, physical2, physical_ground2 = state2
        for state3, coefficient3 in ramond_fock_paths(HARD, 0):
            auxiliary3, auxiliary_ground3, physical3, physical_ground3 = state3
            auxiliary = canonical_ising_value(
                form_parity,
                (),
                auxiliary2,
                auxiliary_ground2,
                auxiliary3,
                auxiliary_ground3,
            )
            physical = _physical_sector_value(
                physical2,
                physical_ground2,
                physical3,
                physical_ground3,
                physical_ground,
                screenings,
            )
            physical_parity2 = (len(physical2) + physical_ground2) % 2
            auxiliary_parity3 = (len(auxiliary3) + auxiliary_ground3) % 2
            koszul = (-1) ** (physical_parity2 * auxiliary_parity3)
            endpoint = (
                (-1) ** int(physical_ground3)
                if bool(right_endpoint_z)
                else 1
            )
            answer += (
                coefficient2
                * coefficient3
                * koszul
                * endpoint
                * auxiliary
                * physical
            )
    # Factoring the multivariate rational expression here is dramatically
    # slower than clearing its known endpoint poles in
    # ``hard_contour_polynomial``.  Keep the exact sum unexpanded.
    return answer


def hard_contour_polynomial(
    screenings,
    form_parity,
    eta,
    right_endpoint_z=False,
):
    """Return ``Delta`` times the denominator-cleared native integrand."""

    screenings = int(screenings)
    if screenings < 0:
        raise ValueError("the screening number must be nonnegative")
    coordinates = sp.symbols(f"t0:{screenings}")
    correlator = hard_chi_integrand(
        coordinates, int(form_parity), int(eta), bool(right_endpoint_z)
    )
    # For n2=n3=3/4, both endpoint Laurent shifts are one.
    clearing = sp.prod(value * (1 - value) for value in coordinates)
    polynomial = sp.cancel(
        vandermonde(coordinates) * clearing * sp.together(correlator)
    )
    numerator, denominator = sp.fraction(polynomial)
    if set(coordinates) & denominator.free_symbols:
        raise AssertionError(
            f"the native hard integrand did not polynomialize: {denominator}"
        )
    return coordinates, sp.expand(numerator / denominator)


def hard_neutrality_momentum(
    b,
    second_momentum,
    third_momentum,
    screenings,
):
    """The first momentum on the displayed Coulomb charge plane."""

    b = sp.sympify(b)
    q = b + 1 / b
    return sp.factor(
        -q / 2
        - sp.sympify(second_momentum)
        - sp.sympify(third_momentum)
        - int(screenings) * b
    )


def hard_screening_value(
    screenings,
    form_parity,
    eta,
    b,
    second_momentum,
    third_momentum,
    *,
    right_endpoint_z=False,
):
    """Evaluate one genuine hard Coulomb node exactly.

    ``second_momentum`` and ``third_momentum`` are the charges used in the
    Coulomb weight.  They may therefore carry the reflected signs of a
    diagnostic raw-path chart.  ``right_endpoint_z=True`` does not implement
    the reflected SCA vertex; it is controlled separately so the failed
    endpoint-only proposal can be audited without contaminating the genuine
    positive-chart evaluator.
    """

    screenings = int(screenings)
    b = sp.sympify(b)
    second_momentum = sp.sympify(second_momentum)
    third_momentum = sp.sympify(third_momentum)
    q = sp.cancel(b + 1 / b)
    coordinates, polynomial = hard_contour_polynomial(
        screenings,
        int(form_parity),
        int(eta),
        bool(right_endpoint_z),
    )
    if screenings == 0:
        return sp.factor(polynomial)

    A = -b * (q / 2 + third_momentum) - sp.Rational(1, 2)
    B = -b * (q / 2 + second_momentum) - sp.Rational(1, 2)
    g = -b * q / 2
    numerator = normalized_selberg_average(
        polynomial, coordinates, A - 1, B - 1, g
    )
    numerator *= ordinary_selberg(
        screenings, A - 1, B - 1, g
    )
    denominator = physical_nsrr_selberg(screenings, A, B, g)
    # The BFL order-field polynomial carries sqrt(2) more than the
    # rationalized Majorana Pfaffian for odd screening number.
    if screenings % 2:
        denominator /= SQRT2
    return sp.factor(
        sp.powsimp(
            sp.cancel(sp.expand_func(numerator / denominator)), force=True
        )
    )


__all__ = (
    "HARD",
    "hard_chi_integrand",
    "hard_contour_polynomial",
    "hard_neutrality_momentum",
    "hard_screening_value",
)
