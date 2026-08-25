"""Two-component free-Majorana NS--R--R chiral vertex.

The SCblock Ramond ground basis and the free-Fock ground basis are related by

    |+>_F = w^+,             |->_F = C w^-,
    C = -(1-i)/sqrt(2).

Consequently neither fixed SCblock ``eta`` structure is the canonical Ising
functional by itself.  The canonical Ising vertex is one ordinary/bordered
Pfaffian.  The endpoint map ``Z=diag(1,-1)`` below records the exact sign of
the *raw chi-path coefficients* of a negative consecutive string.  It is not
the free-field reflection operator and cannot turn that path identity into an
identity of abstract SCA states or signed Coulomb vertices.  The latter still
requires the reflected boson--fermion vertex.
"""

from __future__ import annotations

from functools import lru_cache

import sympy as sp

from .core import pfaffian


I = sp.I
SQRT2 = sp.sqrt(2)
EIGHTH_MINUS = (1 - I) / SQRT2
FOCK_MINUS = -EIGHTH_MINUS
FLIP = sp.Matrix(((0, 1), (1, 0)))
GROUND_Z = sp.diag(1, -1)


@lru_cache(None)
def scblock_fock_ground_matrix(form_parity, eta):
    """``D Gamma_f^eta D`` in the free-Fock Ramond ground basis."""

    form_parity = int(form_parity)
    eta = int(eta)
    if form_parity not in (0, 1) or eta not in (-1, 1):
        raise ValueError("form_parity must be 0 or 1 and eta must be +/-1")
    diagonal = sp.diag(1, FOCK_MINUS)
    if form_parity == 0:
        gamma = sp.diag(1, eta)
    else:
        gamma = sp.Matrix(((0, 1), (I * eta, 0)))
    return sp.simplify(diagonal * gamma * diagonal)


@lru_cache(None)
def canonical_ground_matrix(form_parity):
    """Canonical Ising matrices ``K`` and ``J`` in the same Fock basis."""

    if int(form_parity) == 0:
        return sp.diag(1, -1)
    if int(form_parity) == 1:
        return sp.Matrix(((0, 1), (-1, 0)))
    raise ValueError("form_parity must be 0 or 1")


def _fields(first_modes, second_modes, third_modes):
    # BPZ reverses the first string.  The extra minus sign per first-leg
    # creator is applied once, outside the Pfaffian.
    return (
        tuple((1, sp.Rational(mode)) for mode in reversed(tuple(first_modes)))
        + tuple((2, sp.Rational(mode)) for mode in tuple(second_modes))
        + tuple((3, sp.Rational(mode)) for mode in tuple(third_modes))
    )


def _series_coefficient(expression, variable, power):
    """Exact Taylor/Laurent coefficient with an order chosen from ``power``."""

    power = int(power)
    expanded = sp.series(
        expression, variable, 0, max(4, power + 2)
    ).removeO().expand()
    return sp.expand(expanded).coeff(variable, power)


def _local_data(leg, variable):
    leg = int(leg)
    if leg == 3:
        return (
            variable**2,
            variable / sp.sqrt(1 - variable**2),
            1 / (SQRT2 * variable * sp.sqrt(1 - variable**2)),
        )
    if leg == 2:
        return (
            1 + variable**2,
            -I * sp.sqrt(1 + variable**2) / variable,
            -I / (SQRT2 * variable * sp.sqrt(1 + variable**2)),
        )
    if leg == 1:
        return (
            1 / variable**2,
            -I / sp.sqrt(1 - variable**2),
            -I * variable**2 / (SQRT2 * sp.sqrt(1 - variable**2)),
        )
    raise ValueError(leg)


def _local_power(leg, mode):
    return int(2 * sp.Rational(mode) + (1 if int(leg) == 1 else -1))


def _spin_frame(leg):
    return {1: -sp.Integer(1), 2: I, 3: sp.Integer(1)}[int(leg)]


@lru_cache(None)
def _one_coefficient(leg, mode):
    """Closed local coefficient of the one-fermion spin OPE.

    If ``C_m=binomial(2m,m)/4^m``, the three framed coefficients are

    ``i C_(r-1/2)/sqrt(2)``, ``(-1)^m C_m/sqrt(2)``, and
    ``C_m/sqrt(2)``

    at infinity, one, and zero respectively.  This avoids a symbolic
    series expansion for every odd Pfaffian row.
    """

    leg = int(leg)
    mode = sp.Rational(mode)
    index = mode - sp.Rational(1, 2) if leg == 1 else mode
    if not index.is_integer or index < 0:
        raise ValueError((leg, mode))
    index = int(index)
    central = sp.binomial(2 * index, index) / 4**index
    if leg == 1:
        return sp.factor(I * central / SQRT2)
    if leg == 2:
        return sp.factor((-1) ** index * central / SQRT2)
    if leg == 3:
        return sp.factor(central / SQRT2)
    raise ValueError(leg)


@lru_cache(None)
def _pair_coefficient(leg_a, mode_a, leg_b, mode_b):
    """Arbitrary-level coefficient of the ordered two-spin kernel."""

    leg_a, leg_b = int(leg_a), int(leg_b)
    mode_a, mode_b = sp.Rational(mode_a), sp.Rational(mode_b)
    y, x, ratio = sp.symbols("native_y native_x native_ratio")
    z_a, square_a, _ = _local_data(leg_a, y)
    z_b, square_b, _ = _local_data(leg_b, x)
    kernel = (square_a / square_b + square_b / square_a) / (
        2 * (z_a - z_b)
    )
    power_a = _local_power(leg_a, mode_a)
    power_b = _local_power(leg_b, mode_b)
    if leg_a == leg_b == 1:
        nested = _series_coefficient(
            sp.simplify(kernel.subs(y, ratio * x)), ratio, power_a
        )
        value = _series_coefficient(nested, x, power_a + power_b)
    elif leg_a == leg_b:
        nested = _series_coefficient(
            sp.simplify(kernel.subs(x, ratio * y)), ratio, power_b
        )
        value = _series_coefficient(nested, y, power_a + power_b)
    else:
        value = _series_coefficient(
            _series_coefficient(kernel, y, power_a), x, power_b
        )
    orientation = -1 if (leg_a, leg_b) in ((1, 2), (2, 3)) else 1
    return sp.simplify(
        _spin_frame(leg_a) * _spin_frame(leg_b) * orientation * value
    )


def _gaussian_value(
    fields,
    ground2,
    ground3,
    even_matrix,
    pair_phase,
    odd_matrix,
    bpz_sign,
):
    """One ground-resolved Gaussian form as an ordinary/bordered Pfaffian."""

    size = len(fields)
    covariance = [[sp.Integer(0) for _ in range(size)] for _ in range(size)]
    for left, (leg_left, mode_left) in enumerate(fields):
        for right in range(left + 1, size):
            leg_right, mode_right = fields[right]
            value = _pair_coefficient(
                leg_left, mode_left, leg_right, mode_right
            )
            value *= pair_phase(leg_left) * pair_phase(leg_right)
            covariance[left][right] = sp.factor(value)
            covariance[right][left] = -covariance[left][right]

    if size % 2 == 0:
        return sp.factor(
            bpz_sign * even_matrix[int(ground2), int(ground3)]
            * pfaffian(covariance)
        )

    mean = []
    for leg, mode in fields:
        matrix = odd_matrix(leg)
        mean.append(
            sp.factor(
                _one_coefficient(leg, mode)
                * matrix[int(ground2), int(ground3)]
            )
        )
    augmented = [row + [mean[index]] for index, row in enumerate(covariance)]
    augmented.append([-value for value in mean] + [sp.Integer(0)])
    return sp.factor(bpz_sign * pfaffian(augmented))


@lru_cache(None)
def canonical_ising_value(
    form_parity,
    first_modes,
    second_modes,
    second_ground,
    third_modes,
    third_ground,
):
    """Canonical Ising NS--R--R form, evaluated by one Pfaffian."""

    form_parity = int(form_parity)
    first_modes = tuple(first_modes)
    fields = _fields(first_modes, second_modes, third_modes)
    matrices = {parity: canonical_ground_matrix(parity) for parity in (0, 1)}
    even = matrices[form_parity]

    def odd_matrix(leg):
        leg = int(leg)
        if leg == 1:
            return matrices[1 - form_parity]
        if leg == 2:
            return FLIP * even
        if leg == 3:
            return even * FLIP
        raise ValueError(leg)

    return _gaussian_value(
        fields,
        int(second_ground),
        int(third_ground),
        even,
        lambda leg: sp.Integer(1),
        odd_matrix,
        (-1) ** len(first_modes),
    )


@lru_cache(None)
def z_boundary_value(
    form_parity,
    first_modes,
    second_modes,
    second_ground,
    third_modes,
    third_ground,
):
    """Canonical path functional with the raw endpoint ``Z`` sign.

    This is a ground/path bookkeeping operation only.  It acts on the
    Ramond ground label, not on the nonzero-mode occupation number, and is
    neither a reflected SCA state nor a second chiral Majorana vertex.
    """

    return sp.factor(
        (-1) ** int(third_ground)
        * canonical_ising_value(
            int(form_parity),
            tuple(first_modes),
            tuple(second_modes),
            int(second_ground),
            tuple(third_modes),
            int(third_ground),
        )
    )


def ground_resolution_coefficients(form_parity, eta):
    """Zero-mode coefficients on ``(canonical, canonical-with-Z)``.

    These coefficients reconstruct ``D Gamma_f^eta D`` at level zero only.
    Applying them to excited Pfaffians is incorrect.
    """

    form_parity = int(form_parity)
    eta = int(eta)
    if eta not in (-1, 1):
        raise ValueError("eta must be +/-1")
    if form_parity == 0:
        return (1 + I * eta) / 2, (1 - I * eta) / 2
    if form_parity == 1:
        return (
            FOCK_MINUS * (1 - I * eta) / 2,
            -FOCK_MINUS * (1 + I * eta) / 2,
        )
    raise ValueError("form_parity must be 0 or 1")


def resolved_ground_value(
    form_parity,
    eta,
    second_ground,
    third_ground,
):
    """Ground-level check of the two-chart resolution."""

    ground = canonical_ground_matrix(form_parity)
    ordinary = ground[int(second_ground), int(third_ground)]
    signed = (ground * GROUND_Z)[int(second_ground), int(third_ground)]
    first, second = ground_resolution_coefficients(form_parity, eta)
    return sp.factor(sp.cancel(first * ordinary + second * signed))


def check_ground_change():
    """Verify the two exact change-of-basis identities entry by entry."""

    for form_parity in (0, 1):
        ground = canonical_ground_matrix(form_parity)
        signed = ground * GROUND_Z
        for eta in (1, -1):
            first, second = ground_resolution_coefficients(form_parity, eta)
            residual = scblock_fock_ground_matrix(
                form_parity, eta
            ) - first * ground - second * signed
            if any(sp.simplify(value) != 0 for value in residual):
                raise AssertionError((form_parity, eta, residual))


__all__ = (
    "canonical_ising_value",
    "check_ground_change",
    "ground_resolution_coefficients",
    "resolved_ground_value",
    "scblock_fock_ground_matrix",
    "z_boundary_value",
)
