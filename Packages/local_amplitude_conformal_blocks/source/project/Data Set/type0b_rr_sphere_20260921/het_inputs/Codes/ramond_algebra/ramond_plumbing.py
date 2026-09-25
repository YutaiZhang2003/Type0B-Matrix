#!/usr/bin/env python3
"""Exact Ramond sewing kernels with the full same-level ``G_0`` action.

Xi Yin's Ramond matter sewing operator is

    q**L_0 (1 - nu q**(-1/2) G_0).

The normalized block strips the primary power ``q**h``.  At descendant
level ``N`` this module therefore returns two matrix coefficients:

    q**N K_even + nu q**(N-1/2) K_odd.

The odd variable is not represented as a commuting SymPy symbol.  Its
coefficient is returned separately, so Koszul ordering remains explicit.

For a lift sign ``eta``, this module fixes the operator order

    eta**F (1 - nu q**(-1/2) G_0).

Reversing ``eta**F`` and the parenthesis sends ``nu`` to ``eta*nu``.
All matrices use the polynomial Ramond ground doublet implemented by
``ramond_sca.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from .ramond_sca import G, PBWState, act_mode, gram_matrix, pbw_basis


@dataclass(frozen=True)
class RamondPlumbingKernels:
    """Exact level data for one oriented Ramond plumbing edge.

    ``even_kernel`` multiplies the coefficient of one and ``odd_kernel``
    multiplies the odd modulus before the appropriate powers of ``q`` are
    attached.
    """

    twice_descendant_level: int
    lift_sign: int
    basis: tuple[PBWState, ...]
    gram: sp.Matrix
    inverse_gram: sp.Matrix
    zero_mode: sp.Matrix
    parity_lift: sp.Matrix
    even_kernel: sp.Matrix
    odd_kernel: sp.Matrix


def _validate_twice_level(twice_descendant_level: int) -> None:
    """Require a nonnegative even twice-level in the Ramond sector."""

    if not isinstance(twice_descendant_level, int):
        raise TypeError("twice_descendant_level must be an integer")
    if twice_descendant_level < 0:
        raise ValueError("twice_descendant_level must be nonnegative")
    if twice_descendant_level % 2:
        raise ValueError("a Ramond descendant has an even twice-level")


def _validate_lift_sign(lift_sign: int) -> None:
    """Require the spin lift to be exactly ``+1`` or ``-1``."""

    if lift_sign not in (-1, 1):
        raise ValueError("lift_sign must be +1 or -1")


def zero_mode_matrix(
    twice_descendant_level: int,
    *,
    h: sp.Expr,
    c: sp.Expr,
) -> tuple[tuple[PBWState, ...], sp.Matrix]:
    """Return the full matrix of ``G_0`` on one descendant level.

    Columns label input states and rows label output states in the returned
    full two-parity PBW basis.
    """

    _validate_twice_level(twice_descendant_level)
    h = sp.sympify(h)
    c = sp.sympify(c)
    basis = tuple(pbw_basis(twice_descendant_level))
    positions = {basis_state: index for index, basis_state in enumerate(basis)}
    matrix = sp.zeros(len(basis))
    for column, basis_state in enumerate(basis):
        acted = act_mode(
            G(0),
            {basis_state: sp.S.One},
            h=h,
            c=c,
        )
        for output_state, coefficient in acted.items():
            if output_state not in positions:
                raise AssertionError("G_0 left its Ramond descendant level")
            matrix[positions[output_state], column] += coefficient
    return basis, matrix.applyfunc(sp.expand)


def parity_lift_matrix(
    twice_descendant_level: int,
    *,
    lift_sign: int,
) -> tuple[tuple[PBWState, ...], sp.Matrix]:
    """Return ``eta**F`` on the full same-level PBW basis."""

    _validate_twice_level(twice_descendant_level)
    _validate_lift_sign(lift_sign)
    basis = tuple(pbw_basis(twice_descendant_level))
    diagonal = [
        sp.Integer(lift_sign) ** basis_state.parity for basis_state in basis
    ]
    return basis, sp.diag(*diagonal)


def ramond_plumbing_kernels(
    twice_descendant_level: int,
    *,
    h: sp.Expr,
    c: sp.Expr,
    lift_sign: int = 1,
) -> RamondPlumbingKernels:
    """Construct the inverse-Gram Ramond sewing kernels at one level.

    With the fixed lift ordering, the returned matrices are

    ``K_even = eta**F B**(-1)``,

    ``K_odd = -eta**F G_0 B**(-1)``.
    """

    _validate_twice_level(twice_descendant_level)
    _validate_lift_sign(lift_sign)
    h = sp.sympify(h)
    c = sp.sympify(c)

    gram_basis, gram = gram_matrix(
        twice_descendant_level,
        h=h,
        c=c,
    )
    basis, zero_mode = zero_mode_matrix(
        twice_descendant_level,
        h=h,
        c=c,
    )
    lift_basis, parity_lift = parity_lift_matrix(
        twice_descendant_level,
        lift_sign=lift_sign,
    )
    if tuple(gram_basis) != basis or lift_basis != basis:
        raise AssertionError("Ramond basis order changed between sewing data")

    inverse_gram = gram.inv().applyfunc(
        lambda entry: sp.factor(sp.cancel(entry))
    )
    even_kernel = (parity_lift * inverse_gram).applyfunc(
        lambda entry: sp.factor(sp.cancel(entry))
    )
    odd_kernel = (-parity_lift * zero_mode * inverse_gram).applyfunc(
        lambda entry: sp.factor(sp.cancel(entry))
    )
    return RamondPlumbingKernels(
        twice_descendant_level=twice_descendant_level,
        lift_sign=lift_sign,
        basis=basis,
        gram=gram,
        inverse_gram=inverse_gram,
        zero_mode=zero_mode,
        parity_lift=parity_lift,
        even_kernel=even_kernel,
        odd_kernel=odd_kernel,
    )


def normalized_ramond_edge_coefficients(
    twice_descendant_level: int,
    *,
    q: sp.Expr,
    h: sp.Expr,
    c: sp.Expr,
    lift_sign: int = 1,
) -> tuple[sp.Matrix, sp.Matrix]:
    """Return the coefficients of one and ``nu`` after stripping ``q**h``.

    The two outputs are

    ``q**N K_even`` and ``q**(N-1/2) K_odd``

    at physical descendant level ``N``.
    """

    kernels = ramond_plumbing_kernels(
        twice_descendant_level,
        h=h,
        c=c,
        lift_sign=lift_sign,
    )
    q = sp.sympify(q)
    level = sp.Rational(twice_descendant_level, 2)
    even = (q**level * kernels.even_kernel).applyfunc(
        lambda entry: sp.factor(sp.cancel(entry))
    )
    odd = (q ** (level - sp.Rational(1, 2)) * kernels.odd_kernel).applyfunc(
        lambda entry: sp.factor(sp.cancel(entry))
    )
    return even, odd


def reversed_lift_order_odd_kernel(
    twice_descendant_level: int,
    *,
    h: sp.Expr,
    c: sp.Expr,
    lift_sign: int = 1,
) -> sp.Matrix:
    """Return the odd coefficient if ``G_0`` precedes ``eta**F``.

    This comparison helper implements

    ``-G_0 eta**F B**(-1)``.

    It equals ``lift_sign`` times the convention-fixed odd kernel.
    """

    data = ramond_plumbing_kernels(
        twice_descendant_level,
        h=h,
        c=c,
        lift_sign=lift_sign,
    )
    return (-data.zero_mode * data.parity_lift * data.inverse_gram).applyfunc(
        lambda entry: sp.factor(sp.cancel(entry))
    )


__all__ = [
    "RamondPlumbingKernels",
    "normalized_ramond_edge_coefficients",
    "parity_lift_matrix",
    "ramond_plumbing_kernels",
    "reversed_lift_order_odd_kernel",
    "zero_mode_matrix",
]
