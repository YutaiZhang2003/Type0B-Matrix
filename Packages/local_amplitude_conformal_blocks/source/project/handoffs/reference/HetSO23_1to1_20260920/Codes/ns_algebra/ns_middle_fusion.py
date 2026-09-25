#!/usr/bin/env python3
r"""Exact middle-slot NS singular-vector restrictions.

The fusion polynomials in :mod:`ns_fusion` give a singular-vector
three-point matrix element when the singular state is in the first slot,
and reflection gives the corresponding third-slot element.  A singular
state in the middle slot is not obtained by cyclically reusing either
formula.  This module evaluates it directly from the NS Ward identities.

The public quantity is

.. math::

   \rho\!\left(
      G_{-1/2}^{\alpha}\nu_{h_\infty},
      \chi_{r,s},
      \nu_{h_0}
   \right),

where ``chi_rs`` is normalized to have coefficient one in front of
``G_-1/2**(r*s) nu_h_rs``.  The central charge and singular primary
weight use Xi Yin's convention,

.. math::

   c=\frac32+3(b+b^{-1})^2,\qquad
   h_{r,s}=\frac{(b+b^{-1})^2-(rb+s/b)^2}{8}.

All calculations are exact SymPy calculations.  Non-global middle modes
are moved to the outer slots with equations (L2b) and (S2b) of
Hadasz--Jaskolski--Suchanek, hep-th/0611266.  The recursion terminates on
the exact symbolic version of the Belavin--Geiko global ``tau`` formula.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Mapping

import sympy as sp

from .ns_fusion import c_xi, h_rs
from .ns_sca import G, L, Mode, State, Word, act_mode, gram_matrix, twice_level


__all__ = ["middle_slot_restriction"]


_NULL_B = sp.Symbol("_b_null", nonzero=True)


def _validate_label(r: int, s: int) -> None:
    """Validate a Neveu--Schwarz Kac label."""

    if not isinstance(r, int) or not isinstance(s, int):
        raise TypeError("r and s must be integers")
    if r <= 0 or s <= 0:
        raise ValueError("r and s must be positive")
    if (r + s) % 2:
        raise ValueError("an NS Kac label must have r+s even")


def _validate_alpha(alpha: int) -> None:
    """Validate the parity-resolved trinion label."""

    if alpha not in (0, 1):
        raise ValueError("alpha must be 0 or 1")


def _falling(value: sp.Expr, order: int) -> sp.Expr:
    """Return an exact falling Pochhammer."""

    return sp.prod(value - offset for offset in range(order))


def _rising(value: sp.Expr, order: int) -> sp.Expr:
    """Return an exact rising Pochhammer."""

    return sp.prod(value + offset for offset in range(order))


def _tau_term_exact(
    k: int,
    m: int,
    p: int,
    alpha: tuple[int, int, int],
    d1: sp.Expr,
    d2: sp.Expr,
    d3: sp.Expr,
) -> sp.Expr:
    """Return one exact summand of the eight-case global ``tau`` formula."""

    choose_k = sp.binomial(k, p)
    choose_m = sp.binomial(m, p)
    half = sp.Rational(1, 2)

    if alpha == (0, 0, 0):
        return (
            choose_k
            * _falling(2 * d3 + m - 1, p)
            * _falling(sp.Integer(m), p)
            * _rising(d2 + d3 - d1, m - p)
            * _rising(d1 + d2 - d3 + p - m, k - p)
        )
    if alpha == (1, 0, 0):
        return (
            choose_m
            * _falling(2 * d3 + m, p)
            * _falling(sp.Integer(k), p)
            * _rising(d2 + d3 - d1 + half, m - p)
            * _rising(d1 + d2 - d3 + half - m + p, k - p)
        )
    if alpha == (0, 1, 0):
        return (
            choose_k
            * _falling(2 * d3 + m - 1, p)
            * _falling(sp.Integer(m), p)
            * _rising(d2 + d3 - d1 + half, m - p)
            * _rising(d1 + d2 - d3 + p - m + half, k - p)
        )
    if alpha == (0, 0, 1):
        return (
            choose_k
            * _falling(2 * d3 + m, p)
            * _falling(sp.Integer(m), p)
            * _rising(d2 + d3 - d1 + half, m - p)
            * _rising(d1 + d2 - d3 - half + p - m, k - p)
        )
    if alpha == (1, 1, 0):
        return (
            choose_k
            * _falling(2 * d3 + m - 1, p)
            * _falling(sp.Integer(m), p)
            * _rising(d2 + d3 - d1, m - p)
            * _rising(d1 + d2 - d3 + p - m + 1, k - p)
            * (d1 + d2 - d3)
        )
    if alpha == (1, 0, 1):
        return (
            choose_k
            * _falling(2 * d3 + m, p)
            * _falling(sp.Integer(m), p)
            * _rising(d2 + d3 - d1, m - p)
            * _rising(d1 + d2 - d3 + p - m, k - p)
            * (d1 - d2 + d3)
        )
    if alpha == (0, 1, 1):
        return (
            -choose_k
            * _falling(2 * d3 + m, p)
            * _falling(sp.Integer(m), p)
            * _rising(d2 + d3 - d1, m - p + 1)
            * _rising(d1 + d2 - d3 + p - m, k - p)
        )
    if alpha == (1, 1, 1):
        return (
            choose_k
            * _falling(2 * d3 + m, p)
            * _falling(sp.Integer(m), p)
            * _rising(d2 + d3 - d1 + half, m - p)
            * _rising(d1 + d2 - d3 + p - m + half, k - p)
            * (d1 + d2 + d3 - half)
        )
    raise AssertionError(f"unreachable parity tuple {alpha!r}")


def _global_coordinates(word: Word) -> tuple[int, int]:
    """Return ``(n,beta)`` for a global PBW word."""

    n = sum(mode == L(-1) for mode in word)
    beta = sum(mode == G(sp.Rational(-1, 2)) for mode in word)
    if len(word) != n + beta or beta not in (0, 1):
        raise ValueError("the Ward recursion did not terminate on a global word")
    return n, beta


def _rho_global_exact(
    word1: Word,
    word2: Word,
    word3: Word,
    d1: sp.Expr,
    d2: sp.Expr,
    d3: sp.Expr,
) -> sp.Expr:
    """Evaluate an exact global descendant three-point matrix element."""

    n1, beta1 = _global_coordinates(word1)
    n2, beta2 = _global_coordinates(word2)
    n3, beta3 = _global_coordinates(word3)
    parity = (beta1, beta2, beta3)
    if parity in ((1, 0, 0), (1, 1, 0)):
        # The published component-map kernels must be reflected before they
        # are used as fixed-parity trilinear forms.  This endpoint reflection
        # is the step that makes the odd-null factorization identities hold.
        reflected_parity = (beta3, beta2, beta1)
        reflection_sign = -sp.S.One if beta2 else sp.S.One
        tau_value = reflection_sign * sum(
            (
                _tau_term_exact(
                    n3,
                    n1,
                    p,
                    reflected_parity,
                    d3,
                    d2,
                    d1,
                )
                for p in range(min(n1, n3) + 1)
            ),
            sp.S.Zero,
        )
    else:
        tau_value = sum(
            (
                _tau_term_exact(n1, n3, p, parity, d1, d2, d3)
                for p in range(min(n1, n3) + 1)
            ),
            sp.S.Zero,
        )
    first_factor = (
        d1
        - d2
        - d3
        + sp.Rational(beta1 - beta2 - beta3, 2)
        + n1
        - n3
    )
    return sp.expand(_falling(first_factor, n2) * tau_value)


def _add_acted_terms(
    result: sp.Expr,
    coefficient: sp.Expr,
    acted: Mapping[Word, sp.Expr],
    evaluate,
    *,
    outer_slot: int,
) -> sp.Expr:
    """Accumulate one sparse outer-leg mode action into a Ward sum."""

    for word, state_coefficient in acted.items():
        if outer_slot == 1:
            term = evaluate(word1=word)
        elif outer_slot == 3:
            term = evaluate(word3=word)
        else:
            raise AssertionError("outer_slot must be 1 or 3")
        result += coefficient * state_coefficient * term
    return result


def _rho_middle_word(
    word1: Word,
    word2: Word,
    word3: Word,
    *,
    d1: sp.Expr,
    d2: sp.Expr,
    d3: sp.Expr,
    c: sp.Expr,
) -> sp.Expr:
    """Reduce one middle-slot PBW word with the exact HJS Ward identities."""

    @lru_cache(maxsize=None)
    def evaluate(word1: Word, word2: Word, word3: Word) -> sp.Expr:
        """Evaluate one cached triple of homogeneous descendant words."""

        global_modes = {L(-1), G(sp.Rational(-1, 2))}
        if all(mode in global_modes for mode in word2):
            return _rho_global_exact(word1, word2, word3, d1, d2, d3)

        leading = word2[0]
        rest = word2[1:]

        if leading == L(-1):
            exponent = (
                d1
                + sp.Rational(twice_level(word1), 2)
                - d2
                - sp.Rational(twice_level(rest), 2)
                - d3
                - sp.Rational(twice_level(word3), 2)
            )
            return sp.expand(exponent * evaluate(word1, rest, word3))

        maximum_m = max(twice_level(word1), twice_level(word3)) + 2
        value = sp.S.Zero

        if leading.kind == "L":
            n = -int(leading.index)
            if n <= 1:
                raise AssertionError("a non-global middle L mode must have n>1")
            for m in range(maximum_m + 1):
                ward_coefficient = sp.binomial(n - 2 + m, m)
                acted_first = act_mode(
                    L(n + m),
                    {word1: sp.S.One},
                    h=d1,
                    c=c,
                )
                value = _add_acted_terms(
                    value,
                    ward_coefficient,
                    acted_first,
                    lambda **changes: evaluate(
                        changes.get("word1", word1),
                        rest,
                        changes.get("word3", word3),
                    ),
                    outer_slot=1,
                )
                acted_third = act_mode(
                    L(m - 1),
                    {word3: sp.S.One},
                    h=d3,
                    c=c,
                )
                value = _add_acted_terms(
                    value,
                    (-1) ** n * ward_coefficient,
                    acted_third,
                    lambda **changes: evaluate(
                        changes.get("word1", word1),
                        rest,
                        changes.get("word3", word3),
                    ),
                    outer_slot=3,
                )
            return sp.expand(value)

        k = -leading.index
        if k <= sp.Rational(1, 2):
            raise AssertionError("a non-global middle G mode must have k>1/2")
        sign_exponent = (
            twice_level(word1)
            + twice_level(word3)
            + int(k + sp.Rational(1, 2))
        )
        second_sum_sign = (-1) ** sign_exponent
        for m in range(maximum_m + 1):
            ward_coefficient = sp.binomial(
                k - sp.Rational(3, 2) + m,
                m,
            )
            acted_first = act_mode(
                G(k + m),
                {word1: sp.S.One},
                h=d1,
                c=c,
            )
            value = _add_acted_terms(
                value,
                ward_coefficient,
                acted_first,
                lambda **changes: evaluate(
                    changes.get("word1", word1),
                    rest,
                    changes.get("word3", word3),
                ),
                outer_slot=1,
            )
            acted_third = act_mode(
                G(m - sp.Rational(1, 2)),
                {word3: sp.S.One},
                h=d3,
                c=c,
            )
            value = _add_acted_terms(
                value,
                second_sum_sign * ward_coefficient,
                acted_third,
                lambda **changes: evaluate(
                    changes.get("word1", word1),
                    rest,
                    changes.get("word3", word3),
                ),
                outer_slot=3,
            )
        return sp.expand(value)

    return evaluate(word1, word2, word3)


@lru_cache(maxsize=None)
def _normalized_null_state_template(
    r: int,
    s: int,
) -> tuple[tuple[Word, sp.Expr], ...]:
    """Return the normalized generic-``b`` singular vector as sparse data."""

    _validate_label(r, s)
    level = r * s
    singular_weight = h_rs(r, s, _NULL_B)
    central_charge = c_xi(_NULL_B)
    basis, matrix = gram_matrix(
        level,
        h=singular_weight,
        c=central_charge,
    )
    reduced_matrix = matrix.applyfunc(sp.factor)
    kernel = reduced_matrix.nullspace()
    if len(kernel) != 1:
        raise ValueError(
            f"expected a one-dimensional primitive null space for ({r},{s}), "
            f"found dimension {len(kernel)}"
        )

    half_level = level // 2
    target_word = (L(-1),) * half_level
    if level % 2:
        target_word += (G(sp.Rational(-1, 2)),)
    target_index = basis.index(target_word)
    target_coefficient = sp.factor(kernel[0][target_index])
    if target_coefficient == 0:
        raise ValueError("the Gram-kernel vector has zero leading coefficient")

    normalized = (
        (word, sp.factor(coefficient / target_coefficient))
        for word, coefficient in zip(basis, kernel[0])
        if coefficient != 0
    )
    return tuple(normalized)


def middle_slot_restriction(
    r: int,
    s: int,
    alpha: int,
    h_infinity: sp.Expr,
    h_zero: sp.Expr,
    b: sp.Expr,
) -> sp.Expr:
    r"""Return the exact middle-slot singular-vector restriction.

    The returned matrix element is

    .. math::

       \rho\!\left(
          G_{-1/2}^{\alpha}\nu_{h_\infty},
          \chi_{r,s},
          \nu_{h_0}
       \right).

    ``alpha`` is the parity label of the shifted trinion component used by
    the residue.  The singular vector is normalized by

    .. math::

       \chi_{r,s}
       =G_{-1/2}^{rs}\nu_{h_{r,s}}+\cdots.

    Unlike the first- and third-slot restrictions, this quantity is not in
    general a fusion polynomial or its reflected copy.
    """

    _validate_label(r, s)
    _validate_alpha(alpha)
    h_infinity = sp.sympify(h_infinity)
    h_zero = sp.sympify(h_zero)
    b = sp.sympify(b)
    if b == 0:
        raise ValueError("b must be nonzero")

    singular_weight = h_rs(r, s, b)
    central_charge = c_xi(b)
    first_word = (
        (G(sp.Rational(-1, 2)),)
        if alpha
        else ()
    )
    value = sp.S.Zero
    for middle_word, template_coefficient in _normalized_null_state_template(r, s):
        coefficient = template_coefficient.subs(_NULL_B, b)
        value += coefficient * _rho_middle_word(
            first_word,
            middle_word,
            (),
            d1=h_infinity,
            d2=singular_weight,
            d3=h_zero,
            c=central_charge,
        )
    return sp.factor(value)
