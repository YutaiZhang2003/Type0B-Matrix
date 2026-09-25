#!/usr/bin/env python3
r"""Exact ordered NS--NS--NS descendant three-point tensors.

The ordered slots are ``(infinity, 1, 0)``.  Every input is a canonical
negative-mode NS PBW word from :mod:`ns_sca`, and all calculations use
Xi Yin's ordinary Virasoro central charge.

The global ``osp(1|2)`` tensor is boundary data for the Ward recursion.
In particular, a word made only from ``L_-1`` and ``G_-1/2`` is never
stripped mode by mode.  Once all three words are global, the evaluator
terminates on the exact Belavin--Geiko ``tau`` formula already certified
in :mod:`ns_middle_fusion`.  This atomic terminal is essential for the
normalized odd form: recursively stripping a global ``G_-1/2`` would
change the constant term in ``tau^(1,1,1)``.

Non-global modes on the first and third slots are removed with the
ordered Hadasz--Jaskolski--Suchanek Ward identities.  Once both outer
words are global, the exact middle-slot reducer in
:mod:`ns_middle_fusion` removes any remaining non-global middle modes.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal, Mapping

import sympy as sp

from .ns_middle_fusion import _rho_global_exact, _rho_middle_word
from .ns_sca import (
    G,
    L,
    Word,
    act_mode,
    fermion_parity,
    is_pbw_word,
    pbw_basis,
    twice_level,
)


TensorKey = tuple[Word, Word, Word]

_HALF = sp.Rational(1, 2)
_GLOBAL_MODES = frozenset((L(-1), G(-_HALF)))


def _simplify_entry(expression: sp.Expr) -> sp.Expr:
    """Return one exact tensor entry in stable factored form."""

    return sp.factor(sp.cancel(sp.sympify(expression)))


def _validate_word(word: Word, name: str) -> Word:
    """Materialize and validate one canonical negative-mode NS PBW word."""

    result = tuple(word)
    if result and not is_pbw_word(result):
        raise ValueError(f"{name} must be a canonical negative NS PBW word")
    return result


def _validate_twice_level(twice_level_value: int, name: str) -> int:
    """Require a nonnegative integer twice-level."""

    if not isinstance(twice_level_value, int):
        raise TypeError(f"{name} must be an integer")
    if twice_level_value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return twice_level_value


def _is_global_word(word: Word) -> bool:
    """Return whether a word belongs to the global ``osp(1|2)`` module."""

    return all(mode in _GLOBAL_MODES for mode in word)


def _add_outer_acted_terms(
    value: sp.Expr,
    coefficient: sp.Expr,
    acted: Mapping[Word, sp.Expr],
    evaluate,
    infinity_word: Word,
    middle_word: Word,
    zero_word: Word,
    *,
    acted_slot: Literal["infinity", "zero"],
) -> sp.Expr:
    """Accumulate a sparse NS mode action in one outer tensor slot."""

    for output_word, state_coefficient in acted.items():
        if acted_slot == "infinity":
            term = evaluate(output_word, middle_word, zero_word)
        elif acted_slot == "zero":
            term = evaluate(infinity_word, middle_word, output_word)
        else:
            raise AssertionError("acted_slot must be 'infinity' or 'zero'")
        value += sp.sympify(coefficient) * state_coefficient * term
    return sp.expand(value)


def _add_middle_acted_terms(
    value: sp.Expr,
    coefficient: sp.Expr,
    acted: Mapping[Word, sp.Expr],
    evaluate,
    infinity_word: Word,
    zero_word: Word,
) -> sp.Expr:
    """Accumulate a sparse NS mode action in the middle tensor slot."""

    for output_word, state_coefficient in acted.items():
        value += (
            sp.sympify(coefficient)
            * state_coefficient
            * evaluate(infinity_word, output_word, zero_word)
        )
    return sp.expand(value)


def _legacy_ns_three_point(
    infinity_word: Word,
    middle_word: Word,
    zero_word: Word,
    *,
    h_infinity: sp.Expr,
    h_middle: sp.Expr,
    h_zero: sp.Expr,
    c: sp.Expr,
) -> sp.Expr:
    r"""Evaluate one exact ordered NS descendant three-point tensor.

    The two normalized NS forms are packaged without introducing another
    structure label.  For homogeneous inputs, the selected form parity is

    .. math::

       a=F(\xi_\infty)+F(\xi_{\rm middle})+F(\xi_0)\pmod 2.

    The global subspace is an atomic terminal.  Outside it, the recursion
    first removes non-global slot-1 modes, then non-global slot-3 modes,
    and finally delegates the middle reduction to
    :func:`ns_middle_fusion._rho_middle_word`.
    """

    infinity_word = _validate_word(infinity_word, "infinity_word")
    middle_word = _validate_word(middle_word, "middle_word")
    zero_word = _validate_word(zero_word, "zero_word")
    h_infinity = sp.sympify(h_infinity)
    h_middle = sp.sympify(h_middle)
    h_zero = sp.sympify(h_zero)
    c = sp.sympify(c)

    @lru_cache(maxsize=None)
    def evaluate(
        left: Word,
        middle: Word,
        right: Word,
    ) -> sp.Expr:
        """Apply the terminating ordered Ward reduction."""

        if (
            _is_global_word(left)
            and _is_global_word(middle)
            and _is_global_word(right)
        ):
            return _rho_global_exact(
                left,
                middle,
                right,
                h_infinity,
                h_middle,
                h_zero,
            )

        if not _is_global_word(left):
            leading = left[0]
            rest = left[1:]
            if leading.kind == "L":
                n = int(-leading.index)
                value = _add_outer_acted_terms(
                    sp.S.Zero,
                    sp.S.One,
                    act_mode(
                        L(n),
                        {right: sp.S.One},
                        h=h_zero,
                        c=c,
                    ),
                    evaluate,
                    rest,
                    middle,
                    right,
                    acted_slot="zero",
                )
                for mode_index in range(-1, n + 1):
                    value = _add_middle_acted_terms(
                        value,
                        sp.binomial(n + 1, mode_index + 1),
                        act_mode(
                            L(mode_index),
                            {middle: sp.S.One},
                            h=h_middle,
                            c=c,
                        ),
                        evaluate,
                        rest,
                        right,
                    )
                return sp.expand(value)

            k = sp.Rational(-leading.index)
            if k.q != 2 or k <= _HALF:
                raise AssertionError(
                    "a stripped non-global outer G mode must have "
                    "positive half-integer magnitude greater than 1/2"
                )
            sign = sp.Integer(-1) ** (
                fermion_parity(right) + fermion_parity(rest) + 1
            )
            value = _add_outer_acted_terms(
                sp.S.Zero,
                sign,
                act_mode(
                    G(k),
                    {right: sp.S.One},
                    h=h_zero,
                    c=c,
                ),
                evaluate,
                rest,
                middle,
                right,
                acted_slot="zero",
            )
            for twice_index in range(-1, int(2 * k) + 1, 2):
                mode_index = sp.Rational(twice_index, 2)
                value = _add_middle_acted_terms(
                    value,
                    sp.binomial(k + _HALF, mode_index + _HALF),
                    act_mode(
                        G(mode_index),
                        {middle: sp.S.One},
                        h=h_middle,
                        c=c,
                    ),
                    evaluate,
                    rest,
                    right,
                )
            return sp.expand(value)

        if not _is_global_word(right):
            leading = right[0]
            rest = right[1:]
            if leading.kind == "L":
                n = int(-leading.index)
                value = _add_outer_acted_terms(
                    sp.S.Zero,
                    sp.S.One,
                    act_mode(
                        L(n),
                        {left: sp.S.One},
                        h=h_infinity,
                        c=c,
                    ),
                    evaluate,
                    left,
                    middle,
                    rest,
                    acted_slot="infinity",
                )
                maximum_mode = twice_level(middle) // 2 if middle else 0
                for mode_index in range(-1, maximum_mode + 1):
                    value = _add_middle_acted_terms(
                        value,
                        -sp.binomial(1 - n, mode_index + 1),
                        act_mode(
                            L(mode_index),
                            {middle: sp.S.One},
                            h=h_middle,
                            c=c,
                        ),
                        evaluate,
                        left,
                        rest,
                    )
                return sp.expand(value)

            k = sp.Rational(-leading.index)
            if k.q != 2 or k <= _HALF:
                raise AssertionError(
                    "a stripped non-global outer G mode must have "
                    "positive half-integer magnitude greater than 1/2"
                )
            sign = sp.Integer(-1) ** (
                fermion_parity(left) + fermion_parity(rest) + 1
            )
            value = _add_outer_acted_terms(
                sp.S.Zero,
                sign,
                act_mode(
                    G(k),
                    {left: sp.S.One},
                    h=h_infinity,
                    c=c,
                ),
                evaluate,
                left,
                middle,
                rest,
                acted_slot="infinity",
            )
            middle_twice_level = twice_level(middle) if middle else 0
            for twice_index in range(-1, middle_twice_level + 1, 2):
                mode_index = sp.Rational(twice_index, 2)
                value = _add_middle_acted_terms(
                    value,
                    -sign
                    * sp.binomial(_HALF - k, mode_index + _HALF),
                    act_mode(
                        G(mode_index),
                        {middle: sp.S.One},
                        h=h_middle,
                        c=c,
                    ),
                    evaluate,
                    left,
                    rest,
                )
            return sp.expand(value)

        return _rho_middle_word(
            left,
            middle,
            right,
            d1=h_infinity,
            d2=h_middle,
            d3=h_zero,
            c=c,
        )

    return _simplify_entry(evaluate(infinity_word, middle_word, zero_word))


def _linear_action(
    *,
    slot: int,
    mode,
    words: tuple[Word, Word, Word],
    weights: tuple[sp.Expr, sp.Expr, sp.Expr],
    c: sp.Expr,
) -> sp.Expr:
    """Apply one Ward-identity mode action to a selected tensor slot."""

    value = sp.S.Zero
    for acted_word, coefficient in act_mode(
        mode,
        {words[slot]: sp.S.One},
        h=weights[slot],
        c=c,
    ).items():
        changed = list(words)
        changed[slot] = acted_word
        value += coefficient * _three_point_ward_cached(
            changed[0],
            changed[1],
            changed[2],
            weights[0],
            weights[1],
            weights[2],
            c,
        )
    return sp.expand(value)


@lru_cache(maxsize=None)
def _three_point_ward_cached(
    infinity_word: Word,
    middle_word: Word,
    zero_word: Word,
    h_infinity: sp.Expr,
    h_middle: sp.Expr,
    h_zero: sp.Expr,
    c: sp.Expr,
) -> sp.Expr:
    """Reduce one fixed-parity NS three-form with the HJS Ward identities."""

    words = (infinity_word, middle_word, zero_word)
    weights = (h_infinity, h_middle, h_zero)
    primary_components = ((), (G(-_HALF),))
    if all(word in primary_components for word in words):
        return _rho_global_exact(
            infinity_word,
            middle_word,
            zero_word,
            h_infinity,
            h_middle,
            h_zero,
        )

    # Remove the middle word first.  The eventual endpoint reflection then
    # occurs with a primary component in the middle slot and has no hidden
    # middle-field reflection phase.
    if middle_word:
        mode = middle_word[0]
        tail = middle_word[1:]
        reduced_words = (infinity_word, tail, zero_word)
        if mode.kind == "G":
            k = -mode.index
            parity_13 = (
                fermion_parity(infinity_word) + fermion_parity(zero_word)
            ) % 2
            value = sp.S.Zero
            maximum_first = max(
                0,
                int(sp.floor(sp.Rational(twice_level(infinity_word), 2) - k)),
            )
            for offset in range(maximum_first + 1):
                value += sp.binomial(k - sp.Rational(3, 2) + offset, offset) * (
                    _linear_action(
                        slot=0,
                        mode=G(k + offset),
                        words=reduced_words,
                        weights=weights,
                        c=c,
                    )
                )
            maximum_third = (twice_level(zero_word) + 1) // 2
            ward_sign = -sp.Integer(-1) ** (
                parity_13 + int(k + sp.Rational(1, 2))
            )
            for offset in range(maximum_third + 1):
                value += ward_sign * sp.binomial(
                    k - sp.Rational(3, 2) + offset,
                    offset,
                ) * _linear_action(
                    slot=2,
                    mode=G(sp.Rational(2 * offset - 1, 2)),
                    words=reduced_words,
                    weights=weights,
                    c=c,
                )
            return sp.expand(value)

        n = int(-mode.index)
        if n == 1:
            exponent = (
                h_infinity
                + sp.Rational(twice_level(infinity_word), 2)
                - h_middle
                - sp.Rational(twice_level(tail), 2)
                - h_zero
                - sp.Rational(twice_level(zero_word), 2)
            )
            return sp.expand(
                exponent
                * _three_point_ward_cached(
                    infinity_word,
                    tail,
                    zero_word,
                    h_infinity,
                    h_middle,
                    h_zero,
                    c,
                )
            )
        value = sp.S.Zero
        maximum_first = max(0, twice_level(infinity_word) // 2 - n)
        for offset in range(maximum_first + 1):
            value += sp.binomial(n - 2 + offset, n - 2) * _linear_action(
                slot=0,
                mode=L(n + offset),
                words=reduced_words,
                weights=weights,
                c=c,
            )
        maximum_third = max(0, twice_level(zero_word) // 2 + 1)
        for offset in range(maximum_third + 1):
            value += sp.Integer(-1) ** n * sp.binomial(
                n - 2 + offset,
                n - 2,
            ) * _linear_action(
                slot=2,
                mode=L(offset - 1),
                words=reduced_words,
                weights=weights,
                c=c,
            )
        return sp.expand(value)

    if infinity_word:
        mode = infinity_word[0]
        tail = infinity_word[1:]
        reduced_words = (tail, middle_word, zero_word)
        if mode.kind == "G":
            k = -mode.index
            if k <= _HALF:
                return _three_point_ward_cached(
                    zero_word,
                    middle_word,
                    infinity_word,
                    h_zero,
                    h_middle,
                    h_infinity,
                    c,
                )
            parity_13 = (fermion_parity(tail) + fermion_parity(zero_word)) % 2
            value = sp.Integer(-1) ** (parity_13 + 1) * _linear_action(
                slot=2,
                mode=G(k),
                words=reduced_words,
                weights=weights,
                c=c,
            )
            for offset in range(-1, int(k - _HALF) + 1):
                value += sp.binomial(k + _HALF, offset + 1) * _linear_action(
                    slot=1,
                    mode=G(sp.Rational(2 * offset + 1, 2)),
                    words=reduced_words,
                    weights=weights,
                    c=c,
                )
            return sp.expand(value)

        n = int(-mode.index)
        value = _linear_action(
            slot=2,
            mode=L(n),
            words=reduced_words,
            weights=weights,
            c=c,
        )
        for offset in range(-1, n + 1):
            value += sp.binomial(n + 1, offset + 1) * _linear_action(
                slot=1,
                mode=L(offset),
                words=reduced_words,
                weights=weights,
                c=c,
            )
        return sp.expand(value)

    if zero_word:
        return _three_point_ward_cached(
            zero_word,
            middle_word,
            infinity_word,
            h_zero,
            h_middle,
            h_infinity,
            c,
        )
    return sp.S.One


def ns_three_point(
    infinity_word: Word,
    middle_word: Word,
    zero_word: Word,
    *,
    h_infinity: sp.Expr,
    h_middle: sp.Expr,
    h_zero: sp.Expr,
    c: sp.Expr,
    simplify: bool = True,
) -> sp.Expr:
    r"""Evaluate one exact ordered NS descendant three-point tensor.

    The two normalized NS forms are packaged without another structure label.
    For homogeneous inputs, the selected form parity is the sum modulo two of
    the three descendant fermion parities.  The Ward reduction and its global
    seeds use the fixed-parity trilinear convention required by odd-null
    factorization.
    """

    infinity_word = _validate_word(infinity_word, "infinity_word")
    middle_word = _validate_word(middle_word, "middle_word")
    zero_word = _validate_word(zero_word, "zero_word")
    result = _three_point_ward_cached(
        infinity_word,
        middle_word,
        zero_word,
        sp.sympify(h_infinity),
        sp.sympify(h_middle),
        sp.sympify(h_zero),
        sp.sympify(c),
    )
    return _simplify_entry(result) if simplify else result


def ns_level_tensor(
    infinity_twice_level: int,
    middle_twice_level: int,
    zero_twice_level: int,
    *,
    h_infinity: sp.Expr,
    h_middle: sp.Expr,
    h_zero: sp.Expr,
    c: sp.Expr,
) -> tuple[
    tuple[Word, ...],
    tuple[Word, ...],
    tuple[Word, ...],
    dict[TensorKey, sp.Expr],
]:
    """Build every ordered NS tensor entry at three fixed twice-levels."""

    infinity_twice_level = _validate_twice_level(
        infinity_twice_level,
        "infinity_twice_level",
    )
    middle_twice_level = _validate_twice_level(
        middle_twice_level,
        "middle_twice_level",
    )
    zero_twice_level = _validate_twice_level(
        zero_twice_level,
        "zero_twice_level",
    )
    infinity_basis = tuple(pbw_basis(infinity_twice_level))
    middle_basis = tuple(pbw_basis(middle_twice_level))
    zero_basis = tuple(pbw_basis(zero_twice_level))
    entries: dict[TensorKey, sp.Expr] = {}
    for infinity_word in infinity_basis:
        for middle_word in middle_basis:
            for zero_word in zero_basis:
                entries[(infinity_word, middle_word, zero_word)] = (
                    ns_three_point(
                        infinity_word,
                        middle_word,
                        zero_word,
                        h_infinity=h_infinity,
                        h_middle=h_middle,
                        h_zero=h_zero,
                        c=c,
                    )
                )
    return infinity_basis, middle_basis, zero_basis, entries


__all__ = [
    "TensorKey",
    "ns_level_tensor",
    "ns_three_point",
]
