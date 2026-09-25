#!/usr/bin/env python3
r"""Exact ordered NS--R--R descendant three-point tensors.

The ordered slots are ``(infinity NS, one R, zero R)``, exactly as in
equation (1.7) of ``ramond_blocks_from_two_virasoro.pdf``.  This module is
deliberately separate from :mod:`nrr_three_point_tensor`, whose local order is
``(infinity R, one NS, zero R)``.  Permuting those punctures without also
transforming descendant local coordinates is valid for primaries but not for
descendants.

The supercurrent reductions below are the two NS--R Ward identities of
Hadasz--Jaskolski--Suchanek, arXiv:0810.1203, specialized to ``z=1``.  The
Ramond ground doublets are represented in the polynomial basis

``G_0|+> = |->`` and ``G_0|-> = (h-c/24)|+>``.

No Ramond large-central-charge limit is used here.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Mapping, Sequence

import sympy as sp

from ns_algebra.ns_sca import (
    G as NSG,
    L as NSL,
    Word as NSWord,
    act_mode as act_ns_mode,
    fermion_parity as ns_fermion_parity,
    is_pbw_word as is_ns_pbw_word,
    twice_level as ns_twice_level,
)
from .nrr_three_point_tensor import hjs_ground_tensor
from .ramond_sca import (
    G as RG,
    L as RL,
    PBWState,
    State as RState,
    act_mode as act_r_mode,
    is_pbw_word as is_r_pbw_word,
)


_PHASE = (sp.S.One + sp.I) / sp.sqrt(2)


def _validate_inputs(
    infinity_word: NSWord,
    middle_state: PBWState,
    zero_state: PBWState,
) -> None:
    """Validate the three negative-mode PBW inputs."""

    infinity_word = tuple(infinity_word)
    if infinity_word and not is_ns_pbw_word(infinity_word):
        raise ValueError("the infinity NS word must be in PBW order")
    if not isinstance(middle_state, PBWState):
        raise TypeError("middle_state must be a Ramond PBWState")
    if not isinstance(zero_state, PBWState):
        raise TypeError("zero_state must be a Ramond PBWState")
    if middle_state.word and not is_r_pbw_word(middle_state.word):
        raise ValueError("the middle Ramond word must be in PBW order")
    if zero_state.word and not is_r_pbw_word(zero_state.word):
        raise ValueError("the zero Ramond word must be in PBW order")


def _add_ns_terms(
    value: sp.Expr,
    scale: sp.Expr,
    acted: Mapping[NSWord, sp.Expr],
    evaluate,
    middle_state: PBWState,
    zero_state: PBWState,
) -> sp.Expr:
    """Accumulate a sparse NS-mode action in the infinity slot."""

    for output, coefficient in acted.items():
        value += sp.sympify(scale) * coefficient * evaluate(
            output,
            middle_state,
            zero_state,
        )
    return sp.expand(value)


def _add_r_terms(
    value: sp.Expr,
    scale: sp.Expr,
    acted: RState,
    evaluate,
    infinity_word: NSWord,
    middle_state: PBWState,
    zero_state: PBWState,
    *,
    slot: str,
) -> sp.Expr:
    """Accumulate a sparse Ramond-mode action in one Ramond slot."""

    for output, coefficient in acted.items():
        if slot == "middle":
            term = evaluate(infinity_word, output, zero_state)
        elif slot == "zero":
            term = evaluate(infinity_word, middle_state, output)
        else:  # pragma: no cover - internal programming error
            raise ValueError("slot must be 'middle' or 'zero'")
        value += sp.sympify(scale) * coefficient * term
    return sp.expand(value)


def _finite_p_bound(
    infinity_word: NSWord,
    middle_state: PBWState,
    zero_state: PBWState,
    offset: sp.Rational | int = 0,
) -> int:
    """Return a conservative finite bound for one Ward-identity sum."""

    twice_total = (
        (ns_twice_level(infinity_word) if infinity_word else 0)
        + middle_state.twice_descendant_level
        + zero_state.twice_descendant_level
    )
    return int(sp.ceiling(sp.Rational(twice_total, 2) + sp.Abs(offset))) + 2


def ns_rr_three_point_from_ground_tensor(
    infinity_word: NSWord,
    middle_state: PBWState,
    zero_state: PBWState,
    *,
    h_infinity: sp.Expr,
    h_middle: sp.Expr,
    h_zero: sp.Expr,
    c: sp.Expr,
    ground_tensor: Sequence[Sequence[sp.Expr]] | sp.MatrixBase,
    factor_result: bool = True,
) -> sp.Expr:
    r"""Extend arbitrary polynomial Ramond ground data to descendants.

    ``ground_tensor[p_middle,p_zero]`` gives the four terminal values with
    the NS primary in the infinity slot.  The recursion is exact at every
    finite level and terminates because each isolated negative mode is
    replaced by states of strictly smaller total descendant level.
    ``factor_result=False`` retains the exact expanded Ward polynomial;
    it skips only optional final algebraic factorization, not any states.
    """

    if not isinstance(factor_result, bool):
        raise TypeError("factor_result must be a boolean")
    _validate_inputs(infinity_word, middle_state, zero_state)
    infinity_word = tuple(infinity_word)
    h_infinity = sp.sympify(h_infinity)
    h_middle = sp.sympify(h_middle)
    h_zero = sp.sympify(h_zero)
    c = sp.sympify(c)
    terminal = sp.Matrix(ground_tensor)
    if terminal.shape != (2, 2):
        raise ValueError("ground_tensor must be a two-by-two matrix")
    terminal_values = tuple(
        tuple(sp.sympify(terminal[row, column]) for column in range(2))
        for row in range(2)
    )

    @lru_cache(maxsize=None)
    def evaluate(
        infinity: NSWord,
        middle: PBWState,
        zero: PBWState,
    ) -> sp.Expr:
        """Apply the ordered NS--R Ward reductions recursively."""

        if infinity:
            leading = infinity[0]
            rest = infinity[1:]
            if leading.kind == "L":
                n = -int(leading.index)
                value = _add_r_terms(
                    sp.S.Zero,
                    1,
                    act_r_mode(RL(n), {zero: sp.S.One}, h=h_zero, c=c),
                    evaluate,
                    rest,
                    middle,
                    zero,
                    slot="zero",
                )
                for mode_index in range(-1, n + 1):
                    value = _add_r_terms(
                        value,
                        sp.binomial(n + 1, mode_index + 1),
                        act_r_mode(
                            RL(mode_index),
                            {middle: sp.S.One},
                            h=h_middle,
                            c=c,
                        ),
                        evaluate,
                        rest,
                        middle,
                        zero,
                        slot="middle",
                    )
                return sp.expand(value)

            r = -sp.Rational(leading.index)
            n = r - sp.Rational(1, 2)
            if n.q != 1 or n < 0:
                raise ValueError("an infinity NS G mode must be negative")
            n_int = int(n)
            sign = sp.Integer(-1) ** (
                ns_fermion_parity(rest) + zero.parity + 1
            )
            bound = _finite_p_bound(rest, middle, zero, r)

            # Left side of the first HJS NS--R Ward identity.
            value = sp.S.Zero
            for p in range(bound + 1):
                value = _add_r_terms(
                    value,
                    sp.binomial(r, p),
                    act_r_mode(RG(p), {middle: sp.S.One}, h=h_middle, c=c),
                    evaluate,
                    rest,
                    middle,
                    zero,
                    slot="middle",
                )

            # Remove the p>0 infinity terms; p=0 is the target state.
            for p in range(1, bound + 1):
                value = _add_ns_terms(
                    value,
                    -sp.binomial(sp.Rational(1, 2), p) * (-1) ** p,
                    act_ns_mode(
                        NSG(sp.Rational(p) - r),
                        {rest: sp.S.One},
                        h=h_infinity,
                        c=c,
                    ),
                    evaluate,
                    middle,
                    zero,
                )

            # Move the remaining contour to the zero Ramond leg.
            for p in range(bound + 1):
                value = _add_r_terms(
                    value,
                    sp.I
                    * sign
                    * sp.binomial(sp.Rational(1, 2), p)
                    * (-1) ** p,
                    act_r_mode(
                        RG(n_int + p),
                        {zero: sp.S.One},
                        h=h_zero,
                        c=c,
                    ),
                    evaluate,
                    rest,
                    middle,
                    zero,
                    slot="zero",
                )
            return sp.expand(value)

        if zero.word:
            leading = zero.word[0]
            rest = PBWState(zero.word[1:], zero.ground_parity)
            if leading.kind == "L":
                n = -leading.index
                value = _add_ns_terms(
                    sp.S.Zero,
                    1,
                    act_ns_mode(NSL(n), {(): sp.S.One}, h=h_infinity, c=c),
                    evaluate,
                    middle,
                    rest,
                )
                maximum_mode = middle.twice_descendant_level // 2
                for mode_index in range(-1, maximum_mode + 1):
                    value = _add_r_terms(
                        value,
                        -sp.binomial(1 - n, mode_index + 1),
                        act_r_mode(
                            RL(mode_index),
                            {middle: sp.S.One},
                            h=h_middle,
                            c=c,
                        ),
                        evaluate,
                        (),
                        middle,
                        rest,
                        slot="middle",
                    )
                return sp.expand(value)

            r = -leading.index
            sign = sp.Integer(-1) ** (rest.parity + 1)
            bound = _finite_p_bound((), middle, rest, r)
            lhs = sp.S.Zero
            first_rhs = sp.S.Zero
            remainder = sp.S.Zero
            for p in range(bound + 1):
                lhs = _add_r_terms(
                    lhs,
                    sp.binomial(sp.Rational(1, 2) - r, p),
                    act_r_mode(RG(p), {middle: sp.S.One}, h=h_middle, c=c),
                    evaluate,
                    (),
                    middle,
                    rest,
                    slot="middle",
                )
                first_rhs = _add_ns_terms(
                    first_rhs,
                    sp.binomial(sp.Rational(1, 2), p) * (-1) ** p,
                    act_ns_mode(
                        NSG(sp.Rational(p + r) - sp.Rational(1, 2)),
                        {(): sp.S.One},
                        h=h_infinity,
                        c=c,
                    ),
                    evaluate,
                    middle,
                    rest,
                )
                if p > 0:
                    remainder = _add_r_terms(
                        remainder,
                        sp.binomial(sp.Rational(1, 2), p) * (-1) ** p,
                        act_r_mode(
                            RG(-r + p),
                            {rest: sp.S.One},
                            h=h_zero,
                            c=c,
                        ),
                        evaluate,
                        (),
                        middle,
                        rest,
                        slot="zero",
                    )
            return sp.expand(sp.I * sign * (lhs - first_rhs) - remainder)

        if middle.word:
            leading = middle.word[0]
            rest = PBWState(middle.word[1:], middle.ground_parity)
            if leading.kind == "L":
                n = -leading.index
                rest_level = sp.Rational(rest.twice_descendant_level, 2)
                factor = (-1) ** n * (
                    n * h_zero + h_middle + rest_level - h_infinity
                )
                return sp.expand(factor * evaluate((), rest, zero))

            r = -leading.index
            sign = sp.Integer(-1) ** (zero.parity + 1)
            bound = _finite_p_bound((), rest, zero, r)
            value = sp.S.Zero
            for p in range(bound + 1):
                coefficient = (
                    sp.binomial(sp.Rational(1, 2) - r, p) * (-1) ** p
                )
                value = _add_ns_terms(
                    value,
                    coefficient,
                    act_ns_mode(
                        NSG(sp.Rational(p + r) - sp.Rational(1, 2)),
                        {(): sp.S.One},
                        h=h_infinity,
                        c=c,
                    ),
                    evaluate,
                    rest,
                    zero,
                )
                value = _add_r_terms(
                    value,
                    -sp.I * sign * coefficient * (-1) ** r,
                    act_r_mode(RG(p), {zero: sp.S.One}, h=h_zero, c=c),
                    evaluate,
                    (),
                    rest,
                    zero,
                    slot="zero",
                )
                if p > 0:
                    value = _add_r_terms(
                        value,
                        -sp.binomial(sp.Rational(1, 2), p),
                        act_r_mode(
                            RG(p - r),
                            {rest: sp.S.One},
                            h=h_middle,
                            c=c,
                        ),
                        evaluate,
                        (),
                        rest,
                        zero,
                        slot="middle",
                    )
            return sp.expand(value)

        return terminal_values[middle.ground_parity][zero.ground_parity]

    expression = evaluate(infinity_word, middle_state, zero_state)
    return sp.factor(sp.cancel(expression)) if factor_result else expression


def hjs_ns_rr_polynomial_ground_tensor(
    beta_middle: sp.Expr,
    beta_zero: sp.Expr,
    *,
    structure_sign: int,
) -> sp.Matrix:
    r"""Return one HJS NS--R--R ground tensor in the polynomial basis.

    Both Ramond slots are linear slots, so both odd basis vectors carry the
    same factor ``exp(i*pi/4)*beta``.  This is distinct from the
    anti-linear first-Ramond-slot conversion used for the R--NS--R form.
    """

    if structure_sign not in (-1, 1):
        raise ValueError("structure_sign must be +1 or -1")
    middle = sp.diag(1, _PHASE * sp.sympify(beta_middle))
    zero = sp.diag(1, _PHASE * sp.sympify(beta_zero))
    return (middle * hjs_ground_tensor(structure_sign) * zero).applyfunc(
        lambda entry: sp.factor(sp.expand(entry))
    )


__all__ = [
    "hjs_ns_rr_polynomial_ground_tensor",
    "ns_rr_three_point_from_ground_tensor",
]
