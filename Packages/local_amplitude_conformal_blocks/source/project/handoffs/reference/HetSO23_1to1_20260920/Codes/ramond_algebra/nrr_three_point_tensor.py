#!/usr/bin/env python3
"""Exact ordered R--NS--R descendant three-point tensors.

The ordered slots are ``(infinity R, middle NS at z=1, zero R)``.  The
module evaluates arbitrary finite PBW descendants on all three legs in
Xi Yin's ordinary-central-charge convention.

The published Hadasz--Jaskolski--Suchanek (HJS) Ramond ground basis is

    G_0 w^+_beta = exp(i*pi/4) beta w^-_beta,
    G_0 w^-_beta = exp(3*i*pi/4) beta w^+_beta,

and the three-point form is anti-linear in slot 1.  The exact Ramond
action engine in :mod:`ramond_sca` instead uses the polynomial basis

    G_0 |+> = |->,
    G_0 |-> = (h-c/24)|+>.

This module performs the basis conversion explicitly.  All Ward
recursion is carried out in the polynomial basis, where mode-action
coefficients are rational in ``h`` and ``c``.  The public HJS evaluator
then converts the two external states back with the anti-linear slot-1
factor included.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal, Sequence

import sympy as sp

from ns_algebra.ns_sca import (
    G as NSG,
    L as NSL,
    Word as NSWord,
    act_mode as act_ns_mode,
    is_pbw_word as is_ns_pbw_word,
    pbw_basis as ns_pbw_basis,
    twice_level as ns_twice_level,
)
from .nrr_middle_fusion import rr_middle_ns_word_polynomial
from .ramond_sca import (
    G as RG,
    L as RL,
    PBWState,
    State as RState,
    act_mode as act_r_mode,
    is_pbw_word as is_r_pbw_word,
    pbw_basis as r_pbw_basis,
)


GroundBasis = Literal["hjs", "polynomial"]
TensorKey = tuple[PBWState, NSWord, PBWState]

_PHASE = (sp.S.One + sp.I) / sp.sqrt(2)


def _validate_structure_sign(structure_sign: int) -> None:
    """Require the published HJS trilinear label ``+1`` or ``-1``."""

    if structure_sign not in (-1, 1):
        raise ValueError("structure_sign must be +1 or -1")


def _validate_ground_basis(ground_basis: str) -> None:
    """Require one of the two documented Ramond ground bases."""

    if ground_basis not in ("hjs", "polynomial"):
        raise ValueError("ground_basis must be 'hjs' or 'polynomial'")


def _validate_inputs(
    infinity_state: PBWState,
    middle_word: NSWord,
    zero_state: PBWState,
) -> None:
    """Validate the three negative-mode PBW inputs."""

    if not isinstance(infinity_state, PBWState):
        raise TypeError("infinity_state must be a Ramond PBWState")
    if not isinstance(zero_state, PBWState):
        raise TypeError("zero_state must be a Ramond PBWState")
    middle_word = tuple(middle_word)
    if infinity_state.word and not is_r_pbw_word(infinity_state.word):
        raise ValueError("the infinity Ramond word must be in PBW order")
    if zero_state.word and not is_r_pbw_word(zero_state.word):
        raise ValueError("the zero Ramond word must be in PBW order")
    if middle_word and not is_ns_pbw_word(middle_word):
        raise ValueError("the middle NS word must be in PBW order")


def hjs_ground_tensor(structure_sign: int) -> sp.Matrix:
    r"""Return all four normalized HJS ground tensor entries.

    Rows are ``(w^+_infinity,w^-_infinity)`` and columns are
    ``(w^+_zero,w^-_zero)``:

    .. math::

       \begin{pmatrix}1&1\\ s i&s\end{pmatrix},
       \qquad s=\pm1.
    """

    _validate_structure_sign(structure_sign)
    s = sp.Integer(structure_sign)
    return sp.Matrix([[1, 1], [s * sp.I, s]])


def hjs_to_polynomial_ground_tensor(
    beta_infinity: sp.Expr,
    beta_zero: sp.Expr,
    *,
    structure_sign: int,
) -> sp.Matrix:
    r"""Convert all four HJS ground entries to the polynomial basis.

    The ket basis changes are

    .. math::

       |-\rangle_{\rm P}=e^{i\pi/4}\beta\,w^-,
       \qquad |+\rangle_{\rm P}=w^+.

    Since slot 1 is anti-linear and
    ``conjugate(beta)=-beta`` on the HJS physical line, its effective
    odd conversion factor is ``exp(3*i*pi/4)*beta_infinity``.  Slot 3 is
    linear and uses ``exp(i*pi/4)*beta_zero``.
    """

    _validate_structure_sign(structure_sign)
    beta_infinity = sp.sympify(beta_infinity)
    beta_zero = sp.sympify(beta_zero)
    left = sp.diag(1, sp.I * _PHASE * beta_infinity)
    right = sp.diag(1, _PHASE * beta_zero)
    return (left * hjs_ground_tensor(structure_sign) * right).applyfunc(
        lambda entry: sp.factor(sp.expand(entry))
    )


def _hjs_from_polynomial_tensor_factor(
    infinity_ground_parity: int,
    zero_ground_parity: int,
    beta_infinity: sp.Expr,
    beta_zero: sp.Expr,
) -> sp.Expr:
    """Return the inverse tensor multiplier for HJS descendant states."""

    if infinity_ground_parity and sp.sympify(beta_infinity) == 0:
        raise ValueError(
            "the HJS odd infinity ground state is singular at beta_infinity=0"
        )
    if zero_ground_parity and sp.sympify(beta_zero) == 0:
        raise ValueError(
            "the HJS odd zero ground state is singular at beta_zero=0"
        )
    left = (
        1 / (sp.I * _PHASE * beta_infinity)
        if infinity_ground_parity
        else sp.S.One
    )
    right = (
        1 / (_PHASE * beta_zero)
        if zero_ground_parity
        else sp.S.One
    )
    return sp.factor(left * right)


def polynomial_ground_form_basis(
    form_parity: int,
) -> tuple[sp.ImmutableMatrix, sp.ImmutableMatrix]:
    r"""Return the two coordinate tensors of one homogeneous form space.

    For even forms the ordered coordinates are the ``(+,+)`` and
    ``(-,-)`` terminal values.  For odd forms they are ``(+,-)`` and
    ``(-,+)``.  These four matrix units are independent of Ramond
    momentum branches and of ``c``.
    """

    if form_parity not in (0, 1):
        raise ValueError("form_parity must be zero or one")
    if form_parity == 0:
        pairs = ((0, 0), (1, 1))
    else:
        pairs = ((0, 1), (1, 0))
    result = []
    for row, column in pairs:
        matrix = sp.zeros(2)
        matrix[row, column] = 1
        result.append(sp.ImmutableMatrix(matrix))
    return result[0], result[1]


def hjs_polynomial_form_coordinates(
    beta_infinity: sp.Expr,
    beta_zero: sp.Expr,
    *,
    structure_sign: int,
    form_parity: int,
) -> sp.ImmutableMatrix:
    r"""Return one HJS form as two polynomial-ground coordinates."""

    tensor = hjs_to_polynomial_ground_tensor(
        beta_infinity,
        beta_zero,
        structure_sign=structure_sign,
    )
    basis = polynomial_ground_form_basis(form_parity)
    coordinates = []
    for matrix_unit in basis:
        nonzero = tuple(
            (row, column)
            for row in range(2)
            for column in range(2)
            if matrix_unit[row, column] != 0
        )
        row, column = nonzero[0]
        coordinates.append(tensor[row, column])
    return sp.ImmutableMatrix(2, 1, coordinates)


def _add_r_acted_terms(
    value: sp.Expr,
    coefficient: sp.Expr,
    acted: RState,
    evaluate,
    infinity_state: PBWState,
    middle_word: NSWord,
    zero_state: PBWState,
    *,
    acted_slot: Literal["infinity", "zero"],
) -> sp.Expr:
    """Accumulate a sparse Ramond mode action in one outer slot."""

    for output_state, state_coefficient in acted.items():
        if acted_slot == "infinity":
            term = evaluate(output_state, middle_word, zero_state)
        else:
            term = evaluate(infinity_state, middle_word, output_state)
        value += sp.sympify(coefficient) * state_coefficient * term
    return sp.expand(value)


def _add_ns_acted_terms(
    value: sp.Expr,
    coefficient: sp.Expr,
    acted: dict[NSWord, sp.Expr],
    evaluate,
    infinity_state: PBWState,
    zero_state: PBWState,
) -> sp.Expr:
    """Accumulate a sparse middle-slot NS mode action."""

    for output_word, state_coefficient in acted.items():
        value += (
            sp.sympify(coefficient)
            * state_coefficient
            * evaluate(infinity_state, output_word, zero_state)
        )
    return sp.expand(value)


def rr_three_point_from_ground_tensor(
    infinity_state: PBWState,
    middle_word: NSWord,
    zero_state: PBWState,
    *,
    h_infinity: sp.Expr,
    h_ns: sp.Expr,
    h_zero: sp.Expr,
    c: sp.Expr,
    ground_tensor: Sequence[Sequence[sp.Expr]] | sp.MatrixBase,
    simplify: bool = True,
) -> sp.Expr:
    r"""Extend arbitrary polynomial Ramond ground data to descendants.

    The three inputs are finite negative-mode PBW descendants in slots
    ``(infinity R, middle NS, zero R)``.  The four entries of
    ``ground_tensor`` are the terminal values on the two polynomial
    Ramond ground doublets.  They are not required to be an HJS-normalized
    structure.

    Slot-1 modes are stripped with the HJS outer Ward identities.  Once
    slot 1 is a ground state, slot-3 modes are stripped with the same
    contour identities continued to negative mode number.  The remaining
    middle descendant is evaluated by
    :func:`rr_middle_ns_word_polynomial`.

    ``simplify=False`` preserves the exact Ward recursion but skips the final
    symbolic cancellation and factorization.  This is the production path for
    numerical templates; the simplified form remains the public default.
    """

    _validate_inputs(infinity_state, middle_word, zero_state)
    h_infinity = sp.sympify(h_infinity)
    h_ns = sp.sympify(h_ns)
    h_zero = sp.sympify(h_zero)
    c = sp.sympify(c)
    terminal = sp.Matrix(ground_tensor)
    if terminal.shape != (2, 2):
        raise ValueError("ground_tensor must be a two-by-two matrix")
    terminal = terminal.applyfunc(sp.sympify)
    middle_word = tuple(middle_word)

    @lru_cache(maxsize=None)
    def evaluate(
        left: PBWState,
        middle: NSWord,
        right: PBWState,
    ) -> sp.Expr:
        """Apply a terminating ordered outer-leg Ward reduction."""

        if left.word:
            leading = left.word[0]
            rest = PBWState(left.word[1:], left.ground_parity)
            if leading.kind == "L":
                n = -leading.index
                value = sp.S.Zero
                acted_right = act_r_mode(
                    RL(n),
                    {right: sp.S.One},
                    h=h_zero,
                    c=c,
                )
                value = _add_r_acted_terms(
                    value,
                    1,
                    acted_right,
                    evaluate,
                    rest,
                    middle,
                    right,
                    acted_slot="zero",
                )
                for mode_index in range(-1, n + 1):
                    acted_middle = act_ns_mode(
                        NSL(mode_index),
                        {middle: sp.S.One},
                        h=h_ns,
                        c=c,
                    )
                    value = _add_ns_acted_terms(
                        value,
                        sp.binomial(n + 1, mode_index + 1),
                        acted_middle,
                        evaluate,
                        rest,
                        right,
                    )
                return sp.expand(value)

            n = -leading.index
            sign = sp.Integer(-1) ** (right.parity + rest.parity + 1)
            acted_right = act_r_mode(
                RG(n),
                {right: sp.S.One},
                h=h_zero,
                c=c,
            )
            value = _add_r_acted_terms(
                sp.S.Zero,
                sign,
                acted_right,
                evaluate,
                rest,
                middle,
                right,
                acted_slot="zero",
            )
            middle_twice_level = ns_twice_level(middle) if middle else 0
            for twice_index in range(-1, middle_twice_level + 1, 2):
                mode_index = sp.Rational(twice_index, 2)
                acted_middle = act_ns_mode(
                    NSG(mode_index),
                    {middle: sp.S.One},
                    h=h_ns,
                    c=c,
                )
                value = _add_ns_acted_terms(
                    value,
                    sp.binomial(
                        sp.Rational(2 * n + 1, 2),
                        sp.Rational(twice_index + 1, 2),
                    ),
                    acted_middle,
                    evaluate,
                    rest,
                    right,
                )
            return sp.expand(value)

        if right.word:
            leading = right.word[0]
            rest = PBWState(right.word[1:], right.ground_parity)
            if leading.kind == "L":
                n = -leading.index
                acted_left = act_r_mode(
                    RL(n),
                    {left: sp.S.One},
                    h=h_infinity,
                    c=c,
                )
                value = _add_r_acted_terms(
                    sp.S.Zero,
                    1,
                    acted_left,
                    evaluate,
                    left,
                    middle,
                    rest,
                    acted_slot="infinity",
                )
                maximum_mode = (
                    ns_twice_level(middle) // 2 if middle else 0
                )
                for mode_index in range(-1, maximum_mode + 1):
                    acted_middle = act_ns_mode(
                        NSL(mode_index),
                        {middle: sp.S.One},
                        h=h_ns,
                        c=c,
                    )
                    value = _add_ns_acted_terms(
                        value,
                        -sp.binomial(1 - n, mode_index + 1),
                        acted_middle,
                        evaluate,
                        left,
                        rest,
                    )
                return sp.expand(value)

            n = -leading.index
            sign = sp.Integer(-1) ** (left.parity + rest.parity + 1)
            acted_left = act_r_mode(
                RG(n),
                {left: sp.S.One},
                h=h_infinity,
                c=c,
            )
            value = _add_r_acted_terms(
                sp.S.Zero,
                sign,
                acted_left,
                evaluate,
                left,
                middle,
                rest,
                acted_slot="infinity",
            )
            middle_twice_level = ns_twice_level(middle) if middle else 0
            for twice_index in range(-1, middle_twice_level + 1, 2):
                mode_index = sp.Rational(twice_index, 2)
                acted_middle = act_ns_mode(
                    NSG(mode_index),
                    {middle: sp.S.One},
                    h=h_ns,
                    c=c,
                )
                value = _add_ns_acted_terms(
                    value,
                    -sign
                    * sp.binomial(
                        sp.Rational(1 - 2 * n, 2),
                        sp.Rational(twice_index + 1, 2),
                    ),
                    acted_middle,
                    evaluate,
                    left,
                    rest,
                )
            return sp.expand(value)

        return rr_middle_ns_word_polynomial(
            middle,
            h_ns=h_ns,
            h_infinity=h_infinity,
            h_zero=h_zero,
            c=c,
            ground_tensor=terminal,
            infinity_ground_parity=left.ground_parity,
            zero_ground_parity=right.ground_parity,
            simplify=simplify,
        )

    value = evaluate(infinity_state, middle_word, zero_state)
    return sp.factor(sp.cancel(value)) if simplify else value


def rr_three_point_polynomial(
    infinity_state: PBWState,
    middle_word: NSWord,
    zero_state: PBWState,
    *,
    h_ns: sp.Expr,
    beta_infinity: sp.Expr,
    beta_zero: sp.Expr,
    c: sp.Expr,
    structure_sign: int,
) -> sp.Expr:
    r"""Evaluate the HJS-normalized tensor in the polynomial R basis.

    The HJS ground package is first converted with the anti-linear
    slot-1 factor.  :func:`rr_three_point_from_ground_tensor` then extends
    those four polynomial terminal values without introducing any further
    momentum phases.
    """

    _validate_structure_sign(structure_sign)
    beta_infinity = sp.sympify(beta_infinity)
    beta_zero = sp.sympify(beta_zero)
    c = sp.sympify(c)
    return rr_three_point_from_ground_tensor(
        infinity_state,
        middle_word,
        zero_state,
        h_infinity=sp.expand(c / 24 - beta_infinity**2),
        h_ns=h_ns,
        h_zero=sp.expand(c / 24 - beta_zero**2),
        c=c,
        ground_tensor=hjs_to_polynomial_ground_tensor(
            beta_infinity,
            beta_zero,
            structure_sign=structure_sign,
        ),
    )


def rr_three_point_hjs(
    infinity_state: PBWState,
    middle_word: NSWord,
    zero_state: PBWState,
    *,
    h_ns: sp.Expr,
    beta_infinity: sp.Expr,
    beta_zero: sp.Expr,
    c: sp.Expr,
    structure_sign: int,
) -> sp.Expr:
    r"""Evaluate the ordered descendant tensor in the HJS phase basis.

    The oscillator words are the repository PBW words, while the
    ``ground_parity`` labels select HJS ``w^+`` or ``w^-``.  The
    anti-linear slot-1 basis conversion is included.  At ``beta=0`` an
    odd HJS ground state is not a regular basis vector; use the polynomial
    evaluator and handle the shortened module explicitly instead.
    """

    _validate_structure_sign(structure_sign)
    _validate_inputs(infinity_state, middle_word, zero_state)
    beta_infinity = sp.sympify(beta_infinity)
    beta_zero = sp.sympify(beta_zero)
    polynomial = rr_three_point_polynomial(
        infinity_state,
        middle_word,
        zero_state,
        h_ns=h_ns,
        beta_infinity=beta_infinity,
        beta_zero=beta_zero,
        c=c,
        structure_sign=structure_sign,
    )
    conversion = _hjs_from_polynomial_tensor_factor(
        infinity_state.ground_parity,
        zero_state.ground_parity,
        beta_infinity,
        beta_zero,
    )
    return sp.factor(sp.cancel(conversion * polynomial))


def rr_level_tensor(
    infinity_twice_level: int,
    middle_twice_level: int,
    zero_twice_level: int,
    *,
    h_ns: sp.Expr,
    beta_infinity: sp.Expr,
    beta_zero: sp.Expr,
    c: sp.Expr,
    structure_sign: int,
    ground_basis: GroundBasis = "polynomial",
) -> tuple[
    tuple[PBWState, ...],
    tuple[NSWord, ...],
    tuple[PBWState, ...],
    dict[TensorKey, sp.Expr],
]:
    """Build every ordered tensor entry at three fixed twice-levels.

    The returned tuple contains the infinity, middle, and zero PBW bases,
    followed by a dictionary keyed by the corresponding basis-state
    triples.  This is the direct finite tensor needed by low-level
    theta/glasses sewing contractions.
    """

    _validate_structure_sign(structure_sign)
    _validate_ground_basis(ground_basis)
    for name, level in (
        ("infinity_twice_level", infinity_twice_level),
        ("middle_twice_level", middle_twice_level),
        ("zero_twice_level", zero_twice_level),
    ):
        if not isinstance(level, int):
            raise TypeError(f"{name} must be an integer")
        if level < 0:
            raise ValueError(f"{name} must be nonnegative")
    if infinity_twice_level % 2 or zero_twice_level % 2:
        raise ValueError("Ramond twice-levels must be even")

    infinity_basis = tuple(r_pbw_basis(infinity_twice_level))
    middle_basis = tuple(ns_pbw_basis(middle_twice_level))
    zero_basis = tuple(r_pbw_basis(zero_twice_level))
    evaluator = (
        rr_three_point_hjs
        if ground_basis == "hjs"
        else rr_three_point_polynomial
    )
    entries: dict[TensorKey, sp.Expr] = {}
    for infinity_state in infinity_basis:
        for middle_word in middle_basis:
            for zero_state in zero_basis:
                entries[(infinity_state, middle_word, zero_state)] = evaluator(
                    infinity_state,
                    middle_word,
                    zero_state,
                    h_ns=h_ns,
                    beta_infinity=beta_infinity,
                    beta_zero=beta_zero,
                    c=c,
                    structure_sign=structure_sign,
                )
    return infinity_basis, middle_basis, zero_basis, entries


def rr_level_tensor_from_ground_tensor(
    infinity_twice_level: int,
    middle_twice_level: int,
    zero_twice_level: int,
    *,
    h_infinity: sp.Expr,
    h_ns: sp.Expr,
    h_zero: sp.Expr,
    c: sp.Expr,
    ground_tensor: Sequence[Sequence[sp.Expr]] | sp.MatrixBase,
) -> tuple[
    tuple[PBWState, ...],
    tuple[NSWord, ...],
    tuple[PBWState, ...],
    dict[TensorKey, sp.Expr],
]:
    """Build a fixed-level tensor from arbitrary polynomial ground data."""

    for name, level in (
        ("infinity_twice_level", infinity_twice_level),
        ("middle_twice_level", middle_twice_level),
        ("zero_twice_level", zero_twice_level),
    ):
        if not isinstance(level, int):
            raise TypeError(f"{name} must be an integer")
        if level < 0:
            raise ValueError(f"{name} must be nonnegative")
    if infinity_twice_level % 2 or zero_twice_level % 2:
        raise ValueError("Ramond twice-levels must be even")
    terminal = sp.Matrix(ground_tensor)
    if terminal.shape != (2, 2):
        raise ValueError("ground_tensor must be a two-by-two matrix")

    infinity_basis = tuple(r_pbw_basis(infinity_twice_level))
    middle_basis = tuple(ns_pbw_basis(middle_twice_level))
    zero_basis = tuple(r_pbw_basis(zero_twice_level))
    entries: dict[TensorKey, sp.Expr] = {}
    for infinity_state in infinity_basis:
        for middle_word in middle_basis:
            for zero_state in zero_basis:
                entries[(infinity_state, middle_word, zero_state)] = (
                    rr_three_point_from_ground_tensor(
                        infinity_state,
                        middle_word,
                        zero_state,
                        h_infinity=h_infinity,
                        h_ns=h_ns,
                        h_zero=h_zero,
                        c=c,
                        ground_tensor=terminal,
                    )
                )
    return infinity_basis, middle_basis, zero_basis, entries


__all__ = [
    "GroundBasis",
    "TensorKey",
    "hjs_polynomial_form_coordinates",
    "hjs_ground_tensor",
    "hjs_to_polynomial_ground_tensor",
    "polynomial_ground_form_basis",
    "rr_level_tensor",
    "rr_level_tensor_from_ground_tensor",
    "rr_three_point_from_ground_tensor",
    "rr_three_point_hjs",
    "rr_three_point_polynomial",
]
