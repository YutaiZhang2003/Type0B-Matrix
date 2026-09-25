#!/usr/bin/env python3
"""Direct middle-slot NS-null restriction with Ramond spectators.

The ordered form is

    rho_RR(w^+_{beta_infinity}, xi_NS, w^+_{beta_zero})

with slots ``(infinity, 1, 0)``.  HJS do not give a cyclic formula for a
null in this middle NS slot.  This module evaluates it from their RR Ward
identity, using Xi Yin's ordinary central charge and the exact NS
Gram-kernel singular vector.

The HJS Ramond ground convention is retained locally:

    G_0 w^+_beta = exp(i*pi/4) beta w^-_beta,
    G_0 w^-_beta = exp(3*i*pi/4) beta w^+_beta.

The HJS three-point form is anti-linear in its first slot.  In their
analytic Ramond-momentum convention this changes the effective first-slot
coefficients to ``exp(3*i*pi/4)*beta`` and
``exp(i*pi/4)*beta``, respectively.  Treating the first slot as an
ordinary linear slot rotates the two published HJS structures and is not
the convention implemented here.

The public ``structure_sign`` is ``+1`` or ``-1`` for the published
``rho_RR^(+)`` and ``rho_RR^(-)`` structures.  Ramond momenta, rather than
weights alone, are required because changing ``beta`` to ``-beta`` exchanges
these two structures.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Mapping, Sequence

import sympy as sp

from ns_algebra.ns_fusion import c_xi, h_rs
from ns_algebra.ns_sca import G, L, Mode, Word, act_mode, gram_matrix, twice_level


_NULL_B = sp.Symbol("_nrr_middle_null_b", nonzero=True)


def _validate_label(r: int, s: int) -> None:
    """Require a positive NS Kac pair with even ``r+s``."""

    if not isinstance(r, int) or not isinstance(s, int):
        raise TypeError("r and s must be integers")
    if r <= 0 or s <= 0:
        raise ValueError("r and s must be positive")
    if (r + s) % 2:
        raise ValueError("an NS singular vector requires even r+s")


def _validate_structure_sign(structure_sign: int) -> None:
    """Require ``+1`` or ``-1`` for the HJS trilinear structure."""

    if structure_sign not in (-1, 1):
        raise ValueError("structure_sign must be +1 or -1")


def _validate_parity(parity: int, name: str) -> None:
    """Require one of the two fermion parities."""

    if parity not in (0, 1):
        raise ValueError(f"{name} must be zero or one")


def _ground_form_value(
    infinity_parity: int,
    zero_parity: int,
    structure_sign: int,
) -> sp.Expr:
    """Return the normalized HJS RR form on two ground states.

    The even structures obey

    ``rho_e^(s)(w+,nu,w+)=1`` and
    ``rho_e^(s)(w-,nu,w-)=s``.

    Their odd partners obey

    ``rho_o^(s)(w+,nu,w-)=1`` and
    ``rho_o^(s)(w-,nu,w+)=s*i``.
    """

    _validate_parity(infinity_parity, "infinity_parity")
    _validate_parity(zero_parity, "zero_parity")
    _validate_structure_sign(structure_sign)
    if infinity_parity == 0:
        return sp.S.One
    if zero_parity == 0:
        return sp.I * structure_sign
    return sp.Integer(structure_sign)


def _hjs_zero_mode_action(
    parity: int,
    beta: sp.Expr,
    *,
    first_slot: bool,
) -> tuple[int, sp.Expr]:
    """Return the toggled parity and effective HJS ``G_0`` coefficient.

    The state action on a right or middle slot is

    ``G_0 w^+ = exp(i*pi/4) beta w^-`` and
    ``G_0 w^- = exp(3*i*pi/4) beta w^+``.

    HJS define the first slot anti-linearly.  On the physical Ramond line
    ``conjugate(beta)=-beta``; analytic continuation in their published
    ``beta`` variable therefore interchanges these two effective
    coefficients in the first slot.
    """

    if parity not in (0, 1):
        raise ValueError("Ramond ground parity must be zero or one")
    if not isinstance(first_slot, bool):
        raise TypeError("first_slot must be a boolean")
    beta = sp.sympify(beta)
    phase = (1 + sp.I) / sp.sqrt(2)
    if first_slot:
        coefficient = (sp.I * phase if parity == 0 else phase) * beta
    else:
        coefficient = (phase if parity == 0 else sp.I * phase) * beta
    return parity ^ 1, coefficient


def _add_acted_middle_terms(
    value: sp.Expr,
    coefficient: sp.Expr,
    acted: Mapping[Word, sp.Expr],
    evaluate,
    infinity_parity: int,
    zero_parity: int,
) -> sp.Expr:
    """Accumulate an NS mode action in the middle Ward sum."""

    for word, state_coefficient in acted.items():
        value += (
            sp.sympify(coefficient)
            * state_coefficient
            * evaluate(word, infinity_parity, zero_parity)
        )
    return sp.expand(value)


def rr_middle_ns_word(
    word: Word,
    *,
    h_ns: sp.Expr,
    beta_infinity: sp.Expr,
    beta_zero: sp.Expr,
    c: sp.Expr,
    structure_sign: int,
    infinity_ground_parity: int = 0,
    zero_ground_parity: int = 0,
) -> sp.Expr:
    r"""Evaluate one ordered RR three-point form exactly.

    The returned quantity is the normalized HJS structure appropriate to
    the parity of ``word``:

    .. math::

       \rho_{\rm RR}^{(\pm)}
       (w^a_{\beta_\infty},\,{\tt word}\,\nu_{h_{\rm NS}},
        w^b_{\beta_0}).

    The optional ground parities ``a`` and ``b`` are zero for \(w^+\)
    and one for \(w^-\).  Their defaults retain the original
    \(w^+,w^+\) API.
    """

    _validate_structure_sign(structure_sign)
    _validate_parity(infinity_ground_parity, "infinity_ground_parity")
    _validate_parity(zero_ground_parity, "zero_ground_parity")
    word = tuple(word)
    if word and any(mode.index >= 0 for mode in word):
        raise ValueError("the middle word must contain negative NS modes")
    h_ns = sp.sympify(h_ns)
    beta_infinity = sp.sympify(beta_infinity)
    beta_zero = sp.sympify(beta_zero)
    c = sp.sympify(c)
    h_infinity = c / 24 - beta_infinity**2
    h_zero = c / 24 - beta_zero**2

    @lru_cache(maxsize=None)
    def evaluate(
        middle_word: Word,
        infinity_parity: int,
        zero_parity: int,
    ) -> sp.Expr:
        """Reduce a middle descendant using the ordered HJS RR identity."""

        if not middle_word:
            return _ground_form_value(
                infinity_parity,
                zero_parity,
                structure_sign,
            )

        leading = middle_word[0]
        rest = middle_word[1:]
        if leading.kind == "L":
            n = -int(leading.index)
            rest_level = sp.Rational(twice_level(rest), 2) if rest else 0
            factor = (-1) ** n * (
                n * h_zero + h_ns + rest_level - h_infinity
            )
            return sp.expand(
                factor * evaluate(rest, infinity_parity, zero_parity)
            )

        k = -sp.Rational(leading.index)
        if k <= 0 or k.q != 2:
            raise ValueError("a middle NS G mode must have negative half index")

        value = sp.S.Zero
        if k == sp.Rational(1, 2):
            toggled, coefficient = _hjs_zero_mode_action(
                infinity_parity,
                beta_infinity,
                first_slot=True,
            )
            value += coefficient * evaluate(
                rest,
                toggled,
                zero_parity,
            )

        toggled, coefficient = _hjs_zero_mode_action(
            zero_parity,
            beta_zero,
            first_slot=False,
        )
        right_sign_exponent = int(
            infinity_parity
            + zero_parity
            + sp.Rational(1, 2)
            - k
        )
        value += (
            sp.Integer(-1) ** right_sign_exponent
            * coefficient
            * evaluate(rest, infinity_parity, toggled)
        )

        target_level = k + (
            sp.Rational(twice_level(rest), 2) if rest else sp.S.Zero
        )
        for p in range(1, int(sp.floor(target_level)) + 1):
            acted = act_mode(
                G(sp.Integer(p) - k),
                {rest: sp.S.One},
                h=h_ns,
                c=c,
            )
            value = _add_acted_middle_terms(
                value,
                -sp.binomial(sp.Rational(1, 2), p),
                acted,
                evaluate,
                infinity_parity,
                zero_parity,
            )
        return sp.expand(value)

    return sp.factor(
        evaluate(
            word,
            infinity_ground_parity,
            zero_ground_parity,
        )
    )


def _polynomial_zero_mode_action(
    parity: int,
    h: sp.Expr,
    c: sp.Expr,
) -> tuple[int, sp.Expr]:
    """Return the polynomial-doublet action of ``G_0``."""

    _validate_parity(parity, "Ramond ground parity")
    if parity == 0:
        return 1, sp.S.One
    return 0, sp.sympify(h) - sp.sympify(c) / 24


def rr_middle_ns_word_polynomial(
    word: Word,
    *,
    h_ns: sp.Expr,
    h_infinity: sp.Expr,
    h_zero: sp.Expr,
    c: sp.Expr,
    ground_tensor: Sequence[Sequence[sp.Expr]] | sp.MatrixBase,
    infinity_ground_parity: int = 0,
    zero_ground_parity: int = 0,
    simplify: bool = True,
) -> sp.Expr:
    r"""Evaluate a middle NS word from arbitrary polynomial ground data.

    ``ground_tensor[p_infinity,p_zero]`` specifies the four terminal
    values in the polynomial Ramond ground doublets.  Unlike an HJS
    structure, these entries may be chosen independently.  The Ward
    recursion then has coefficients rational in the fixed weights and
    ``c``; no Ramond momentum square root is introduced.

    ``simplify=False`` skips the final symbolic factorization.  It returns
    the same exact expression in expanded form and is intended for numerical
    compilation, where factoring can dominate the runtime at higher level.
    """

    _validate_parity(infinity_ground_parity, "infinity_ground_parity")
    _validate_parity(zero_ground_parity, "zero_ground_parity")
    word = tuple(word)
    if word and any(mode.index >= 0 for mode in word):
        raise ValueError("the middle word must contain negative NS modes")
    terminal = sp.Matrix(ground_tensor)
    if terminal.shape != (2, 2):
        raise ValueError("ground_tensor must be a two-by-two matrix")
    terminal_values = tuple(
        tuple(sp.sympify(terminal[row, column]) for column in range(2))
        for row in range(2)
    )
    h_ns = sp.sympify(h_ns)
    h_infinity = sp.sympify(h_infinity)
    h_zero = sp.sympify(h_zero)
    c = sp.sympify(c)

    @lru_cache(maxsize=None)
    def evaluate(
        middle_word: Word,
        infinity_parity: int,
        zero_parity: int,
    ) -> sp.Expr:
        """Reduce one middle word using polynomial ``G_0`` actions."""

        if not middle_word:
            return terminal_values[infinity_parity][zero_parity]

        leading = middle_word[0]
        rest = middle_word[1:]
        if leading.kind == "L":
            n = -int(leading.index)
            rest_level = sp.Rational(twice_level(rest), 2) if rest else 0
            factor = (-1) ** n * (
                n * h_zero + h_ns + rest_level - h_infinity
            )
            return sp.expand(
                factor * evaluate(rest, infinity_parity, zero_parity)
            )

        k = -sp.Rational(leading.index)
        if k <= 0 or k.q != 2:
            raise ValueError("a middle NS G mode must have negative half index")

        value = sp.S.Zero
        if k == sp.Rational(1, 2):
            toggled, coefficient = _polynomial_zero_mode_action(
                infinity_parity,
                h_infinity,
                c,
            )
            value += coefficient * evaluate(rest, toggled, zero_parity)

        toggled, coefficient = _polynomial_zero_mode_action(
            zero_parity,
            h_zero,
            c,
        )
        right_sign_exponent = int(
            infinity_parity
            + zero_parity
            + sp.Rational(1, 2)
            - k
        )
        value += (
            sp.Integer(-1) ** right_sign_exponent
            * coefficient
            * evaluate(rest, infinity_parity, toggled)
        )

        target_level = k + (
            sp.Rational(twice_level(rest), 2) if rest else sp.S.Zero
        )
        for p in range(1, int(sp.floor(target_level)) + 1):
            acted = act_mode(
                G(sp.Integer(p) - k),
                {rest: sp.S.One},
                h=h_ns,
                c=c,
            )
            value = _add_acted_middle_terms(
                value,
                -sp.binomial(sp.Rational(1, 2), p),
                acted,
                evaluate,
                infinity_parity,
                zero_parity,
            )
        return sp.expand(value)

    value = evaluate(
        word,
        infinity_ground_parity,
        zero_ground_parity,
    )
    return sp.factor(value) if simplify else value


@lru_cache(maxsize=None)
def _normalized_ns_null_template(
    r: int,
    s: int,
) -> tuple[tuple[Word, sp.Expr], ...]:
    """Return the repository-normalized NS singular vector at generic ``b``."""

    _validate_label(r, s)
    twice_null_level = r * s
    basis, matrix = gram_matrix(
        twice_null_level,
        h=h_rs(r, s, _NULL_B),
        c=c_xi(_NULL_B),
    )
    kernels = matrix.applyfunc(sp.factor).nullspace()
    if len(kernels) != 1:
        raise ValueError(
            f"expected one primitive ({r},{s}) NS null direction, "
            f"found {len(kernels)}"
        )

    target_word: Word = (L(-1),) * (twice_null_level // 2)
    if twice_null_level % 2:
        target_word += (G(sp.Rational(-1, 2)),)
    target_index = basis.index(target_word)
    target_coefficient = sp.factor(kernels[0][target_index])
    if target_coefficient == 0:
        raise ValueError("the NS null has zero repository-normalization term")
    return tuple(
        (word, sp.factor(coefficient / target_coefficient))
        for word, coefficient in zip(basis, kernels[0])
        if coefficient != 0
    )


def rr_middle_ns_null_restriction(
    r: int,
    s: int,
    beta_infinity: sp.Expr,
    beta_zero: sp.Expr,
    b: sp.Expr,
    *,
    structure_sign: int,
) -> sp.Expr:
    r"""Return the direct middle-slot NS singular-vector coefficient.

    This evaluates

    .. math::

       \rho_{\rm RR}^{(\pm)}
       (w^+_{\beta_\infty},\chi_{r,s},w^+_{\beta_0})

    in slots ``(infinity,1,0)``.  For odd ``rs`` the shifted HJS
    structure is opposite, with the descendant-parity phase returned by
    :func:`rr_middle_odd_null_transport`.
    """

    _validate_label(r, s)
    _validate_structure_sign(structure_sign)
    b = sp.sympify(b)
    central_charge = c_xi(b)
    singular_weight = h_rs(r, s, b)
    value = sp.S.Zero
    for word, coefficient in _normalized_ns_null_template(r, s):
        specialized = coefficient.subs(_NULL_B, b)
        value += specialized * rr_middle_ns_word(
            word,
            h_ns=singular_weight,
            beta_infinity=beta_infinity,
            beta_zero=beta_zero,
            c=central_charge,
            structure_sign=structure_sign,
        )
    return sp.factor(sp.cancel(value))


def rr_middle_ns_null_polynomial_map(
    r: int,
    s: int,
    h_infinity: sp.Expr,
    h_zero: sp.Expr,
    b: sp.Expr,
    *,
    parent_form_parity: int,
) -> sp.ImmutableMatrix:
    r"""Return the two-by-two polynomial form map induced by an NS null.

    Rows label the two parent ground coordinates and columns label the
    two shifted coordinates.  Even coordinates are ordered
    ``(+,+),(-,-)`` and odd coordinates ``(+,-),(-,+)``.  The shifted
    form parity is ``parent_form_parity xor (r*s mod 2)``.
    """

    _validate_label(r, s)
    _validate_parity(parent_form_parity, "parent_form_parity")
    h_infinity = sp.sympify(h_infinity)
    h_zero = sp.sympify(h_zero)
    b = sp.sympify(b)
    if b == 0:
        raise ValueError("b must be nonzero")
    central_charge = c_xi(b)
    singular_weight = h_rs(r, s, b)
    shifted_form_parity = parent_form_parity ^ ((r * s) % 2)

    def coordinate_pairs(form_parity: int) -> tuple[tuple[int, int], ...]:
        return (
            ((0, 0), (1, 1))
            if form_parity == 0
            else ((0, 1), (1, 0))
        )

    parent_pairs = coordinate_pairs(parent_form_parity)
    shifted_pairs = coordinate_pairs(shifted_form_parity)
    result = sp.zeros(2)
    for parent_coordinate, (parent_left, parent_right) in enumerate(
        parent_pairs
    ):
        terminal = sp.zeros(2)
        terminal[parent_left, parent_right] = 1
        for shifted_coordinate, (
            shifted_left,
            shifted_right,
        ) in enumerate(shifted_pairs):
            value = sp.S.Zero
            for word, template_coefficient in _normalized_ns_null_template(
                r,
                s,
            ):
                value += template_coefficient.subs(_NULL_B, b) * (
                    rr_middle_ns_word_polynomial(
                        word,
                        h_ns=singular_weight,
                        h_infinity=h_infinity,
                        h_zero=h_zero,
                        c=central_charge,
                        ground_tensor=terminal,
                        infinity_ground_parity=shifted_left,
                        zero_ground_parity=shifted_right,
                    )
                )
            result[parent_coordinate, shifted_coordinate] = sp.factor(
                sp.cancel(value)
            )
    return sp.ImmutableMatrix(result)


def rr_middle_ns_null_endpoint(
    r: int,
    s: int,
    shifted_component: int,
    beta_infinity: sp.Expr,
    beta_zero: sp.Expr,
    b: sp.Expr,
    *,
    structure_sign: int,
) -> sp.Expr:
    r"""Return the component-normalized middle NS-null endpoint.

    Let ``epsilon=r*s mod 2`` and let ``shifted_component`` be the parity
    of the trilinear form after the singular vector is stripped.  The
    endpoint coefficient is normalized by evaluating that shifted form
    on

    .. math::

       (w^{\,\texttt{shifted_component}}_{\beta_\infty},
        \nu_{h_{r,s}+rs/2},
        w^+_{\beta_0})

    and dividing by its HJS ground terminal.  This is the direct
    \(R\)-NS-\(R\) analogue of
    :func:`ns_middle_fusion.middle_slot_restriction`.

    For an odd null, component zero returns the raw \(w^+,w^+\)
    coefficient \(C_s\), while component one returns \(-iC_s\).  The
    shifted HJS structure is \(-s\).  For an even null the structure and
    endpoint coefficient are both preserved.
    """

    _validate_label(r, s)
    _validate_parity(shifted_component, "shifted_component")
    _validate_structure_sign(structure_sign)
    b = sp.sympify(b)
    central_charge = c_xi(b)
    singular_weight = h_rs(r, s, b)
    raw_value = sp.S.Zero
    for word, coefficient in _normalized_ns_null_template(r, s):
        specialized = coefficient.subs(_NULL_B, b)
        raw_value += specialized * rr_middle_ns_word(
            word,
            h_ns=singular_weight,
            beta_infinity=beta_infinity,
            beta_zero=beta_zero,
            c=central_charge,
            structure_sign=structure_sign,
            infinity_ground_parity=shifted_component,
        )

    shifted_structure = (
        -structure_sign if (r * s) % 2 else structure_sign
    )
    shifted_terminal = _ground_form_value(
        shifted_component,
        0,
        shifted_structure,
    )
    return sp.factor(sp.cancel(raw_value / shifted_terminal))


def rr_middle_odd_null_transport(
    structure_sign: int,
    descendant_parity: int,
) -> tuple[int, sp.Expr]:
    r"""Return the shifted HJS structure and phase for an odd NS null.

    If ``chi`` is an odd NS singular vector and ``X`` has fermion parity
    ``p``, the ordered middle-slot factorization is

    .. math::

       \rho_{\rm RR}^{(s)}(w^+,X\chi,w^+)
       =C_s\,(-i)^p\,
        \rho_{\rm RR}^{(-s)}(w^+,X\nu_{h+N},w^+),
       \qquad p=0,1.

    Thus the HJS structure always flips.  The phase is one for an even
    shifted descendant and ``-i`` for an odd shifted descendant; it does
    not depend on the Kac labels.
    """

    _validate_structure_sign(structure_sign)
    if descendant_parity not in (0, 1):
        raise ValueError("descendant_parity must be zero or one")
    phase = sp.S.One if descendant_parity == 0 else -sp.I
    return -structure_sign, phase


__all__ = [
    "rr_middle_ns_null_endpoint",
    "rr_middle_ns_null_polynomial_map",
    "rr_middle_odd_null_transport",
    "rr_middle_ns_null_restriction",
    "rr_middle_ns_word",
    "rr_middle_ns_word_polynomial",
]
