#!/usr/bin/env python3
"""Exact symbolic algebra for a generic Ramond super-Virasoro module.

The algebra uses Xi Yin's ordinary Virasoro central charge:

    [L_m,L_n] = (m-n)L_{m+n}
        + c m(m^2-1) delta_{m+n,0}/12,
    [L_m,G_n] = (m/2-n)G_{m+n},
    {G_m,G_n} = 2L_{m+n}
        + c(4m^2-1) delta_{m+n,0}/12.

All Ramond mode indices are ordinary integers.  Descendant levels are
nevertheless exposed as integer twice-levels, consistently with the NS
modules in this repository.

The two ground states use the polynomial Hadasz--Jaskolski--Suchanek basis

    G_0 |h,+> = |h,->,
    G_0 |h,-> = (h-c/24)|h,+>.

It is equivalent, away from ``h=c/24``, to the symmetric square-root basis
used in the TeX notes.  This basis keeps exact Gram matrices polynomial in
``h`` and ``c`` while retaining the full zero-mode action.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from typing import Iterable, Literal, Mapping

import sympy as sp


ModeKind = Literal["L", "G"]


@dataclass(frozen=True)
class Mode:
    """One Ramond super-Virasoro mode with an ordinary integer index."""

    kind: ModeKind
    index: int

    def __post_init__(self) -> None:
        if self.kind not in ("L", "G"):
            raise ValueError("mode kind must be 'L' or 'G'")
        if not isinstance(self.index, int):
            raise TypeError("a Ramond mode index must be an integer")

    @property
    def parity(self) -> int:
        """Return 0 for a bosonic L mode and 1 for a fermionic G mode."""

        return 0 if self.kind == "L" else 1

    def __str__(self) -> str:
        return f"{self.kind}_{self.index}"


Word = tuple[Mode, ...]
WordState = dict[Word, sp.Expr]


@dataclass(frozen=True)
class PBWState:
    """A negative-mode PBW word together with its ground-state parity."""

    word: Word = ()
    ground_parity: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "word", tuple(self.word))
        if self.ground_parity not in (0, 1):
            raise ValueError("ground_parity must be 0 (|h,+>) or 1 (|h,->)")
        if any(not isinstance(mode, Mode) for mode in self.word):
            raise TypeError("every word entry must be a Ramond Mode")

    @property
    def parity(self) -> int:
        """Return the total fermion parity of the descendant."""

        return self.ground_parity ^ fermion_parity(self.word)

    @property
    def twice_descendant_level(self) -> int:
        """Return twice the oscillator level of this state."""

        return twice_level(self.word)


State = dict[PBWState, sp.Expr]


def L(index: int) -> Mode:
    """Construct the Ramond Virasoro mode ``L_index``."""

    return Mode("L", index)


def G(index: int) -> Mode:
    """Construct the integer-moded Ramond supercurrent mode ``G_index``."""

    return Mode("G", index)


def ground_state(parity: int = 0) -> PBWState:
    """Return ``|h,+>`` for parity 0 or ``|h,->`` for parity 1."""

    return PBWState((), parity)


def twice_level(word: Word) -> int:
    """Return twice the descendant level of a negative Ramond word."""

    if any(mode.index >= 0 for mode in word):
        raise ValueError("descendant levels are defined for negative-mode words")
    return -2 * sum(mode.index for mode in word)


def fermion_parity(word: Word) -> int:
    """Return the number of G modes in ``word`` modulo two."""

    return sum(mode.parity for mode in word) % 2


def format_word(word: Word) -> str:
    """Format an oscillator word without choosing a ground-state label."""

    if not word:
        return "1"
    return " ".join(str(mode) for mode in word)


def format_basis_state(state: PBWState) -> str:
    """Format a PBW state as an oscillator word acting on ``|h,+/- >``."""

    ground_label = "+" if state.ground_parity == 0 else "-"
    if not state.word:
        return f"|h,{ground_label}>"
    return f"{format_word(state.word)} |h,{ground_label}>"


def ground_action_matrix(h: sp.Expr, c: sp.Expr) -> sp.Matrix:
    """Return the matrix of G_0 in the ordered basis ``(|h,+>,|h,->)``.

    Columns label input states.  Thus the lower-left entry implements
    ``G_0|h,+>=|h,->``.
    """

    delta = sp.sympify(h) - sp.sympify(c) / 24
    return sp.Matrix([[0, delta], [1, 0]])


def ground_gram_matrix(h: sp.Expr, c: sp.Expr) -> sp.Matrix:
    """Return the contravariant ground Gram matrix ``diag(1,h-c/24)``."""

    delta = sp.sympify(h) - sp.sympify(c) / 24
    return sp.diag(1, delta)


def ground_inner_product(
    left_parity: int,
    right_parity: int,
    *,
    h: sp.Expr,
    c: sp.Expr,
) -> sp.Expr:
    """Pair two ground states in the polynomial doublet basis."""

    if left_parity not in (0, 1) or right_parity not in (0, 1):
        raise ValueError("ground parities must be 0 or 1")
    if left_parity != right_parity:
        return sp.S.Zero
    if left_parity == 0:
        return sp.S.One
    return sp.expand(sp.sympify(h) - sp.sympify(c) / 24)


def _mode_order_key(mode: Mode) -> tuple[int, int]:
    # Negative PBW words have all L modes before all G modes.  Within each
    # kind, the more negative mode is placed first.
    return (0 if mode.kind == "L" else 1, mode.index)


def _word_order_key(word: Word) -> tuple[tuple[int, int], ...]:
    return tuple(_mode_order_key(mode) for mode in word)


def _basis_state_order_key(
    basis_state: PBWState,
) -> tuple[tuple[tuple[int, int], ...], int]:
    return (_word_order_key(basis_state.word), basis_state.ground_parity)


def is_pbw_word(word: Word) -> bool:
    """Return whether ``word`` obeys the negative Ramond PBW convention."""

    if any(mode.index >= 0 for mode in word):
        return False
    for left, right in zip(word, word[1:]):
        if _mode_order_key(left) > _mode_order_key(right):
            return False
        if left.kind == right.kind == "G" and left == right:
            return False
    return True


@lru_cache(maxsize=None)
def _integer_partitions(total: int, maximum: int) -> tuple[tuple[int, ...], ...]:
    if total < 0:
        return ()
    if total == 0:
        return ((),)
    out: list[tuple[int, ...]] = []
    for first in range(min(total, maximum), 0, -1):
        for tail in _integer_partitions(total - first, first):
            out.append((first,) + tail)
    return tuple(out)


def _fermion_subsets(maximum_sum: int) -> Iterable[tuple[int, ...]]:
    positive_indices = tuple(range(1, maximum_sum + 1))
    for size in range(len(positive_indices) + 1):
        for subset in combinations(positive_indices, size):
            if sum(subset) <= maximum_sum:
                yield subset


@lru_cache(maxsize=None)
def _pbw_words_cached(descendant_level: int) -> tuple[Word, ...]:
    if descendant_level < 0:
        return ()
    out: list[Word] = []
    for fermion_indices in _fermion_subsets(descendant_level):
        remaining = descendant_level - sum(fermion_indices)
        for partition in _integer_partitions(remaining, remaining):
            l_modes = tuple(L(-part) for part in partition)
            g_modes = tuple(G(-part) for part in reversed(fermion_indices))
            out.append(l_modes + g_modes)
    return tuple(out)


@lru_cache(maxsize=None)
def _pbw_basis_cached(twice_descendant_level: int) -> tuple[PBWState, ...]:
    if twice_descendant_level < 0 or twice_descendant_level % 2:
        return ()
    words = _pbw_words_cached(twice_descendant_level // 2)
    return tuple(
        PBWState(word, ground_parity)
        for word in words
        for ground_parity in (0, 1)
    )


def pbw_basis(
    twice_descendant_level: int,
    *,
    parity: int | None = None,
) -> list[PBWState]:
    """Generate the Ramond PBW basis at one exact twice-level.

    The generic ground doublet gives both total parities at every integer
    level.  An odd twice-level has no Ramond descendants and returns an
    empty list.
    """

    if not isinstance(twice_descendant_level, int):
        raise TypeError("twice_descendant_level must be an integer")
    if twice_descendant_level < 0:
        raise ValueError("twice_descendant_level must be nonnegative")
    if parity not in (None, 0, 1):
        raise ValueError("parity must be None, 0, or 1")
    basis = list(_pbw_basis_cached(twice_descendant_level))
    if parity is not None:
        basis = [basis_state for basis_state in basis if basis_state.parity == parity]
    return basis


def pbw_bases(
    maximum_twice_level: int,
    *,
    parity: int | None = None,
) -> dict[int, list[PBWState]]:
    """Generate Ramond PBW bases through an inclusive twice-level cutoff."""

    if not isinstance(maximum_twice_level, int):
        raise TypeError("maximum_twice_level must be an integer")
    if maximum_twice_level < 0:
        raise ValueError("maximum_twice_level must be nonnegative")
    return {
        level: pbw_basis(level, parity=parity)
        for level in range(maximum_twice_level + 1)
    }


def _add_scaled_word_state(
    out: WordState,
    scale: sp.Expr,
    state: Mapping[Word, sp.Expr],
) -> None:
    scale = sp.sympify(scale)
    if scale == 0:
        return
    for word, coefficient in state.items():
        value = sp.expand(out.get(word, sp.S.Zero) + scale * coefficient)
        if value == 0:
            out.pop(word, None)
        else:
            out[word] = value


def _add_scaled_state(
    out: State,
    scale: sp.Expr,
    state: Mapping[PBWState, sp.Expr],
) -> None:
    scale = sp.sympify(scale)
    if scale == 0:
        return
    for basis_state, coefficient in state.items():
        value = sp.expand(
            out.get(basis_state, sp.S.Zero) + scale * coefficient
        )
        if value == 0:
            out.pop(basis_state, None)
        else:
            out[basis_state] = value


def _word_state_items(
    state: Mapping[Word, sp.Expr],
) -> tuple[tuple[Word, sp.Expr], ...]:
    nonzero = (
        (word, sp.expand(coefficient))
        for word, coefficient in state.items()
        if coefficient != 0
    )
    return tuple(sorted(nonzero, key=lambda item: _word_order_key(item[0])))


def _state_items(
    state: Mapping[PBWState, sp.Expr],
) -> tuple[tuple[PBWState, sp.Expr], ...]:
    nonzero = (
        (basis_state, sp.expand(coefficient))
        for basis_state, coefficient in state.items()
        if coefficient != 0
    )
    return tuple(
        sorted(nonzero, key=lambda item: _basis_state_order_key(item[0]))
    )


def _super_bracket(
    left: Mode,
    right: Mode,
    c: sp.Expr,
) -> tuple[tuple[sp.Expr, Mode | None], ...]:
    """Return homogeneous terms in the Ramond superbracket.

    ``None`` denotes a scalar central term.
    """

    c = sp.sympify(c)
    terms: list[tuple[sp.Expr, Mode | None]] = []
    if left.kind == right.kind == "L":
        m = sp.Integer(left.index)
        n = sp.Integer(right.index)
        coefficient = m - n
        if coefficient != 0:
            terms.append((coefficient, L(left.index + right.index)))
        if left.index + right.index == 0:
            central = c * m * (m**2 - 1) / 12
            if central != 0:
                terms.append((central, None))
    elif left.kind == "L" and right.kind == "G":
        m = sp.Integer(left.index)
        n = sp.Integer(right.index)
        coefficient = m / 2 - n
        if coefficient != 0:
            terms.append((coefficient, G(left.index + right.index)))
    elif left.kind == "G" and right.kind == "L":
        n = sp.Integer(left.index)
        m = sp.Integer(right.index)
        coefficient = n - m / 2
        if coefficient != 0:
            terms.append((coefficient, G(left.index + right.index)))
    else:
        m = sp.Integer(left.index)
        terms.append((sp.Integer(2), L(left.index + right.index)))
        if left.index + right.index == 0:
            central = c * (4 * m**2 - 1) / 12
            if central != 0:
                terms.append((central, None))
    return tuple(terms)


@lru_cache(maxsize=None)
def _normal_order_negative_word_items(
    word: Word,
) -> tuple[tuple[Word, sp.Expr], ...]:
    for mode in word:
        if mode.index >= 0:
            raise ValueError("normal ordering accepts only negative modes")

    for position in range(len(word) - 1):
        left = word[position]
        right = word[position + 1]

        if left.kind == right.kind == "G" and left == right:
            reduced = (
                word[:position]
                + (L(left.index + right.index),)
                + word[position + 2 :]
            )
            return _normal_order_negative_word_items(reduced)

        if _mode_order_key(left) > _mode_order_key(right):
            out: WordState = {}
            swapped = word[:position] + (right, left) + word[position + 2 :]
            swap_sign = -1 if left.parity and right.parity else 1
            _add_scaled_word_state(
                out,
                sp.Integer(swap_sign),
                dict(_normal_order_negative_word_items(swapped)),
            )
            for coefficient, resulting_mode in _super_bracket(
                left, right, sp.S.Zero
            ):
                if resulting_mode is None:
                    raise AssertionError("negative modes cannot give a central term")
                bracket_word = (
                    word[:position]
                    + (resulting_mode,)
                    + word[position + 2 :]
                )
                _add_scaled_word_state(
                    out,
                    coefficient,
                    dict(_normal_order_negative_word_items(bracket_word)),
                )
            return _word_state_items(out)

    return ((word, sp.S.One),)


def normal_order_negative_word(word: Word) -> WordState:
    """Reduce a negative Ramond word to canonical oscillator PBW words."""

    return dict(_normal_order_negative_word_items(tuple(word)))


@lru_cache(maxsize=None)
def _act_mode_on_basis_items(
    mode: Mode,
    basis_state: PBWState,
    h: sp.Expr,
    c: sp.Expr,
) -> tuple[tuple[PBWState, sp.Expr], ...]:
    word = basis_state.word
    if word and not is_pbw_word(word):
        raise ValueError("the input descendant word must be in PBW order")

    if mode.index < 0:
        return tuple(
            (
                PBWState(ordered_word, basis_state.ground_parity),
                coefficient,
            )
            for ordered_word, coefficient in _normal_order_negative_word_items(
                (mode,) + word
            )
        )

    if mode.kind == "L" and mode.index == 0:
        eigenvalue = h + sp.Rational(twice_level(word), 2) if word else h
        return ((basis_state, sp.expand(eigenvalue)),)

    if mode.kind == "G" and mode.index == 0 and not word:
        delta = sp.expand(h - c / 24)
        if basis_state.ground_parity == 0:
            return ((ground_state(1), sp.S.One),)
        return ((ground_state(0), delta),)

    if mode.index > 0 and not word:
        return ()

    first = word[0]
    rest_state = PBWState(word[1:], basis_state.ground_parity)
    out: State = {}

    # A B = [A,B]_super + (-1)^(|A||B|) B A.
    moved_sign = -1 if mode.parity and first.parity else 1
    moved = dict(_act_mode_on_basis_items(mode, rest_state, h, c))
    for moved_state, coefficient in moved.items():
        ordered = _normal_order_negative_word_items(
            (first,) + moved_state.word
        )
        for ordered_word, order_coefficient in ordered:
            _add_scaled_state(
                out,
                moved_sign * coefficient * order_coefficient,
                {PBWState(ordered_word, moved_state.ground_parity): sp.S.One},
            )

    for coefficient, resulting_mode in _super_bracket(mode, first, c):
        if resulting_mode is None:
            _add_scaled_state(out, coefficient, {rest_state: sp.S.One})
        elif resulting_mode.index < 0:
            ordered = _normal_order_negative_word_items(
                (resulting_mode,) + rest_state.word
            )
            for ordered_word, order_coefficient in ordered:
                _add_scaled_state(
                    out,
                    coefficient * order_coefficient,
                    {
                        PBWState(
                            ordered_word, rest_state.ground_parity
                        ): sp.S.One
                    },
                )
        elif resulting_mode.kind == "L" and resulting_mode.index == 0:
            rest_level = (
                sp.Rational(twice_level(rest_state.word), 2)
                if rest_state.word
                else sp.S.Zero
            )
            _add_scaled_state(
                out,
                coefficient * (h + rest_level),
                {rest_state: sp.S.One},
            )
        else:
            acted = dict(
                _act_mode_on_basis_items(resulting_mode, rest_state, h, c)
            )
            _add_scaled_state(out, coefficient, acted)

    return _state_items(out)


def act_mode(
    mode: Mode,
    state: Mapping[PBWState, sp.Expr],
    *,
    h: sp.Expr,
    c: sp.Expr,
) -> State:
    """Act with one integer Ramond mode on a symbolic PBW state."""

    h = sp.sympify(h)
    c = sp.sympify(c)
    out: State = {}
    for basis_state, state_coefficient in state.items():
        if not isinstance(basis_state, PBWState):
            raise TypeError("every state key must be a PBWState")
        if basis_state.word and not is_pbw_word(basis_state.word):
            raise ValueError("every state word must be in PBW order")
        acted = dict(_act_mode_on_basis_items(mode, basis_state, h, c))
        _add_scaled_state(out, sp.sympify(state_coefficient), acted)
    return out


def descendant_inner_product(
    left: PBWState,
    right: PBWState,
    *,
    h: sp.Expr,
    c: sp.Expr,
) -> sp.Expr:
    """Evaluate the contravariant pairing of two Ramond PBW states."""

    if not isinstance(left, PBWState) or not isinstance(right, PBWState):
        raise TypeError("left and right must be PBWState objects")
    if (left.word and not is_pbw_word(left.word)) or (
        right.word and not is_pbw_word(right.word)
    ):
        raise ValueError("both descendant words must be in PBW order")
    if left.twice_descendant_level != right.twice_descendant_level:
        return sp.S.Zero
    if left.parity != right.parity:
        return sp.S.Zero

    state: State = {right: sp.S.One}
    for negative_mode in left.word:
        positive_mode = Mode(negative_mode.kind, -negative_mode.index)
        state = act_mode(positive_mode, state, h=h, c=c)
        if not state:
            return sp.S.Zero

    pairing = sp.S.Zero
    for residual_state, coefficient in state.items():
        if residual_state.word:
            continue
        pairing += coefficient * ground_inner_product(
            left.ground_parity,
            residual_state.ground_parity,
            h=h,
            c=c,
        )
    return sp.expand(pairing)


def gram_matrix(
    twice_descendant_level: int,
    *,
    h: sp.Expr,
    c: sp.Expr,
    parity: int | None = None,
) -> tuple[list[PBWState], sp.Matrix]:
    """Return a Ramond PBW basis and exact Gram matrix at one level."""

    basis = pbw_basis(twice_descendant_level, parity=parity)
    matrix = sp.Matrix(
        [
            [
                descendant_inner_product(left, right, h=h, c=c)
                for right in basis
            ]
            for left in basis
        ]
    )
    return basis, matrix


def gram_determinant(
    twice_descendant_level: int,
    *,
    h: sp.Expr,
    c: sp.Expr,
    parity: int | None = None,
) -> sp.Expr:
    """Return the factored Ramond Gram determinant at one level."""

    _, matrix = gram_matrix(
        twice_descendant_level,
        h=h,
        c=c,
        parity=parity,
    )
    return sp.factor(matrix.det())


def gram_nullspace(
    twice_descendant_level: int,
    *,
    h: sp.Expr,
    c: sp.Expr,
    parity: int | None = None,
) -> tuple[list[PBWState], list[sp.Matrix]]:
    """Return the basis and exact Gram-kernel columns at one level.

    This is intended for low levels after substituting an exact degenerate
    weight and central charge.  Generic symbolic nullspaces become costly
    rapidly as the PBW dimension grows.
    """

    basis, matrix = gram_matrix(
        twice_descendant_level,
        h=h,
        c=c,
        parity=parity,
    )
    reduced = matrix.applyfunc(lambda entry: sp.factor(sp.cancel(entry)))
    return basis, reduced.nullspace()


def xi_central_charge(b: sp.Expr) -> sp.Expr:
    """Return ``c=3/2+3(b+b^{-1})^2`` in Xi Yin's convention."""

    b = sp.sympify(b)
    return sp.factor(sp.Rational(3, 2) + 3 * (b + 1 / b) ** 2)


def ramond_degenerate_weight(r: int, s: int, b: sp.Expr) -> sp.Expr:
    """Return the Ramond Kac weight in Xi Yin's central-charge convention."""

    if not isinstance(r, int) or not isinstance(s, int) or r <= 0 or s <= 0:
        raise ValueError("r and s must be positive integers")
    if (r + s) % 2 == 0:
        raise ValueError("a Ramond degenerate pair must have r+s odd")
    b = sp.sympify(b)
    q = b + 1 / b
    degenerate_momentum = r * b + s / b
    return sp.factor(
        (q**2 - degenerate_momentum**2) / 8 + sp.Rational(1, 16)
    )


__all__ = [
    "G",
    "L",
    "Mode",
    "PBWState",
    "State",
    "Word",
    "WordState",
    "act_mode",
    "descendant_inner_product",
    "fermion_parity",
    "format_basis_state",
    "format_word",
    "gram_determinant",
    "gram_matrix",
    "gram_nullspace",
    "ground_action_matrix",
    "ground_gram_matrix",
    "ground_inner_product",
    "ground_state",
    "is_pbw_word",
    "normal_order_negative_word",
    "pbw_basis",
    "pbw_bases",
    "ramond_degenerate_weight",
    "twice_level",
    "xi_central_charge",
]
