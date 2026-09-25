#!/usr/bin/env python3
"""Exact symbolic algebra for a generic Neveu--Schwarz Verma module.

The central charge convention is the one used in Xi Yin's string theory
notes:

    [L_m,L_n] = (m-n)L_{m+n}
        + c m(m^2-1) delta_{m+n,0}/12,
    [L_m,G_r] = (m/2-r)G_{m+r},
    {G_r,G_s} = 2L_{r+s}
        + c(4r^2-1) delta_{r+s,0}/12.

All mode indices are stored doubled.  Thus ``Mode("G", -3)`` is
``G_{-3/2}``, while ``Mode("L", -4)`` is ``L_{-2}``.  This representation
uses integers throughout and makes half-integer level bookkeeping exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from itertools import combinations
from typing import Iterable, Literal, Mapping

import sympy as sp


ModeKind = Literal["L", "G"]


@dataclass(frozen=True)
class Mode:
    """One NS super-Virasoro mode with twice its index stored as an integer."""

    kind: ModeKind
    twice_index: int

    def __post_init__(self) -> None:
        if self.kind not in ("L", "G"):
            raise ValueError("mode kind must be 'L' or 'G'")
        if not isinstance(self.twice_index, int):
            raise TypeError("twice_index must be an integer")
        if self.kind == "L" and self.twice_index % 2:
            raise ValueError("an L mode has an integer index")
        if self.kind == "G" and self.twice_index % 2 == 0:
            raise ValueError("an NS G mode has a half-integer index")

    @property
    def index(self) -> sp.Rational:
        """Return the ordinary (possibly half-integer) mode index."""

        return sp.Rational(self.twice_index, 2)

    @property
    def parity(self) -> int:
        """Return 0 for a bosonic L mode and 1 for a fermionic G mode."""

        return 0 if self.kind == "L" else 1

    def __str__(self) -> str:
        return f"{self.kind}_{self.index}"


Word = tuple[Mode, ...]
State = dict[Word, sp.Expr]


def L(index: int) -> Mode:
    """Construct ``L_index`` using its ordinary integer index."""

    if not isinstance(index, int):
        raise TypeError("the index of L must be an integer")
    return Mode("L", 2 * index)


def G(index: int | Fraction | sp.Rational) -> Mode:
    """Construct an NS ``G_index`` from an exact half-integer index."""

    value = sp.Rational(index)
    twice_index = 2 * value
    if twice_index.q != 1 or int(twice_index) % 2 == 0:
        raise ValueError("the index of an NS G mode must be half-integral")
    return Mode("G", int(twice_index))


def twice_level(word: Word) -> int:
    """Return twice the descendant level of a negative-mode word."""

    if any(mode.twice_index >= 0 for mode in word):
        raise ValueError("descendant levels are defined for negative-mode words")
    return -sum(mode.twice_index for mode in word)


def fermion_parity(word: Word) -> int:
    """Return the number of G modes modulo two."""

    return sum(mode.parity for mode in word) % 2


def format_word(word: Word) -> str:
    """Format a descendant word in a compact human-readable form."""

    if not word:
        return "|h>"
    factors = " ".join(str(mode) for mode in word)
    return f"{factors} |h>"


def _mode_order_key(mode: Mode) -> tuple[int, int]:
    # Negative PBW words have all L modes before all G modes.  Within each
    # kind, a more negative mode is placed first.
    return (0 if mode.kind == "L" else 1, mode.twice_index)


def _word_order_key(word: Word) -> tuple[tuple[int, int], ...]:
    return tuple(_mode_order_key(mode) for mode in word)


def is_pbw_word(word: Word) -> bool:
    """Return whether ``word`` is in the canonical negative-mode PBW order."""

    if any(mode.twice_index >= 0 for mode in word):
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
    odd_parts = tuple(range(1, maximum_sum + 1, 2))
    for size in range(len(odd_parts) + 1):
        for subset in combinations(odd_parts, size):
            if sum(subset) <= maximum_sum:
                yield subset


@lru_cache(maxsize=None)
def _pbw_basis_cached(twice_descendant_level: int) -> tuple[Word, ...]:
    if twice_descendant_level < 0:
        return ()
    out: list[Word] = []
    for fermion_indices in _fermion_subsets(twice_descendant_level):
        remaining = twice_descendant_level - sum(fermion_indices)
        if remaining % 2:
            continue
        for partition in _integer_partitions(remaining // 2, remaining // 2):
            l_modes = tuple(L(-part) for part in partition)
            g_modes = tuple(Mode("G", -part) for part in reversed(fermion_indices))
            out.append(l_modes + g_modes)
    return tuple(out)


def pbw_basis(
    twice_descendant_level: int,
    *,
    parity: int | None = None,
) -> list[Word]:
    """Generate the canonical PBW basis at a fixed twice-level.

    ``parity`` may be ``0`` or ``1`` to retain only even or odd states.
    In an NS Verma module a fixed level already fixes the parity, but the
    filter is useful when basis words from several levels are combined.
    """

    if not isinstance(twice_descendant_level, int):
        raise TypeError("twice_descendant_level must be an integer")
    if twice_descendant_level < 0:
        raise ValueError("twice_descendant_level must be nonnegative")
    if parity not in (None, 0, 1):
        raise ValueError("parity must be None, 0, or 1")
    basis = list(_pbw_basis_cached(twice_descendant_level))
    if parity is not None:
        basis = [word for word in basis if fermion_parity(word) == parity]
    return basis


def pbw_bases(
    maximum_twice_level: int,
    *,
    parity: int | None = None,
) -> dict[int, list[Word]]:
    """Generate PBW bases at every twice-level up to an inclusive cutoff."""

    if not isinstance(maximum_twice_level, int):
        raise TypeError("maximum_twice_level must be an integer")
    if maximum_twice_level < 0:
        raise ValueError("maximum_twice_level must be nonnegative")
    return {
        level: pbw_basis(level, parity=parity)
        for level in range(maximum_twice_level + 1)
    }


def _add_scaled_state(out: State, scale: sp.Expr, state: Mapping[Word, sp.Expr]) -> None:
    scale = sp.sympify(scale)
    if scale == 0:
        return
    for word, coefficient in state.items():
        value = sp.expand(out.get(word, sp.S.Zero) + scale * coefficient)
        if value == 0:
            out.pop(word, None)
        else:
            out[word] = value


def _state_items(state: Mapping[Word, sp.Expr]) -> tuple[tuple[Word, sp.Expr], ...]:
    nonzero = (
        (word, sp.expand(coefficient))
        for word, coefficient in state.items()
        if coefficient != 0
    )
    return tuple(sorted(nonzero, key=lambda item: _word_order_key(item[0])))


def _super_bracket(
    left: Mode,
    right: Mode,
    c: sp.Expr,
) -> tuple[tuple[sp.Expr, Mode | None], ...]:
    """Return homogeneous terms in [left,right]_super.

    ``None`` denotes a scalar central term.
    """

    c = sp.sympify(c)
    terms: list[tuple[sp.Expr, Mode | None]] = []
    if left.kind == right.kind == "L":
        m = left.index
        n = right.index
        coefficient = m - n
        if coefficient != 0:
            terms.append((coefficient, Mode("L", left.twice_index + right.twice_index)))
        if left.twice_index + right.twice_index == 0:
            central = c * m * (m**2 - 1) / 12
            if central != 0:
                terms.append((central, None))
    elif left.kind == "L" and right.kind == "G":
        m = left.index
        r = right.index
        coefficient = m / 2 - r
        if coefficient != 0:
            terms.append((coefficient, Mode("G", left.twice_index + right.twice_index)))
    elif left.kind == "G" and right.kind == "L":
        r = left.index
        m = right.index
        coefficient = r - m / 2
        if coefficient != 0:
            terms.append((coefficient, Mode("G", left.twice_index + right.twice_index)))
    else:
        r = left.index
        terms.append((sp.Integer(2), Mode("L", left.twice_index + right.twice_index)))
        if left.twice_index + right.twice_index == 0:
            central = c * (4 * r**2 - 1) / 12
            if central != 0:
                terms.append((central, None))
    return tuple(terms)


@lru_cache(maxsize=None)
def _normal_order_negative_word_items(word: Word) -> tuple[tuple[Word, sp.Expr], ...]:
    for mode in word:
        if mode.twice_index >= 0:
            raise ValueError("normal_order_negative_word accepts only negative modes")

    for position in range(len(word) - 1):
        left = word[position]
        right = word[position + 1]

        # In a PBW basis a fermionic generator occurs at most once.  The
        # relation {G_r,G_r}=2L_{2r} gives G_r^2=L_{2r}.
        if left.kind == right.kind == "G" and left == right:
            reduced = (
                word[:position]
                + (Mode("L", left.twice_index + right.twice_index),)
                + word[position + 2 :]
            )
            return _normal_order_negative_word_items(reduced)

        if _mode_order_key(left) > _mode_order_key(right):
            out: State = {}
            swapped = word[:position] + (right, left) + word[position + 2 :]
            swap_sign = -1 if left.parity and right.parity else 1
            _add_scaled_state(
                out,
                sp.Integer(swap_sign),
                dict(_normal_order_negative_word_items(swapped)),
            )
            for coefficient, resulting_mode in _super_bracket(left, right, sp.S.Zero):
                if resulting_mode is None:
                    raise AssertionError("two negative modes cannot produce a central term")
                bracket_word = (
                    word[:position] + (resulting_mode,) + word[position + 2 :]
                )
                _add_scaled_state(
                    out,
                    coefficient,
                    dict(_normal_order_negative_word_items(bracket_word)),
                )
            return _state_items(out)

    return ((word, sp.S.One),)


def normal_order_negative_word(word: Word) -> State:
    """Reduce a negative-mode word to a linear combination of PBW words."""

    return dict(_normal_order_negative_word_items(tuple(word)))


@lru_cache(maxsize=None)
def _act_mode_on_word_items(
    mode: Mode,
    word: Word,
    h: sp.Expr,
    c: sp.Expr,
) -> tuple[tuple[Word, sp.Expr], ...]:
    if not is_pbw_word(word) and word:
        raise ValueError("the input descendant word must be in PBW order")

    if mode.twice_index < 0:
        return _normal_order_negative_word_items((mode,) + word)

    if mode.twice_index == 0:
        if mode.kind != "L":
            raise ValueError("there is no G_0 mode in the NS sector")
        eigenvalue = h + sp.Rational(twice_level(word), 2) if word else h
        return ((word, sp.expand(eigenvalue)),)

    if not word:
        # Positive modes annihilate the highest-weight state.
        return ()

    first = word[0]
    rest = word[1:]
    out: State = {}

    # A B = [A,B]_super + (-1)^(|A||B|) B A.
    moved_sign = -1 if mode.parity and first.parity else 1
    moved = dict(_act_mode_on_word_items(mode, rest, h, c))
    for moved_word, coefficient in moved.items():
        ordered = dict(_normal_order_negative_word_items((first,) + moved_word))
        _add_scaled_state(out, moved_sign * coefficient, ordered)

    for coefficient, resulting_mode in _super_bracket(mode, first, c):
        if resulting_mode is None:
            _add_scaled_state(out, coefficient, {rest: sp.S.One})
        elif resulting_mode.twice_index < 0:
            ordered = dict(_normal_order_negative_word_items((resulting_mode,) + rest))
            _add_scaled_state(out, coefficient, ordered)
        elif resulting_mode.twice_index == 0:
            if resulting_mode.kind != "L":
                raise AssertionError("the NS algebra cannot produce G_0")
            rest_level = sp.Rational(twice_level(rest), 2) if rest else sp.S.Zero
            _add_scaled_state(out, coefficient * (h + rest_level), {rest: sp.S.One})
        else:
            acted = dict(_act_mode_on_word_items(resulting_mode, rest, h, c))
            _add_scaled_state(out, coefficient, acted)

    return _state_items(out)


def act_mode(
    mode: Mode,
    state: Mapping[Word, sp.Expr],
    *,
    h: sp.Expr,
    c: sp.Expr,
) -> State:
    """Act with one mode on a symbolic linear combination of PBW states."""

    h = sp.sympify(h)
    c = sp.sympify(c)
    out: State = {}
    for word, state_coefficient in state.items():
        word = tuple(word)
        if word and not is_pbw_word(word):
            raise ValueError("every state word must be in PBW order")
        acted = dict(_act_mode_on_word_items(mode, word, h, c))
        _add_scaled_state(out, sp.sympify(state_coefficient), acted)
    return out


def descendant_inner_product(
    left: Word,
    right: Word,
    *,
    h: sp.Expr,
    c: sp.Expr,
) -> sp.Expr:
    """Evaluate the contravariant Gram pairing of two PBW descendants."""

    left = tuple(left)
    right = tuple(right)
    if (left and not is_pbw_word(left)) or (right and not is_pbw_word(right)):
        raise ValueError("both descendant words must be in PBW order")
    if twice_level(left) != twice_level(right):
        return sp.S.Zero
    if fermion_parity(left) != fermion_parity(right):
        return sp.S.Zero

    state: State = {right: sp.S.One}
    # If left=A_1...A_k|h>, then <left|=<h|A_k^dag...A_1^dag.
    # Acting on the ket from the right therefore applies
    # A_1^dag, A_2^dag, ..., A_k^dag in this loop order.
    for negative_mode in left:
        positive_mode = Mode(negative_mode.kind, -negative_mode.twice_index)
        state = act_mode(positive_mode, state, h=h, c=c)
        if not state:
            return sp.S.Zero
    return sp.expand(state.get((), sp.S.Zero))


def gram_matrix(
    twice_descendant_level: int,
    *,
    h: sp.Expr,
    c: sp.Expr,
    parity: int | None = None,
) -> tuple[list[Word], sp.Matrix]:
    """Return the PBW basis and exact Gram matrix at one twice-level."""

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
    """Return the factored determinant of the Gram matrix at one level."""

    _, matrix = gram_matrix(
        twice_descendant_level,
        h=h,
        c=c,
        parity=parity,
    )
    return sp.factor(matrix.det())


def xi_central_charge(b: sp.Expr) -> sp.Expr:
    """Return ``c=3/2+3(b+b^{-1})^2`` in Xi Yin's convention."""

    b = sp.sympify(b)
    return sp.factor(sp.Rational(3, 2) + 3 * (b + 1 / b) ** 2)


def ns_degenerate_weight(r: int, s: int, b: sp.Expr) -> sp.Expr:
    """Return the NS degenerate weight ``h_{r,s}`` in Xi's convention."""

    if not isinstance(r, int) or not isinstance(s, int) or r <= 0 or s <= 0:
        raise ValueError("r and s must be positive integers")
    if (r + s) % 2:
        raise ValueError("an NS degenerate pair must have r+s even")
    b = sp.sympify(b)
    q = b + 1 / b
    degenerate_momentum = r * b + s / b
    return sp.factor((q**2 - degenerate_momentum**2) / 8)


__all__ = [
    "G",
    "L",
    "Mode",
    "State",
    "Word",
    "act_mode",
    "descendant_inner_product",
    "fermion_parity",
    "format_word",
    "gram_determinant",
    "gram_matrix",
    "is_pbw_word",
    "normal_order_negative_word",
    "ns_degenerate_weight",
    "pbw_basis",
    "pbw_bases",
    "twice_level",
    "xi_central_charge",
]
