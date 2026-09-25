#!/usr/bin/env python3
r"""Genus-zero five-point blocks in the sphere linear (comb) channel.

For external weights ``(d1,...,d5)`` and internal weights ``(h1,h2)``, the
direct finite-level definition implemented here is

.. math::

   F_{N_1,N_2}=
   \rho(d_5,d_4,A_2)(B_{h_2}^{-1})^{A_2B_2}
   \rho(B_2,d_3,A_1)(B_{h_1}^{-1})^{A_1B_1}
   \rho(B_1,d_2,d_1).

The two expansion variables are the standard linear-channel plumbing
coordinates ``q1=z2/z3`` and ``q2=z3``.  This is a comb/linear channel, not
a necklace: cutting the sphere exposes two internal edges and three
trinions.

Both an ordinary Virasoro oracle (the pure-``L`` subspace) and the full NS
super-Virasoro oracle are supplied.  The NS oracle accepts an independent
external descendant word on every leg and independent even/odd three-point
form weights on every trinion.  It can therefore assemble the block pieces
created by a chosen PCO routing.

The recursive evaluators implement the two-edge internal-weight
``h``-recursion obtained by specializing Cho--Collier--Yin's sphere
``N``-point linear-channel formula to ``N=5``.  The NS version is its
super-Virasoro analogue, with odd null states explicitly changing the
fermion routing.  Direct low-level coefficients remain the definition and
the regression oracle.
"""

from __future__ import annotations

import cmath
from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Callable, Literal, Mapping, Sequence

import numpy as np
import sympy as sp

from ns_algebra.ns_fusion import c_xi, h_rs as ns_h_rs
from ns_algebra.ns_sca import (
    G,
    L,
    Word,
    fermion_parity,
    gram_matrix,
    twice_level,
)
from ns_algebra.ns_three_point_tensor import ns_three_point
from spin23_genus1_recursion import FinitePartDiagnostics, dictionary_finite_part


Sector = Literal["Virasoro", "NS"]
Method = Literal["direct", "h-recursion", "finite-part h-recursion"]
VertexBackend = Literal["direct", "template"]
LevelPair = tuple[int, int]
FormWeights = tuple[tuple[complex, complex], ...]

EMPTY_WORD: Word = ()
G_MINUS_HALF: Word = (G(sp.Rational(-1, 2)),)

(
    _TEMPLATE_H_INFINITY,
    _TEMPLATE_H_MIDDLE,
    _TEMPLATE_H_ZERO,
    _TEMPLATE_C,
) = sp.symbols(
    "fivepoint_h_infinity fivepoint_h_middle fivepoint_h_zero fivepoint_c"
)


class FivePointRecursionPole(ArithmeticError):
    """Raised when an unregulated five-point recursion hits a pole."""


def _as_numeric(expression: sp.Expr | complex, digits: int) -> complex:
    value = complex(sp.N(sp.sympify(expression), digits))
    if not math.isfinite(value.real) or not math.isfinite(value.imag):
        raise ArithmeticError(f"non-finite five-point block entry {value!r}")
    return value


def _validate_cutoffs(
    maximum_levels: int | Sequence[int],
) -> tuple[int, int]:
    if isinstance(maximum_levels, bool):
        raise TypeError("descendant cutoffs must be integers")
    if isinstance(maximum_levels, int):
        cutoffs = (maximum_levels, maximum_levels)
    else:
        cutoffs = tuple(maximum_levels)
    if len(cutoffs) != 2:
        raise ValueError("two descendant cutoffs are required")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in cutoffs):
        raise TypeError("descendant cutoffs must be integers")
    if any(value < 0 for value in cutoffs):
        raise ValueError("descendant cutoffs must be nonnegative")
    return int(cutoffs[0]), int(cutoffs[1])


def _validate_weights(
    internal_weights: Sequence[complex],
    external_weights: Sequence[complex],
) -> tuple[tuple[complex, complex], tuple[complex, complex, complex, complex, complex]]:
    internal = tuple(complex(value) for value in internal_weights)
    external = tuple(complex(value) for value in external_weights)
    if len(internal) != 2:
        raise ValueError("two internal weights are required")
    if len(external) != 5:
        raise ValueError("five external weights are required")
    return internal, external  # type: ignore[return-value]


def _validate_external_words(
    external_words: Sequence[Word] | None,
) -> tuple[Word, Word, Word, Word, Word]:
    words = (EMPTY_WORD,) * 5 if external_words is None else tuple(
        tuple(word) for word in external_words
    )
    if len(words) != 5:
        raise ValueError("five external descendant words are required")
    # ns_three_point performs the detailed canonical-PBW validation.
    return words  # type: ignore[return-value]


def _validate_form_weights(
    form_weights: Sequence[Sequence[complex]] | None,
) -> FormWeights:
    if form_weights is None:
        return ((1.0 + 0.0j, 1.0 + 0.0j),) * 3
    weights = tuple(tuple(complex(value) for value in pair) for pair in form_weights)
    if len(weights) != 3 or any(len(pair) != 2 for pair in weights):
        raise ValueError(
            "form_weights must contain one (even, odd) pair per trinion"
        )
    return weights  # type: ignore[return-value]


def _validate_vertex_backend(vertex_backend: str) -> VertexBackend:
    if vertex_backend not in ("direct", "template"):
        raise ValueError("vertex_backend must be 'direct' or 'template'")
    return vertex_backend  # type: ignore[return-value]


def pco_external_words(
    zero_picture_legs: Sequence[int],
) -> tuple[Word, Word, Word, Word, Word]:
    r"""Return a five-leg routing with ``G_-1/2`` on selected legs.

    Legs are numbered ``1,...,5``.  A usual five-point sphere picture choice
    has three zero-picture legs, but the function deliberately accepts any
    number: the full PCO contains matter, Liouville, and ghost terms, and a
    physical amplitude is a sum of such sector-specific descendant routings.
    """

    legs = tuple(zero_picture_legs)
    if len(set(legs)) != len(legs):
        raise ValueError("zero_picture_legs must not contain duplicates")
    if any(
        isinstance(leg, bool)
        or not isinstance(leg, int)
        or leg not in range(1, 6)
        for leg in legs
    ):
        raise ValueError("zero-picture legs must be distinct integers from 1 to 5")
    selected = set(legs)
    return tuple(
        G_MINUS_HALF if leg in selected else EMPTY_WORD for leg in range(1, 6)
    )  # type: ignore[return-value]


@lru_cache(maxsize=None)
def _edge_data(
    sector: Sector,
    level: int,
    h: complex,
    c: complex,
    digits: int,
) -> tuple[tuple[Word, ...], np.ndarray, float]:
    twice_descendant_level = 2 * level if sector == "Virasoro" else level
    full_basis, full_gram = gram_matrix(
        twice_descendant_level,
        h=sp.sympify(h),
        c=sp.sympify(c),
    )
    if sector == "Virasoro":
        indices = [
            index
            for index, word in enumerate(full_basis)
            if all(mode.kind == "L" for mode in word)
        ]
        basis = tuple(full_basis[index] for index in indices)
        selected_gram = full_gram.extract(indices, indices)
    else:
        basis = tuple(full_basis)
        selected_gram = full_gram
    matrix = np.asarray(
        [
            [
                _as_numeric(selected_gram[row, column], digits)
                for column in range(selected_gram.cols)
            ]
            for row in range(selected_gram.rows)
        ],
        dtype=np.complex128,
    )
    condition = float(np.linalg.cond(matrix))
    inverse = np.linalg.inv(matrix)
    inverse.setflags(write=False)
    return basis, inverse, condition


def _weighted_three_point(
    infinity_word: Word,
    middle_word: Word,
    zero_word: Word,
    *,
    h_infinity: complex,
    h_middle: complex,
    h_zero: complex,
    c: complex,
    form_weight: tuple[complex, complex],
    digits: int,
) -> complex:
    parity = (
        fermion_parity(infinity_word)
        + fermion_parity(middle_word)
        + fermion_parity(zero_word)
    ) % 2
    value = ns_three_point(
        infinity_word,
        middle_word,
        zero_word,
        h_infinity=h_infinity,
        h_middle=h_middle,
        h_zero=h_zero,
        c=c,
        simplify=False,
    )
    return form_weight[parity] * _as_numeric(value, digits)


@lru_cache(maxsize=None)
def _vertex_template(
    infinity_basis: tuple[Word, ...],
    middle_word: Word,
    zero_basis: tuple[Word, ...],
) -> tuple[Callable[..., object], np.ndarray]:
    """Compile one exact weight-independent NS Ward tensor."""

    expressions: list[sp.Expr] = []
    parities = np.empty((len(infinity_basis), len(zero_basis)), dtype=np.int8)
    for row, infinity_word in enumerate(infinity_basis):
        for column, zero_word in enumerate(zero_basis):
            parities[row, column] = (
                fermion_parity(infinity_word)
                + fermion_parity(middle_word)
                + fermion_parity(zero_word)
            ) % 2
            expressions.append(
                ns_three_point(
                    infinity_word,
                    middle_word,
                    zero_word,
                    h_infinity=_TEMPLATE_H_INFINITY,
                    h_middle=_TEMPLATE_H_MIDDLE,
                    h_zero=_TEMPLATE_H_ZERO,
                    c=_TEMPLATE_C,
                    simplify=False,
                )
            )
    evaluator = sp.lambdify(
        (
            _TEMPLATE_H_INFINITY,
            _TEMPLATE_H_MIDDLE,
            _TEMPLATE_H_ZERO,
            _TEMPLATE_C,
        ),
        tuple(expressions),
        modules="numpy",
        cse=True,
    )
    parities.setflags(write=False)
    return evaluator, parities


def _template_vertex_data(
    infinity_basis: tuple[Word, ...],
    middle_word: Word,
    zero_basis: tuple[Word, ...],
    *,
    h_infinity: complex,
    h_middle: complex,
    h_zero: complex,
    c: complex,
    form_weight: tuple[complex, complex],
) -> np.ndarray:
    evaluator, parities = _vertex_template(
        infinity_basis, middle_word, zero_basis
    )
    raw = evaluator(h_infinity, h_middle, h_zero, c)
    entries = np.asarray(raw, dtype=np.complex128).reshape(parities.shape)
    return entries * np.asarray(form_weight, dtype=np.complex128)[parities]


def _direct_coefficient(
    level_1: int,
    level_2: int,
    *,
    sector: Sector,
    c: complex,
    internal_weights: tuple[complex, complex],
    external_weights: tuple[complex, complex, complex, complex, complex],
    external_words: tuple[Word, Word, Word, Word, Word],
    form_weights: FormWeights,
    digits: int,
    condition_limit: float,
    vertex_backend: VertexBackend,
) -> tuple[complex, tuple[float, float]]:
    h1, h2 = internal_weights
    d1, d2, d3, d4, d5 = external_weights
    w1, w2, w3, w4, w5 = external_words
    basis1, inverse1, condition1 = _edge_data(
        sector, level_1, h1, c, digits
    )
    basis2, inverse2, condition2 = _edge_data(
        sector, level_2, h2, c, digits
    )
    for edge, (level, condition) in enumerate(
        ((level_1, condition1), (level_2, condition2)), start=1
    ):
        if not math.isfinite(condition) or condition > condition_limit:
            unit = "" if sector == "Virasoro" else "/2"
            raise np.linalg.LinAlgError(
                f"{sector} Gram matrix on edge {edge} at level {level}{unit} "
                f"has condition number {condition:.3e}, above "
                f"{condition_limit:.3e}"
            )

    if vertex_backend == "template":
        right = _template_vertex_data(
            basis1,
            w2,
            (w1,),
            h_infinity=h1,
            h_middle=d2,
            h_zero=d1,
            c=c,
            form_weight=form_weights[0],
        )[:, 0]
        central = _template_vertex_data(
            basis2,
            w3,
            basis1,
            h_infinity=h2,
            h_middle=d3,
            h_zero=h1,
            c=c,
            form_weight=form_weights[1],
        )
        left = _template_vertex_data(
            (w5,),
            w4,
            basis2,
            h_infinity=d5,
            h_middle=d4,
            h_zero=h2,
            c=c,
            form_weight=form_weights[2],
        )[0, :]
    else:
        right = np.asarray(
            [
                _weighted_three_point(
                    internal_word,
                    w2,
                    w1,
                    h_infinity=h1,
                    h_middle=d2,
                    h_zero=d1,
                    c=c,
                    form_weight=form_weights[0],
                    digits=digits,
                )
                for internal_word in basis1
            ]
            ,
            dtype=np.complex128,
        )
        central = np.asarray(
            [
                [
                    _weighted_three_point(
                        word2,
                        w3,
                        word1,
                        h_infinity=h2,
                        h_middle=d3,
                        h_zero=h1,
                        c=c,
                        form_weight=form_weights[1],
                        digits=digits,
                    )
                    for word1 in basis1
                ]
                for word2 in basis2
            ],
            dtype=np.complex128,
        )
        left = np.asarray(
            [
                _weighted_three_point(
                    w5,
                    w4,
                    internal_word,
                    h_infinity=d5,
                    h_middle=d4,
                    h_zero=h2,
                    c=c,
                    form_weight=form_weights[2],
                    digits=digits,
                )
                for internal_word in basis2
            ],
            dtype=np.complex128,
        )
    coefficient = left @ inverse2 @ central @ inverse1 @ right
    return complex(coefficient), (condition1, condition2)


@dataclass(frozen=True)
class SphereFivePointSeries:
    """A finite stripped five-point block series in ``(q1,q2)``."""

    coefficients: Mapping[LevelPair, complex]
    sector: Sector
    method: Method
    c: complex
    internal_weights: tuple[complex, complex]
    external_weights: tuple[complex, complex, complex, complex, complex]
    external_words: tuple[Word, Word, Word, Word, Word]
    maximum_levels: tuple[int, int]
    gram_condition_numbers: Mapping[tuple[int, int], float]

    def descendant_value(
        self,
        q_1: complex,
        q_2: complex,
        *,
        maximum_levels: int | Sequence[int] | None = None,
    ) -> complex:
        """Evaluate the stripped descendant series at two plumbing values."""

        cutoffs = (
            self.maximum_levels
            if maximum_levels is None
            else _validate_cutoffs(maximum_levels)
        )
        if any(
            requested > available
            for requested, available in zip(cutoffs, self.maximum_levels)
        ):
            raise ValueError("an evaluation cutoff exceeds the constructed series")
        unit = 1.0 if self.sector == "Virasoro" else 0.5
        q_values = (complex(q_1), complex(q_2))
        total = 0.0j
        for levels, coefficient in self.coefficients.items():
            if any(level > cutoff for level, cutoff in zip(levels, cutoffs)):
                continue
            term = complex(coefficient)
            for q_value, level in zip(q_values, levels):
                if q_value == 0 and level:
                    term = 0.0j
                    break
                if level:
                    term *= cmath.exp(unit * level * cmath.log(q_value))
            total += term
        return total

    def component_descendant_value(
        self,
        q_1: complex,
        q_2: complex,
        component_parities: Sequence[int],
        *,
        maximum_levels: int | Sequence[int] | None = None,
    ) -> complex:
        """Evaluate one fixed pair of internal fermion parities.

        For an ordinary Virasoro series the only accepted routing is
        ``(0,0)``.  For an NS series the two labels select coefficient keys
        whose twice-levels have the corresponding parities.
        """

        routing = tuple(component_parities)
        if len(routing) != 2 or any(value not in (0, 1) for value in routing):
            raise ValueError("component_parities must be a pair of zeroes or ones")
        if self.sector == "Virasoro" and routing != (0, 0):
            raise ValueError("an ordinary Virasoro series has only routing (0,0)")
        cutoffs = (
            self.maximum_levels
            if maximum_levels is None
            else _validate_cutoffs(maximum_levels)
        )
        if any(
            requested > available
            for requested, available in zip(cutoffs, self.maximum_levels)
        ):
            raise ValueError("an evaluation cutoff exceeds the constructed series")
        unit = 1.0 if self.sector == "Virasoro" else 0.5
        q_values = (complex(q_1), complex(q_2))
        total = 0.0j
        for levels, coefficient in self.coefficients.items():
            if any(level > cutoff for level, cutoff in zip(levels, cutoffs)):
                continue
            if self.sector == "NS" and any(
                level % 2 != parity for level, parity in zip(levels, routing)
            ):
                continue
            term = complex(coefficient)
            for q_value, level in zip(q_values, levels):
                if q_value == 0 and level:
                    term = 0.0j
                    break
                if level:
                    term *= cmath.exp(unit * level * cmath.log(q_value))
            total += term
        return total

    def _primary_factor(self, q_1: complex, q_2: complex) -> complex:
        q_1, q_2 = complex(q_1), complex(q_2)
        if q_1 == 0 or q_2 == 0:
            raise ValueError("primary plumbing powers are singular at q=0")
        effective = tuple(
            weight + 0.5 * twice_level(word) if word else weight
            for weight, word in zip(self.external_weights, self.external_words)
        )
        h1, h2 = self.internal_weights
        exponent1 = h1 - effective[0] - effective[1]
        exponent2 = h2 - effective[0] - effective[1] - effective[2]
        primary = cmath.exp(exponent1 * cmath.log(q_1))
        primary *= cmath.exp(exponent2 * cmath.log(q_2))
        return primary

    def component_value(
        self,
        q_1: complex,
        q_2: complex,
        component_parities: Sequence[int],
        *,
        include_primary_powers: bool = True,
        maximum_levels: int | Sequence[int] | None = None,
    ) -> complex:
        """Evaluate one routed component, optionally with primary powers."""

        descendant = self.component_descendant_value(
            q_1,
            q_2,
            component_parities,
            maximum_levels=maximum_levels,
        )
        if not include_primary_powers:
            return descendant
        return self._primary_factor(q_1, q_2) * descendant

    def value(
        self,
        q_1: complex,
        q_2: complex,
        *,
        include_primary_powers: bool = True,
        maximum_levels: int | Sequence[int] | None = None,
    ) -> complex:
        r"""Evaluate the block in the CCY linear-channel coordinates.

        With ``z2=q1*q2`` and ``z3=q2``, the leading powers are
        ``q1^(h1-D1-D2) q2^(h2-D1-D2-D3)``, where ``Di`` includes the
        supplied external descendant level.
        """

        descendant = self.descendant_value(
            q_1, q_2, maximum_levels=maximum_levels
        )
        if not include_primary_powers:
            return descendant
        return self._primary_factor(q_1, q_2) * descendant

    @property
    def leading_coefficient(self) -> complex:
        """Return the ground-level coefficient."""

        return complex(self.coefficients.get((0, 0), 0.0j))


def _direct_series(
    *,
    sector: Sector,
    c: complex,
    internal_weights: Sequence[complex],
    external_weights: Sequence[complex],
    maximum_levels: int | Sequence[int],
    external_words: Sequence[Word] | None,
    form_weights: Sequence[Sequence[complex]] | None,
    digits: int,
    condition_limit: float,
    vertex_backend: VertexBackend,
) -> SphereFivePointSeries:
    internal, external = _validate_weights(internal_weights, external_weights)
    cutoffs = _validate_cutoffs(maximum_levels)
    words = _validate_external_words(external_words)
    weights = _validate_form_weights(form_weights)
    if sector == "Virasoro":
        if any(any(mode.kind != "L" for mode in word) for word in words):
            raise ValueError("ordinary Virasoro external words may contain only L modes")
        if any(pair != (1.0 + 0.0j, 1.0 + 0.0j) for pair in weights):
            raise ValueError("form_weights apply only to the NS block")
    coefficients: dict[LevelPair, complex] = {}
    conditions: dict[tuple[int, int], float] = {}
    for level1 in range(cutoffs[0] + 1):
        for level2 in range(cutoffs[1] + 1):
            coefficient, edge_conditions = _direct_coefficient(
                level1,
                level2,
                sector=sector,
                c=complex(c),
                internal_weights=internal,
                external_weights=external,
                external_words=words,
                form_weights=weights,
                digits=int(digits),
                condition_limit=float(condition_limit),
                vertex_backend=vertex_backend,
            )
            coefficients[(level1, level2)] = coefficient
            conditions[(0, level1)] = edge_conditions[0]
            conditions[(1, level2)] = edge_conditions[1]
    return SphereFivePointSeries(
        coefficients=coefficients,
        sector=sector,
        method="direct",
        c=complex(c),
        internal_weights=internal,
        external_weights=external,
        external_words=words,
        maximum_levels=cutoffs,
        gram_condition_numbers=conditions,
    )


def direct_virasoro_sphere_fivepoint_series(
    *,
    c: complex,
    internal_weights: Sequence[complex],
    external_weights: Sequence[complex],
    maximum_levels: int | Sequence[int],
    external_words: Sequence[Word] | None = None,
    digits: int = 50,
    condition_limit: float = 1.0e13,
    vertex_backend: VertexBackend = "direct",
) -> SphereFivePointSeries:
    """Construct the direct ordinary-Virasoro five-point oracle."""

    backend = _validate_vertex_backend(vertex_backend)
    return _direct_series(
        sector="Virasoro",
        c=c,
        internal_weights=internal_weights,
        external_weights=external_weights,
        maximum_levels=maximum_levels,
        external_words=external_words,
        form_weights=None,
        digits=digits,
        condition_limit=condition_limit,
        vertex_backend=backend,
    )


def direct_ns_sphere_fivepoint_series(
    *,
    c: complex,
    internal_weights: Sequence[complex],
    external_weights: Sequence[complex],
    maximum_twice_levels: int | Sequence[int],
    external_words: Sequence[Word] | None = None,
    form_weights: Sequence[Sequence[complex]] | None = None,
    digits: int = 50,
    condition_limit: float = 1.0e13,
    vertex_backend: VertexBackend = "direct",
) -> SphereFivePointSeries:
    r"""Construct the direct NS five-point oracle.

    Coefficient keys store twice-levels.  ``form_weights[v]=(C_even,C_odd)``
    multiplies the normalized form selected at the left, central, or right
    trinion.  Unit weights expose the algebraic block components.
    """

    backend = _validate_vertex_backend(vertex_backend)
    return _direct_series(
        sector="NS",
        c=c,
        internal_weights=internal_weights,
        external_weights=external_weights,
        maximum_levels=maximum_twice_levels,
        external_words=external_words,
        form_weights=form_weights,
        digits=digits,
        condition_limit=condition_limit,
        vertex_backend=backend,
    )


def virasoro_central_charge(b: complex) -> complex:
    """Return ``c=1+6(b+b^-1)^2``."""

    b = complex(b)
    if b == 0:
        raise ValueError("b must be nonzero")
    charge = b + 1.0 / b
    return 1.0 + 6.0 * charge * charge


def virasoro_liouville_weight(momentum: complex, b: complex) -> complex:
    """Return ``h=Q^2/4+P^2`` for Liouville momentum ``P``."""

    b = complex(b)
    if b == 0:
        raise ValueError("b must be nonzero")
    charge = b + 1.0 / b
    return charge * charge / 4.0 + complex(momentum) ** 2


def ns_liouville_weight(momentum: complex, b: complex) -> complex:
    """Return ``h_NS=Q^2/8+P^2/2`` for super-Liouville momentum ``P``."""

    b = complex(b)
    if b == 0:
        raise ValueError("b must be nonzero")
    charge = b + 1.0 / b
    return charge * charge / 8.0 + complex(momentum) ** 2 / 2.0


def ns_fivepoint_structure_weights(
    *,
    internal_momenta: Sequence[complex],
    external_momenta: Sequence[complex],
    precision: int = 40,
) -> FormWeights:
    r"""Return the three ``(C_even,C_odd)`` pairs at ``b=1``.

    The trinion ordering follows the direct comb tensor:
    ``(P1,p2,p1)``, ``(P2,p3,P1)``, and ``(p5,p4,P2)``.  The delta-normalized
    NS constants are permutation symmetric, but spelling out the routing
    makes the two internal momentum integrations unambiguous.
    """

    from spin23_super_liouville_data import ns_structure_constants

    internal = tuple(complex(value) for value in internal_momenta)
    external = tuple(complex(value) for value in external_momenta)
    if len(internal) != 2 or len(external) != 5:
        raise ValueError("two internal and five external momenta are required")
    p1, p2 = internal
    d1, d2, d3, d4, d5 = external
    return tuple(
        ns_structure_constants(*momenta, precision=precision)
        for momenta in (
            (p1, d2, d1),
            (p2, d3, p1),
            (d5, d4, p2),
        )
    )  # type: ignore[return-value]


def ns_component_structure_product(
    form_weights: Sequence[Sequence[complex]],
    component_parities: Sequence[int],
) -> complex:
    """Return the structure-constant product for one edge routing."""

    weights = _validate_form_weights(form_weights)
    routing = tuple(component_parities)
    if len(routing) != 2 or any(value not in (0, 1) for value in routing):
        raise ValueError("component_parities must be a pair of zeroes or ones")
    parity1, parity2 = routing
    return (
        weights[0][parity1]
        * weights[1][parity1 ^ parity2]
        * weights[2][parity2]
    )


def _virasoro_degenerate_weight(b: complex, r: int, s: int) -> complex:
    charge = b + 1.0 / b
    return (charge * charge - (r * b + s / b) ** 2) / 4.0


def _virasoro_inverse_norm(
    b: complex,
    r: int,
    s: int,
    pole_tolerance: float,
) -> complex:
    result = 0.5 + 0.0j
    for m in range(1 - r, r + 1):
        for n in range(1 - s, s + 1):
            if (m, n) in ((0, 0), (r, s)):
                continue
            denominator = m * b + n / b
            if abs(denominator) <= pole_tolerance:
                raise ZeroDivisionError(
                    "resonant Virasoro inverse norm; take the assembled finite part"
                )
            result /= denominator
    return result


def _virasoro_fusion(
    b: complex,
    r: int,
    s: int,
    first_weight: complex,
    second_weight: complex,
) -> complex:
    charge = b + 1.0 / b
    lambda1 = cmath.sqrt(charge * charge - 4.0 * first_weight)
    lambda2 = cmath.sqrt(charge * charge - 4.0 * second_weight)
    result = 1.0 + 0.0j
    for p in range(1 - r, r, 2):
        for q in range(1 - s, s, 2):
            shift = p * b + q / b
            result *= (lambda1 + lambda2 + shift) / 2.0
            result *= (lambda1 - lambda2 + shift) / 2.0
    return result


class VirasoroSphereFivePointHRecursion:
    """Generic-``b`` ordinary-Virasoro five-point ``h``-recursion."""

    def __init__(
        self,
        *,
        b: complex,
        internal_weights: Sequence[complex],
        external_weights: Sequence[complex],
        pole_tolerance: float = 1.0e-12,
    ) -> None:
        self.b = complex(b)
        if self.b == 0:
            raise ValueError("b must be nonzero")
        self.c = virasoro_central_charge(self.b)
        self.internal_weights, self.external_weights = _validate_weights(
            internal_weights, external_weights
        )
        self.pole_tolerance = float(pole_tolerance)
        if not math.isfinite(self.pole_tolerance) or self.pole_tolerance <= 0:
            raise ValueError("pole_tolerance must be finite and positive")

    def _residue(
        self,
        r: int,
        s: int,
        first_weight: complex,
        second_weight: complex,
        first_external: complex,
        second_external: complex,
    ) -> complex:
        return (
            _virasoro_inverse_norm(self.b, r, s, self.pole_tolerance)
            * _virasoro_fusion(
                self.b, r, s, first_weight, first_external
            )
            * _virasoro_fusion(
                self.b, r, s, second_weight, second_external
            )
        )

    @lru_cache(maxsize=None)
    def _coefficient(
        self,
        level1: int,
        level2: int,
        h: complex,
        difference: complex,
        endpoint1_offset: complex,
        endpoint5_offset: complex,
    ) -> complex:
        if level1 < 0 or level2 < 0:
            return 0.0j
        result = 1.0 + 0.0j if level1 == level2 == 0 else 0.0j
        _, d2, d3, d4, _ = self.external_weights

        for r in range(1, level1 + 1):
            for s in range(1, level1 // r + 1):
                null_level = r * s
                if null_level > level1:
                    continue
                degenerate = _virasoro_degenerate_weight(self.b, r, s)
                denominator = h - degenerate
                scale = max(1.0, abs(h), abs(degenerate))
                if abs(denominator) <= self.pole_tolerance * scale:
                    raise FivePointRecursionPole(
                        f"edge 1 hits Virasoro ({r},{s}) Kac pole"
                    )
                residue = self._residue(
                    r,
                    s,
                    degenerate + endpoint1_offset,
                    degenerate + difference,
                    d2,
                    d3,
                )
                tail = self._coefficient(
                    level1 - null_level,
                    level2,
                    degenerate + null_level,
                    difference - null_level,
                    endpoint1_offset - null_level,
                    endpoint5_offset - null_level,
                )
                result += residue * tail / denominator

        for r in range(1, level2 + 1):
            for s in range(1, level2 // r + 1):
                null_level = r * s
                if null_level > level2:
                    continue
                degenerate = _virasoro_degenerate_weight(self.b, r, s)
                denominator = h + difference - degenerate
                scale = max(1.0, abs(h + difference), abs(degenerate))
                if abs(denominator) <= self.pole_tolerance * scale:
                    raise FivePointRecursionPole(
                        f"edge 2 hits Virasoro ({r},{s}) Kac pole"
                    )
                residue = self._residue(
                    r,
                    s,
                    degenerate - difference + endpoint5_offset,
                    degenerate - difference,
                    d4,
                    d3,
                )
                tail = self._coefficient(
                    level1,
                    level2 - null_level,
                    degenerate - difference,
                    difference + null_level,
                    endpoint1_offset,
                    endpoint5_offset,
                )
                result += residue * tail / denominator
        return complex(result)

    def coefficient(self, level1: int, level2: int) -> complex:
        """Return one stripped coefficient."""

        level1, level2 = _validate_cutoffs((level1, level2))
        h1, h2 = self.internal_weights
        d1, _, _, _, d5 = self.external_weights
        return self._coefficient(
            level1,
            level2,
            h1,
            h2 - h1,
            d1 - h1,
            d5 - h1,
        )

    def coefficients(
        self, maximum_levels: int | Sequence[int]
    ) -> dict[LevelPair, complex]:
        """Return a rectangular coefficient table."""

        cutoffs = _validate_cutoffs(maximum_levels)
        return {
            (level1, level2): self.coefficient(level1, level2)
            for level1 in range(cutoffs[0] + 1)
            for level2 in range(cutoffs[1] + 1)
        }

    def series(self, maximum_levels: int | Sequence[int]) -> SphereFivePointSeries:
        """Return the recursive coefficients in the common series wrapper."""

        cutoffs = _validate_cutoffs(maximum_levels)
        return SphereFivePointSeries(
            coefficients=self.coefficients(cutoffs),
            sector="Virasoro",
            method="h-recursion",
            c=self.c,
            internal_weights=self.internal_weights,
            external_weights=self.external_weights,
            external_words=(EMPTY_WORD,) * 5,
            maximum_levels=cutoffs,
            gram_condition_numbers={},
        )


class NSSphereFivePointHRecursion:
    r"""Generic-``b`` NS five-point ``h``-recursion for primary external legs.

    ``component_parities=(p1,p2)`` selects coefficients whose edge
    twice-levels have parities ``p1`` and ``p2``.  The corresponding three
    trinion forms are ``(p1, p1 xor p2, p2)``.  Removing an odd null state
    flips the routing on that edge in the recursive tail.
    """

    def __init__(
        self,
        *,
        b: complex,
        internal_weights: Sequence[complex],
        external_weights: Sequence[complex],
        component_parities: Sequence[int] = (0, 0),
        pole_tolerance: float = 1.0e-12,
        digits: int = 40,
    ) -> None:
        self.b = complex(b)
        if self.b == 0:
            raise ValueError("b must be nonzero")
        self.c = complex(sp.N(c_xi(sp.sympify(self.b)), digits))
        self.internal_weights, self.external_weights = _validate_weights(
            internal_weights, external_weights
        )
        routing = tuple(component_parities)
        if len(routing) != 2 or any(value not in (0, 1) for value in routing):
            raise ValueError("component_parities must be a pair of zeroes or ones")
        self.component_parities = routing  # type: ignore[assignment]
        self.pole_tolerance = float(pole_tolerance)
        if not math.isfinite(self.pole_tolerance) or self.pole_tolerance <= 0:
            raise ValueError("pole_tolerance must be finite and positive")
        self.digits = int(digits)
        if self.digits < 16:
            raise ValueError("digits must be at least 16")

    def _fusion(
        self,
        r: int,
        s: int,
        tail_form_parity: int,
        middle_weight: complex,
        spectator_weight: complex,
    ) -> complex:
        # This is the native-complex specialization of ns_fusion.sigma.
        # Keeping it out of SymPy is crucial: it is evaluated at every node
        # of every finite-part contour sample.
        polynomial_index = (
            tail_form_parity
            if (r * s) % 2 == 0
            else 1 - tail_form_parity
        )
        charge = self.b + 1.0 / self.b
        lambda_middle = cmath.sqrt(
            charge * charge - 8.0 * middle_weight
        )
        lambda_spectator = cmath.sqrt(
            charge * charge - 8.0 * spectator_weight
        )
        target = 2 if polynomial_index == 0 else 0
        denominator = 2.0 * math.sqrt(2.0)
        result = 1.0 + 0.0j
        for p in range(1 - r, r, 2):
            for q in range(1 - s, s, 2):
                if (p + q - (r + s)) % 4 != target:
                    continue
                shift = p * self.b + q / self.b
                result *= (
                    lambda_middle - lambda_spectator + shift
                ) / denominator
                result *= (
                    lambda_middle + lambda_spectator + shift
                ) / denominator
        return result

    def _inverse_norm(self, r: int, s: int) -> complex:
        result = 0.5 + 0.0j
        for p in range(1 - r, r + 1):
            for q in range(1 - s, s + 1):
                if (p + q) % 2 or (p, q) in ((0, 0), (r, s)):
                    continue
                denominator = (p * self.b + q / self.b) / math.sqrt(2.0)
                if abs(denominator) <= self.pole_tolerance:
                    raise ZeroDivisionError(
                        "resonant NS inverse norm; take the assembled finite part"
                    )
                result /= denominator
        return result

    @lru_cache(maxsize=None)
    def _coefficient(
        self,
        twice_level1: int,
        twice_level2: int,
        h: complex,
        difference: complex,
        endpoint1_offset: complex,
        endpoint5_offset: complex,
        parity1: int,
        parity2: int,
    ) -> complex:
        if twice_level1 < 0 or twice_level2 < 0:
            return 0.0j
        if twice_level1 % 2 != parity1 or twice_level2 % 2 != parity2:
            return 0.0j
        result = (
            1.0 + 0.0j
            if twice_level1 == twice_level2 == parity1 == parity2 == 0
            else 0.0j
        )
        _, d2, d3, d4, _ = self.external_weights

        for r in range(1, twice_level1 + 1):
            for s in range(1, twice_level1 // r + 1):
                product = r * s
                if product > twice_level1 or (r + s) % 2:
                    continue
                degenerate = complex(sp.N(ns_h_rs(r, s, sp.sympify(self.b)), self.digits))
                denominator = h - degenerate
                scale = max(1.0, abs(h), abs(degenerate))
                if abs(denominator) <= self.pole_tolerance * scale:
                    raise FivePointRecursionPole(
                        f"edge 1 hits NS ({r},{s}) Kac pole"
                    )
                epsilon = product % 2
                next_parity1 = parity1 ^ epsilon
                endpoint_form = next_parity1
                central_form = next_parity1 ^ parity2
                residue = self._inverse_norm(r, s)
                residue *= self._fusion(
                    r,
                    s,
                    endpoint_form,
                    d2,
                    degenerate + endpoint1_offset,
                )
                residue *= self._fusion(
                    r,
                    s,
                    central_form,
                    d3,
                    degenerate + difference,
                )
                half_level = product / 2.0
                tail = self._coefficient(
                    twice_level1 - product,
                    twice_level2,
                    degenerate + half_level,
                    difference - half_level,
                    endpoint1_offset - half_level,
                    endpoint5_offset - half_level,
                    next_parity1,
                    parity2,
                )
                result += residue * tail / denominator

        for r in range(1, twice_level2 + 1):
            for s in range(1, twice_level2 // r + 1):
                product = r * s
                if product > twice_level2 or (r + s) % 2:
                    continue
                degenerate = complex(sp.N(ns_h_rs(r, s, sp.sympify(self.b)), self.digits))
                denominator = h + difference - degenerate
                scale = max(1.0, abs(h + difference), abs(degenerate))
                if abs(denominator) <= self.pole_tolerance * scale:
                    raise FivePointRecursionPole(
                        f"edge 2 hits NS ({r},{s}) Kac pole"
                    )
                epsilon = product % 2
                next_parity2 = parity2 ^ epsilon
                central_form = parity1 ^ next_parity2
                endpoint_form = next_parity2
                residue = self._inverse_norm(r, s)
                residue *= self._fusion(
                    r,
                    s,
                    central_form,
                    d3,
                    degenerate - difference,
                )
                residue *= self._fusion(
                    r,
                    s,
                    endpoint_form,
                    d4,
                    degenerate - difference + endpoint5_offset,
                )
                half_level = product / 2.0
                tail = self._coefficient(
                    twice_level1,
                    twice_level2 - product,
                    degenerate - difference,
                    difference + half_level,
                    endpoint1_offset,
                    endpoint5_offset,
                    parity1,
                    next_parity2,
                )
                result += residue * tail / denominator
        return complex(result)

    def coefficient(self, twice_level1: int, twice_level2: int) -> complex:
        """Return one stripped coefficient in the selected component."""

        twice_level1, twice_level2 = _validate_cutoffs(
            (twice_level1, twice_level2)
        )
        h1, h2 = self.internal_weights
        d1, _, _, _, d5 = self.external_weights
        return self._coefficient(
            twice_level1,
            twice_level2,
            h1,
            h2 - h1,
            d1 - h1,
            d5 - h1,
            self.component_parities[0],
            self.component_parities[1],
        )

    def coefficients(
        self, maximum_twice_levels: int | Sequence[int]
    ) -> dict[LevelPair, complex]:
        """Return the selected component on a rectangular twice-level grid."""

        cutoffs = _validate_cutoffs(maximum_twice_levels)
        return {
            (level1, level2): self.coefficient(level1, level2)
            for level1 in range(cutoffs[0] + 1)
            for level2 in range(cutoffs[1] + 1)
            if level1 % 2 == self.component_parities[0]
            and level2 % 2 == self.component_parities[1]
        }

    def series(
        self, maximum_twice_levels: int | Sequence[int]
    ) -> SphereFivePointSeries:
        """Return the recursive component in the common series wrapper."""

        cutoffs = _validate_cutoffs(maximum_twice_levels)
        return SphereFivePointSeries(
            coefficients=self.coefficients(cutoffs),
            sector="NS",
            method="h-recursion",
            c=self.c,
            internal_weights=self.internal_weights,
            external_weights=self.external_weights,
            external_words=(EMPTY_WORD,) * 5,
            maximum_levels=cutoffs,
            gram_condition_numbers={},
        )


class _SelfDualSphereFivePointBase:
    """Shared coefficientwise ``b=1`` finite-part cache."""

    sector: Sector

    def __init__(
        self,
        *,
        internal_momenta: Sequence[complex],
        external_momenta: Sequence[complex],
        radius: float,
        check_radius: float,
        samples: int,
    ) -> None:
        internal = tuple(complex(value) for value in internal_momenta)
        external = tuple(complex(value) for value in external_momenta)
        if len(internal) != 2 or len(external) != 5:
            raise ValueError("two internal and five external momenta are required")
        self.internal_momenta = internal
        self.external_momenta = external
        self.radius = float(radius)
        self.check_radius = float(check_radius)
        self.samples = int(samples)
        self._cache: dict[
            tuple[int, int],
            tuple[
                dict[LevelPair, complex],
                dict[LevelPair, FinitePartDiagnostics],
            ],
        ] = {}

    def _table_at(
        self, b: complex, cutoffs: tuple[int, int]
    ) -> Mapping[LevelPair, complex]:
        raise NotImplementedError

    def _data(
        self, maximum_levels: int | Sequence[int]
    ) -> tuple[
        dict[LevelPair, complex],
        dict[LevelPair, FinitePartDiagnostics],
    ]:
        cutoffs = _validate_cutoffs(maximum_levels)
        if cutoffs not in self._cache:
            keys = tuple(
                (level1, level2)
                for level1 in range(cutoffs[0] + 1)
                for level2 in range(cutoffs[1] + 1)
                if self._include_key(level1, level2)
            )
            self._cache[cutoffs] = dictionary_finite_part(
                lambda b: self._table_at(b, cutoffs),
                keys=keys,
                radius=self.radius,
                check_radius=self.check_radius,
                samples=self.samples,
            )
        return self._cache[cutoffs]

    def _include_key(self, level1: int, level2: int) -> bool:
        return True

    def coefficients(
        self, maximum_levels: int | Sequence[int]
    ) -> dict[LevelPair, complex]:
        """Return the assembled self-dual coefficient table."""

        return dict(self._data(maximum_levels)[0])

    def diagnostics(
        self, maximum_levels: int | Sequence[int]
    ) -> dict[LevelPair, FinitePartDiagnostics]:
        """Return the independent-radius diagnostic for each coefficient."""

        return dict(self._data(maximum_levels)[1])


class SelfDualVirasoroSphereFivePointHRecursion(_SelfDualSphereFivePointBase):
    """Assembled ``b=1`` finite part of the Virasoro five-point recursion."""

    sector: Sector = "Virasoro"

    def __init__(
        self,
        *,
        internal_momenta: Sequence[complex],
        external_momenta: Sequence[complex],
        radius: float = 0.035,
        check_radius: float = 0.045,
        samples: int = 24,
    ) -> None:
        super().__init__(
            internal_momenta=internal_momenta,
            external_momenta=external_momenta,
            radius=radius,
            check_radius=check_radius,
            samples=samples,
        )

    def _table_at(
        self, b: complex, cutoffs: tuple[int, int]
    ) -> Mapping[LevelPair, complex]:
        return VirasoroSphereFivePointHRecursion(
            b=b,
            internal_weights=tuple(
                virasoro_liouville_weight(momentum, b)
                for momentum in self.internal_momenta
            ),
            external_weights=tuple(
                virasoro_liouville_weight(momentum, b)
                for momentum in self.external_momenta
            ),
        ).coefficients(cutoffs)

    def series(
        self, maximum_levels: int | Sequence[int]
    ) -> SphereFivePointSeries:
        """Return the self-dual table in the common series wrapper."""

        cutoffs = _validate_cutoffs(maximum_levels)
        return SphereFivePointSeries(
            coefficients=self.coefficients(cutoffs),
            sector="Virasoro",
            method="finite-part h-recursion",
            c=virasoro_central_charge(1.0),
            internal_weights=tuple(
                virasoro_liouville_weight(momentum, 1.0)
                for momentum in self.internal_momenta
            ),  # type: ignore[arg-type]
            external_weights=tuple(
                virasoro_liouville_weight(momentum, 1.0)
                for momentum in self.external_momenta
            ),  # type: ignore[arg-type]
            external_words=(EMPTY_WORD,) * 5,
            maximum_levels=cutoffs,
            gram_condition_numbers={},
        )


class SelfDualNSSphereFivePointHRecursion(_SelfDualSphereFivePointBase):
    """Assembled ``b=1`` finite part of one NS five-point component."""

    sector: Sector = "NS"

    def __init__(
        self,
        *,
        internal_momenta: Sequence[complex],
        external_momenta: Sequence[complex],
        component_parities: Sequence[int] = (0, 0),
        radius: float = 0.035,
        check_radius: float = 0.045,
        samples: int = 24,
    ) -> None:
        routing = tuple(component_parities)
        if len(routing) != 2 or any(value not in (0, 1) for value in routing):
            raise ValueError("component_parities must be a pair of zeroes or ones")
        self.component_parities = routing
        super().__init__(
            internal_momenta=internal_momenta,
            external_momenta=external_momenta,
            radius=radius,
            check_radius=check_radius,
            samples=samples,
        )

    def _include_key(self, level1: int, level2: int) -> bool:
        return (
            level1 % 2 == self.component_parities[0]
            and level2 % 2 == self.component_parities[1]
        )

    def _table_at(
        self, b: complex, cutoffs: tuple[int, int]
    ) -> Mapping[LevelPair, complex]:
        return NSSphereFivePointHRecursion(
            b=b,
            internal_weights=tuple(
                ns_liouville_weight(momentum, b)
                for momentum in self.internal_momenta
            ),
            external_weights=tuple(
                ns_liouville_weight(momentum, b)
                for momentum in self.external_momenta
            ),
            component_parities=self.component_parities,
        ).coefficients(cutoffs)

    def series(
        self, maximum_twice_levels: int | Sequence[int]
    ) -> SphereFivePointSeries:
        """Return the self-dual component in the common series wrapper."""

        cutoffs = _validate_cutoffs(maximum_twice_levels)
        return SphereFivePointSeries(
            coefficients=self.coefficients(cutoffs),
            sector="NS",
            method="finite-part h-recursion",
            c=complex(c_xi(sp.Integer(1))),
            internal_weights=tuple(
                ns_liouville_weight(momentum, 1.0)
                for momentum in self.internal_momenta
            ),  # type: ignore[arg-type]
            external_weights=tuple(
                ns_liouville_weight(momentum, 1.0)
                for momentum in self.external_momenta
            ),  # type: ignore[arg-type]
            external_words=(EMPTY_WORD,) * 5,
            maximum_levels=cutoffs,
            gram_condition_numbers={},
        )


__all__ = [
    "EMPTY_WORD",
    "FivePointRecursionPole",
    "G_MINUS_HALF",
    "NSSphereFivePointHRecursion",
    "SelfDualNSSphereFivePointHRecursion",
    "SelfDualVirasoroSphereFivePointHRecursion",
    "SphereFivePointSeries",
    "VirasoroSphereFivePointHRecursion",
    "direct_ns_sphere_fivepoint_series",
    "direct_virasoro_sphere_fivepoint_series",
    "ns_component_structure_product",
    "ns_fivepoint_structure_weights",
    "ns_liouville_weight",
    "pco_external_words",
    "virasoro_central_charge",
    "virasoro_liouville_weight",
]
