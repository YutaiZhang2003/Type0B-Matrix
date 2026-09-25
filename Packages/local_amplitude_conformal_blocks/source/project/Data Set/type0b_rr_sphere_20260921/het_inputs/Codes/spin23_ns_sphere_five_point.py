#!/usr/bin/env python3
r"""NS sphere five-point blocks in the linear plumbing channel.

This module implements the block of Belavin--Geiko, arXiv:1806.09563,
section 3.2.  With external weights ``(d1,...,d5)`` and internal weights
``(h1,h2)``, the ordered trivalent graph is

.. math::

   (d_1,d_2)\longrightarrow h_1
   \xrightarrow{\ d_3\ }h_2\longrightarrow(d_4,d_5).

The two plumbing parameters are denoted by ``q1`` and ``q2``.  The returned
series is stripped of the primary plumbing powers and is therefore a finite
sum in ``q1**N1 * q2**N2``.

Two independent constructions are provided:

``direct_five_point_block_series``
    Contracts the two inverse NS Gram matrices with the three exact ordered
    descendant three-point tensors.  This is the finite-level definition and
    is the reference oracle.

``recursive_five_point_block_series``
    Implements the published fixed-internal-weight ``c``-recursion in both
    internal edges.  Its regular term is the exact ``osp(1|2)`` five-point
    light block, evaluated with the same Ward tensors as the direct oracle.

The paper does not give a five-point elliptic/internal-weight ``h`` recursion.
That separate problem is intentionally not represented by the ``recursion``
label used here.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, Mapping, Sequence

import numpy as np
import sympy as sp
from scipy.linalg import lu_factor, lu_solve

from ns_algebra.ns_fusion import A_rs, J_rs, b_rs, c_rs, h_rs, sigma
from ns_algebra.ns_sca import L, Word, gram_matrix
from ns_algebra.ns_three_point_tensor import ns_three_point
from spin23_ns_sphere_blocks import EMPTY_WORD, G_MINUS_HALF, RecursionPoleCollision


FivePointMethod = Literal["direct", "global_beta", "c_recursion"]
EdgeOrder = Literal["left_first", "right_first"]


def _validate_parity(value: int, name: str) -> int:
    if value not in (0, 1):
        raise ValueError(f"{name} must be 0 or 1")
    return int(value)


def _validate_cutoffs(value: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(value, int):
        cutoffs = (value, value)
    else:
        cutoffs = tuple(value)
        if len(cutoffs) != 2:
            raise ValueError("maximum_twice_levels must contain two cutoffs")
    if any(not isinstance(item, int) for item in cutoffs):
        raise TypeError("five-point level cutoffs must be integers")
    if any(item < 0 for item in cutoffs):
        raise ValueError("five-point level cutoffs must be nonnegative")
    return int(cutoffs[0]), int(cutoffs[1])


def _validate_weights(
    external_weights: Sequence[complex],
    internal_weights: Sequence[complex],
) -> tuple[tuple[complex, ...], tuple[complex, complex]]:
    external = tuple(complex(value) for value in external_weights)
    internal = tuple(complex(value) for value in internal_weights)
    if len(external) != 5:
        raise ValueError("five external weights are required")
    if len(internal) != 2:
        raise ValueError("two internal weights are required")
    return external, (internal[0], internal[1])


def five_point_external_words(
    external_alphas: Sequence[int] = (0, 0, 0),
) -> tuple[Word, Word, Word, Word, Word]:
    r"""Return the published lower/upper component pattern.

    Legs 1 and 5 are fixed to lower components.  ``external_alphas`` are the
    component labels on legs 2, 3, and 4; an upper component is represented
    by ``G_-1/2``.
    """

    alphas = tuple(external_alphas)
    if len(alphas) != 3:
        raise ValueError("external_alphas must contain the labels for legs 2, 3, 4")
    alpha2, alpha3, alpha4 = (
        _validate_parity(value, f"alpha{index}")
        for index, value in zip((2, 3, 4), alphas)
    )
    word = lambda alpha: G_MINUS_HALF if alpha else EMPTY_WORD
    return (
        EMPTY_WORD,
        word(alpha2),
        word(alpha3),
        word(alpha4),
        EMPTY_WORD,
    )


def _as_numeric(expression: sp.Expr | complex, digits: int) -> complex:
    value = complex(sp.N(sp.sympify(expression), digits))
    if not math.isfinite(value.real) or not math.isfinite(value.imag):
        raise ArithmeticError(f"non-finite NS five-point entry {value!r}")
    return value


@lru_cache(maxsize=None)
def _gram_data(
    twice_descendant_level: int,
    c: complex,
    h: complex,
    digits: int,
) -> tuple[tuple[Word, ...], np.ndarray, np.ndarray, float]:
    basis, gram = gram_matrix(
        twice_descendant_level,
        h=sp.sympify(h),
        c=sp.sympify(c),
    )
    matrix = np.asarray(
        [
            [_as_numeric(gram[row, column], digits) for column in range(gram.cols)]
            for row in range(gram.rows)
        ],
        dtype=np.complex128,
    )
    condition = float(np.linalg.cond(matrix))
    lu, pivots = lu_factor(matrix, check_finite=False)
    return tuple(basis), lu, pivots, condition


def _restricted_gram_data(
    twice_descendant_level: int,
    c: complex,
    h: complex,
    digits: int,
    basis_override: Sequence[Word],
) -> tuple[tuple[Word, ...], np.ndarray, np.ndarray, float]:
    basis, gram = gram_matrix(
        twice_descendant_level,
        h=sp.sympify(h),
        c=sp.sympify(c),
    )
    requested = tuple(tuple(word) for word in basis_override)
    indices = [basis.index(word) for word in requested]
    restricted = gram.extract(indices, indices)
    matrix = np.asarray(
        [
            [
                _as_numeric(restricted[row, column], digits)
                for column in range(restricted.cols)
            ]
            for row in range(restricted.rows)
        ],
        dtype=np.complex128,
    )
    condition = float(np.linalg.cond(matrix))
    lu, pivots = lu_factor(matrix, check_finite=False)
    return requested, lu, pivots, condition


def _direct_coefficient(
    twice_level_left: int,
    twice_level_right: int,
    *,
    c: complex,
    internal_weights: tuple[complex, complex],
    external_weights: tuple[complex, ...],
    external_words: tuple[Word, Word, Word, Word, Word],
    digits: int,
    condition_limit: float,
    left_basis_override: Sequence[Word] | None = None,
    right_basis_override: Sequence[Word] | None = None,
) -> tuple[complex, tuple[float, float]]:
    h_left, h_right = internal_weights
    if left_basis_override is None:
        left_basis, left_lu, left_pivots, left_condition = _gram_data(
            twice_level_left, complex(c), complex(h_left), int(digits)
        )
    else:
        left_basis, left_lu, left_pivots, left_condition = _restricted_gram_data(
            twice_level_left,
            complex(c),
            complex(h_left),
            int(digits),
            left_basis_override,
        )
    if right_basis_override is None:
        right_basis, right_lu, right_pivots, right_condition = _gram_data(
            twice_level_right, complex(c), complex(h_right), int(digits)
        )
    else:
        right_basis, right_lu, right_pivots, right_condition = _restricted_gram_data(
            twice_level_right,
            complex(c),
            complex(h_right),
            int(digits),
            right_basis_override,
        )
    for name, condition in (
        ("left", left_condition),
        ("right", right_condition),
    ):
        if not math.isfinite(condition) or condition > condition_limit:
            raise np.linalg.LinAlgError(
                f"{name} NS Gram matrix has condition number {condition:.3e}, "
                f"above the limit {condition_limit:.3e}"
            )

    d1, d2, d3, d4, d5 = external_weights
    word1, word2, word3, word4, word5 = external_words
    left_endpoint = np.asarray(
        [
            _as_numeric(
                ns_three_point(
                    word1,
                    word2,
                    internal_word,
                    h_infinity=d1,
                    h_middle=d2,
                    h_zero=h_left,
                    c=c,
                ),
                digits,
            )
            for internal_word in left_basis
        ],
        dtype=np.complex128,
    )
    middle = np.asarray(
        [
            [
                _as_numeric(
                    ns_three_point(
                        left_word,
                        word3,
                        right_word,
                        h_infinity=h_left,
                        h_middle=d3,
                        h_zero=h_right,
                        c=c,
                    ),
                    digits,
                )
                for right_word in right_basis
            ]
            for left_word in left_basis
        ],
        dtype=np.complex128,
    )
    right_endpoint = np.asarray(
        [
            _as_numeric(
                ns_three_point(
                    internal_word,
                    word4,
                    word5,
                    h_infinity=h_right,
                    h_middle=d4,
                    h_zero=d5,
                    c=c,
                ),
                digits,
            )
            for internal_word in right_basis
        ],
        dtype=np.complex128,
    )

    left_propagated_middle = lu_solve(
        (left_lu, left_pivots), middle, check_finite=False
    )
    right_propagated_endpoint = lu_solve(
        (right_lu, right_pivots), right_endpoint, check_finite=False
    )
    coefficient = left_endpoint @ left_propagated_middle @ right_propagated_endpoint
    return complex(coefficient), (left_condition, right_condition)


@dataclass(frozen=True)
class FivePointBlockSeries:
    """One finite bivariate component of the linear-channel NS block."""

    coefficients: Mapping[tuple[int, int], complex]
    component_parities: tuple[int, int]
    method: FivePointMethod
    c: complex
    internal_weights: tuple[complex, complex]
    external_weights: tuple[complex, complex, complex, complex, complex]
    external_alphas: tuple[int, int, int]
    maximum_twice_levels: tuple[int, int]
    gram_condition_numbers: Mapping[tuple[int, int], tuple[float, float]]

    def descendant_value(self, q1: complex, q2: complex) -> complex:
        """Evaluate the plumbing series without primary sewing powers."""

        q1 = complex(q1)
        q2 = complex(q2)
        if q1 == 0 or q2 == 0:
            return sum(
                coefficient
                * (0.0j if q1 == 0 and left_level else q1 ** (left_level / 2))
                * (0.0j if q2 == 0 and right_level else q2 ** (right_level / 2))
                for (left_level, right_level), coefficient in self.coefficients.items()
            )
        log_q1 = cmath.log(q1)
        log_q2 = cmath.log(q2)
        return sum(
            coefficient
            * cmath.exp(0.5 * left_level * log_q1)
            * cmath.exp(0.5 * right_level * log_q2)
            for (left_level, right_level), coefficient in self.coefficients.items()
        )


def direct_five_point_block_series(
    *,
    c: complex,
    internal_weights: Sequence[complex],
    external_weights: Sequence[complex],
    maximum_twice_levels: int | Sequence[int],
    component_parities: Sequence[int] = (0, 0),
    external_alphas: Sequence[int] = (0, 0, 0),
    digits: int = 50,
    condition_limit: float = 1.0e13,
) -> FivePointBlockSeries:
    """Construct the inverse-Gram five-point block through finite levels."""

    external, internal = _validate_weights(external_weights, internal_weights)
    cutoffs = _validate_cutoffs(maximum_twice_levels)
    parities_tuple = tuple(component_parities)
    if len(parities_tuple) != 2:
        raise ValueError("component_parities must contain two entries")
    parities = (
        _validate_parity(parities_tuple[0], "left component parity"),
        _validate_parity(parities_tuple[1], "right component parity"),
    )
    alphas_tuple = tuple(external_alphas)
    words = five_point_external_words(alphas_tuple)
    alphas = tuple(_validate_parity(value, "external alpha") for value in alphas_tuple)

    coefficients: dict[tuple[int, int], complex] = {}
    conditions: dict[tuple[int, int], tuple[float, float]] = {}
    for left_level in range(parities[0], cutoffs[0] + 1, 2):
        for right_level in range(parities[1], cutoffs[1] + 1, 2):
            coefficient, condition = _direct_coefficient(
                left_level,
                right_level,
                c=complex(c),
                internal_weights=internal,
                external_weights=external,
                external_words=words,
                digits=int(digits),
                condition_limit=float(condition_limit),
            )
            coefficients[(left_level, right_level)] = coefficient
            conditions[(left_level, right_level)] = condition
    return FivePointBlockSeries(
        coefficients=coefficients,
        component_parities=parities,
        method="direct",
        c=complex(c),
        internal_weights=internal,
        external_weights=external,  # type: ignore[arg-type]
        external_alphas=alphas,  # type: ignore[arg-type]
        maximum_twice_levels=cutoffs,
        gram_condition_numbers=conditions,
    )


def _global_word(twice_descendant_level: int) -> Word:
    integer_level, fermion = divmod(twice_descendant_level, 2)
    return (L(-1),) * integer_level + (G_MINUS_HALF if fermion else EMPTY_WORD)


def _global_seed(
    *,
    c: complex,
    internal_weights: tuple[complex, complex],
    external_weights: tuple[complex, ...],
    external_words: tuple[Word, Word, Word, Word, Word],
    cutoffs: tuple[int, int],
    parities: tuple[int, int],
    digits: int,
) -> dict[tuple[int, int], complex]:
    result: dict[tuple[int, int], complex] = {}
    for left_level in range(parities[0], cutoffs[0] + 1, 2):
        for right_level in range(parities[1], cutoffs[1] + 1, 2):
            coefficient, _ = _direct_coefficient(
                left_level,
                right_level,
                c=c,
                internal_weights=internal_weights,
                external_weights=external_weights,
                external_words=external_words,
                digits=digits,
                condition_limit=float("inf"),
                left_basis_override=(_global_word(left_level),),
                right_basis_override=(_global_word(right_level),),
            )
            result[(left_level, right_level)] = coefficient
    return result


def global_five_point_beta_series(
    *,
    internal_weights: Sequence[complex],
    external_weights: Sequence[complex],
    maximum_twice_levels: int | Sequence[int],
    beta_labels: Sequence[int] = (0, 0),
    external_alphas: Sequence[int] = (0, 0, 0),
    digits: int = 50,
) -> FivePointBlockSeries:
    r"""Return one ``(beta1,beta2)`` component of the light block.

    The labels ``beta_i in {0,1}`` select integer/half-integer global
    descendants on the two internal edges, exactly as in eq. (3.14) of
    Belavin--Geiko.  This is the ``osp(1|2)`` regular term of the five-point
    ``c``-recursion.  The literature does not call it a separate
    "beta-recursion"; exposing it here keeps the parity-resolved layer
    explicit.
    """

    external, internal = _validate_weights(external_weights, internal_weights)
    cutoffs = _validate_cutoffs(maximum_twice_levels)
    beta_tuple = tuple(beta_labels)
    if len(beta_tuple) != 2:
        raise ValueError("beta_labels must contain two entries")
    betas = (
        _validate_parity(beta_tuple[0], "beta1"),
        _validate_parity(beta_tuple[1], "beta2"),
    )
    alphas_tuple = tuple(external_alphas)
    if len(alphas_tuple) != 3:
        raise ValueError("external_alphas must contain three entries")
    alphas = tuple(_validate_parity(value, "external alpha") for value in alphas_tuple)
    words = five_point_external_words(alphas)
    coefficients = _global_seed(
        c=0.0,
        internal_weights=internal,
        external_weights=external,
        external_words=words,
        cutoffs=cutoffs,
        parities=betas,
        digits=int(digits),
    )
    return FivePointBlockSeries(
        coefficients=coefficients,
        component_parities=betas,
        method="global_beta",
        c=0.0j,
        internal_weights=internal,
        external_weights=external,  # type: ignore[arg-type]
        external_alphas=alphas,  # type: ignore[arg-type]
        maximum_twice_levels=cutoffs,
        gram_condition_numbers={},
    )


@dataclass(frozen=True)
class FivePointHPoleData:
    """Fixed-central-charge singular data for one five-point internal edge."""

    edge: Literal["left", "right"]
    r: int
    s: int
    h_pole: complex
    residue: complex
    null_twice_level: int
    shifted_component_parities: tuple[int, int]


def five_point_h_pole_data(
    *,
    b: complex,
    edge: Literal["left", "right"],
    r: int,
    s: int,
    internal_weights: Sequence[complex],
    external_weights: Sequence[complex],
    component_parities: Sequence[int] = (0, 0),
    external_alphas: Sequence[int] = (0, 0, 0),
    digits: int = 50,
) -> FivePointHPoleData:
    """Return the derivable singular layer of a five-point h-recursion.

    The all-orders joint large-internal-weight seed is not published and is
    deliberately outside this function.  All upper/lower external patterns
    and both internal beta labels are supported by the pole data.
    """

    external, internal = _validate_weights(external_weights, internal_weights)
    parities_tuple = tuple(component_parities)
    if len(parities_tuple) != 2:
        raise ValueError("component_parities must contain two entries")
    parity_left = _validate_parity(parities_tuple[0], "left component parity")
    parity_right = _validate_parity(parities_tuple[1], "right component parity")
    alphas_tuple = tuple(external_alphas)
    if len(alphas_tuple) != 3:
        raise ValueError("external_alphas must contain three entries")
    alpha2, alpha3, alpha4 = (
        _validate_parity(value, "external alpha") for value in alphas_tuple
    )
    if edge not in ("left", "right"):
        raise ValueError("edge must be 'left' or 'right'")

    b_symbol = sp.sympify(b)
    null_twice_level = r * s
    h_pole = _as_numeric(h_rs(r, s, b_symbol), digits)
    h_left, h_right = internal
    d1, d2, d3, d4, d5 = external
    inverse_norm = A_rs(r, s, b_symbol)
    if edge == "left":
        shifted_left = parity_left ^ (null_twice_level % 2)
        endpoint_component = alpha2 ^ shifted_left
        middle_component = shifted_left ^ alpha3 ^ parity_right
        residue_expression = (
            (-1) ** (null_twice_level * alpha3)
            * inverse_norm
            * sigma(r, s, endpoint_component, d2, d1, b_symbol)
            * sigma(r, s, middle_component, d3, h_right, b_symbol)
        )
        shifted_parities = (shifted_left, parity_right)
    else:
        shifted_right = parity_right ^ (null_twice_level % 2)
        middle_component = parity_left ^ alpha3 ^ shifted_right
        endpoint_component = shifted_right ^ alpha4
        residue_expression = (
            (-1) ** (null_twice_level * alpha4)
            * inverse_norm
            * sigma(r, s, middle_component, d3, h_left, b_symbol)
            * sigma(r, s, endpoint_component, d4, d5, b_symbol)
        )
        shifted_parities = (parity_left, shifted_right)
    return FivePointHPoleData(
        edge=edge,
        r=r,
        s=s,
        h_pole=h_pole,
        residue=_as_numeric(sp.cancel(residue_expression), digits),
        null_twice_level=null_twice_level,
        shifted_component_parities=shifted_parities,
    )


def _residue_data(
    *,
    edge: Literal["left", "right"],
    r: int,
    s: int,
    internal_weights: tuple[complex, complex],
    external_weights: tuple[complex, ...],
    external_alphas: tuple[int, int, int],
    component_parities: tuple[int, int],
    digits: int,
) -> tuple[complex, complex]:
    h_left, h_right = internal_weights
    d1, d2, d3, d4, d5 = external_weights
    alpha2, alpha3, alpha4 = external_alphas
    parity_left, parity_right = component_parities
    h_edge = h_left if edge == "left" else h_right
    b_at_pole = _as_numeric(b_rs(r, s, sp.sympify(h_edge)), digits)
    pole = _as_numeric(c_rs(r, s, sp.sympify(h_edge)), digits)
    b_symbol = sp.sympify(b_at_pole)
    jacobian_norm = _as_numeric(
        sp.cancel(J_rs(r, s, sp.sympify(h_edge)) * A_rs(r, s, b_symbol)),
        digits,
    )
    if edge == "left":
        shifted_left = parity_left ^ ((r * s) % 2)
        endpoint_component = alpha2 ^ shifted_left
        middle_component = shifted_left ^ alpha3 ^ parity_right
        endpoint = _as_numeric(
            sigma(r, s, endpoint_component, d2, d1, b_symbol), digits
        )
        middle = _as_numeric(
            sigma(r, s, middle_component, d3, h_right, b_symbol), digits
        )
        # ``ns_three_point`` packages the two normalized forms by total word
        # parity.  Translating the reflected odd form in (3.13) to that
        # convention gives this sign when the middle external leg is upper.
        reflection_sign = (-1) ** (r * s * alpha3)
    else:
        shifted_right = parity_right ^ ((r * s) % 2)
        middle_component = parity_left ^ alpha3 ^ shifted_right
        endpoint_component = shifted_right ^ alpha4
        middle = _as_numeric(
            sigma(r, s, middle_component, d3, h_left, b_symbol), digits
        )
        endpoint = _as_numeric(
            sigma(r, s, endpoint_component, d4, d5, b_symbol), digits
        )
        reflection_sign = (-1) ** (r * s * alpha4)
    return pole, reflection_sign * jacobian_norm * endpoint * middle


def _add_shifted_bivariate(
    target: dict[tuple[int, int], complex],
    source: Mapping[tuple[int, int], complex],
    shift: tuple[int, int],
    scale: complex,
) -> None:
    for (left_level, right_level), coefficient in source.items():
        key = (left_level + shift[0], right_level + shift[1])
        target[key] = target.get(key, 0.0j) + scale * coefficient


def recursive_five_point_block_series(
    *,
    c: complex,
    internal_weights: Sequence[complex],
    external_weights: Sequence[complex],
    maximum_twice_levels: int | Sequence[int],
    component_parities: Sequence[int] = (0, 0),
    external_alphas: Sequence[int] = (0, 0, 0),
    digits: int = 50,
    pole_tolerance: float = 1.0e-11,
    edge_order: EdgeOrder = "left_first",
) -> FivePointBlockSeries:
    r"""Construct the published two-edge five-point ``c``-recursion.

    ``edge_order`` changes only the deterministic summation order.  Agreement
    of the two choices is a useful check of the mixed-edge partial fractions;
    it is not a crossing equation between distinct pants decompositions.
    """

    external, initial_internal = _validate_weights(external_weights, internal_weights)
    cutoffs = _validate_cutoffs(maximum_twice_levels)
    parities_tuple = tuple(component_parities)
    if len(parities_tuple) != 2:
        raise ValueError("component_parities must contain two entries")
    initial_parities = (
        _validate_parity(parities_tuple[0], "left component parity"),
        _validate_parity(parities_tuple[1], "right component parity"),
    )
    alphas_tuple = tuple(external_alphas)
    if len(alphas_tuple) != 3:
        raise ValueError("external_alphas must contain three entries")
    alphas = tuple(_validate_parity(value, "external alpha") for value in alphas_tuple)
    words = five_point_external_words(alphas)
    if edge_order == "left_first":
        ordered_edges: tuple[Literal["left", "right"], ...] = ("left", "right")
    elif edge_order == "right_first":
        ordered_edges = ("right", "left")
    else:
        raise ValueError("edge_order must be 'left_first' or 'right_first'")

    @lru_cache(maxsize=None)
    def recurse(
        current_c: complex,
        h_left: complex,
        h_right: complex,
        remaining_left: int,
        remaining_right: int,
        parity_left: int,
        parity_right: int,
    ) -> tuple[tuple[tuple[int, int], complex], ...]:
        current_internal = (h_left, h_right)
        total = _global_seed(
            c=current_c,
            internal_weights=current_internal,
            external_weights=external,
            external_words=words,
            cutoffs=(remaining_left, remaining_right),
            parities=(parity_left, parity_right),
            digits=digits,
        )
        for edge in ordered_edges:
            remaining = remaining_left if edge == "left" else remaining_right
            for r in range(2, remaining + 1):
                for s in range(1, remaining // r + 1):
                    null_twice_level = r * s
                    if (r + s) % 2 or null_twice_level > remaining:
                        continue
                    pole, residue = _residue_data(
                        edge=edge,
                        r=r,
                        s=s,
                        internal_weights=current_internal,
                        external_weights=external,
                        external_alphas=alphas,  # type: ignore[arg-type]
                        component_parities=(parity_left, parity_right),
                        digits=digits,
                    )
                    denominator = current_c - pole
                    if abs(denominator) <= pole_tolerance * max(
                        1.0, abs(current_c), abs(pole)
                    ):
                        raise RecursionPoleCollision(
                            f"c={current_c!r} collides with the {edge} "
                            f"({r},{s}) pole {pole!r}"
                        )
                    odd_null = null_twice_level % 2
                    if edge == "left":
                        subblock = dict(
                            recurse(
                                pole,
                                h_left + null_twice_level / 2,
                                h_right,
                                remaining_left - null_twice_level,
                                remaining_right,
                                parity_left ^ odd_null,
                                parity_right,
                            )
                        )
                        shift = (null_twice_level, 0)
                    else:
                        subblock = dict(
                            recurse(
                                pole,
                                h_left,
                                h_right + null_twice_level / 2,
                                remaining_left,
                                remaining_right - null_twice_level,
                                parity_left,
                                parity_right ^ odd_null,
                            )
                        )
                        shift = (0, null_twice_level)
                    _add_shifted_bivariate(
                        total,
                        subblock,
                        shift,
                        residue / denominator,
                    )
        return tuple(sorted(total.items()))

    coefficients = dict(
        recurse(
            complex(c),
            initial_internal[0],
            initial_internal[1],
            cutoffs[0],
            cutoffs[1],
            initial_parities[0],
            initial_parities[1],
        )
    )
    return FivePointBlockSeries(
        coefficients=coefficients,
        component_parities=initial_parities,
        method="c_recursion",
        c=complex(c),
        internal_weights=initial_internal,
        external_weights=external,  # type: ignore[arg-type]
        external_alphas=alphas,  # type: ignore[arg-type]
        maximum_twice_levels=cutoffs,
        gram_condition_numbers={},
    )


def relative_five_point_series_difference(
    first: FivePointBlockSeries,
    second: FivePointBlockSeries,
) -> float:
    """Return the largest coefficientwise relative difference."""

    keys = set(first.coefficients) | set(second.coefficients)
    if not keys:
        return 0.0
    return max(
        abs(first.coefficients.get(key, 0.0j) - second.coefficients.get(key, 0.0j))
        / max(
            1.0,
            abs(first.coefficients.get(key, 0.0j)),
            abs(second.coefficients.get(key, 0.0j)),
        )
        for key in keys
    )


__all__ = [
    "FivePointBlockSeries",
    "FivePointHPoleData",
    "direct_five_point_block_series",
    "five_point_h_pole_data",
    "five_point_external_words",
    "global_five_point_beta_series",
    "recursive_five_point_block_series",
    "relative_five_point_series_difference",
]
