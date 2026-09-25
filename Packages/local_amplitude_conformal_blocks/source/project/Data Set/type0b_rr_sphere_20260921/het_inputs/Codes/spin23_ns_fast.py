#!/usr/bin/env python3
r"""Conditioned direct :math:`b=1` NS necklace blocks.

This module evaluates the same finite inverse-Gram/Ward definition as
:func:`spin23_genus1_blocks.direct_ns_torus_necklace_series`, specialized to
real nonnegative continuum momenta at ``b=1``.  Exact Gram and Ward
expressions can be generated offline; runtime evaluation then consists only
of NumPy arithmetic and parity-block-free Cholesky solves.  No recursive,
interpolated, or fitted conformal-block representation is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
import math
from typing import Callable, Sequence

import numpy as np
from scipy.linalg import solve_triangular
import sympy as sp

from ns_algebra.ns_sca import (
    G,
    Word as NSWord,
    fermion_parity,
    gram_matrix as ns_gram_matrix,
    pbw_basis,
    twice_level,
)
from ns_algebra.ns_three_point_tensor import ns_three_point
from spin23_genus1_blocks import (
    ExternalWeightPolynomialBlockSeries,
    TorusNecklaceBlockSeries,
)
from spin23_super_liouville_data import ns_weight


_B1_C_EXACT = sp.Rational(27, 2)
_B1_C = float(_B1_C_EXACT)
_B1_P = sp.symbols("ns_b1_P", real=True)
_B1_H = (1 + _B1_P**2) / 2
_VERTEX_P_LEFT, _VERTEX_H_NS, _VERTEX_P_RIGHT = sp.symbols(
    "ns_b1_P_left ns_b1_h_ns ns_b1_P_right",
    real=True,
)

try:
    from spin23_generated.ns_b1_kernels import (
        GENERATED_MAX_TWICE_LEVEL as _BASE_GENERATED_MAX_TWICE_LEVEL,
        evaluate_gram as _evaluate_base_generated_gram,
        evaluate_vertex_coefficients as _evaluate_base_generated_vertex_coefficients,
        has_vertex as _base_has_vertex,
    )
except ImportError:
    _BASE_GENERATED_MAX_TWICE_LEVEL = -1
    _evaluate_base_generated_gram = None
    _evaluate_base_generated_vertex_coefficients = None
    _base_has_vertex = None

try:
    from spin23_generated.ns_b1_level5_x3_kernels import (
        GENERATED_MAX_TWICE_LEVEL as _SPARSE_GENERATED_MAX_TWICE_LEVEL,
        evaluate_gram as _evaluate_sparse_generated_gram,
        evaluate_vertex_coefficients as _evaluate_sparse_generated_vertex_coefficients,
        has_gram as _sparse_has_gram,
        has_vertex as _sparse_has_vertex,
    )
except ImportError:
    _SPARSE_GENERATED_MAX_TWICE_LEVEL = -1
    _evaluate_sparse_generated_gram = None
    _evaluate_sparse_generated_vertex_coefficients = None
    _sparse_has_gram = None
    _sparse_has_vertex = None


GENERATED_MAX_TWICE_LEVEL = max(
    _BASE_GENERATED_MAX_TWICE_LEVEL,
    _SPARSE_GENERATED_MAX_TWICE_LEVEL,
)


@dataclass(frozen=True)
class FastNSEdgeFactor:
    """One conditioned NS sewing edge at fixed momentum and level."""

    basis: tuple[NSWord, ...]
    equilibrium_scale: np.ndarray
    cholesky_lower: np.ndarray
    lift_factor: float
    condition_number: float
    factorization_residual: float
    normalized_gram: np.ndarray
    equilibrated_gram: np.ndarray


def _validate_real_momentum(momentum: complex | float, name: str) -> float:
    value = complex(momentum)
    if not math.isfinite(value.real) or not math.isfinite(value.imag):
        raise ValueError(f"{name} must be finite")
    if abs(value.imag) > 1.0e-14 * max(1.0, abs(value.real)):
        raise ValueError(f"{name} must be real in the direct-fast backend")
    if value.real < 0.0:
        raise ValueError(f"{name} must be nonnegative in the direct-fast backend")
    return float(value.real)


def _validate_sign(sign: int, name: str) -> int:
    if sign not in (-1, 1):
        raise ValueError(f"{name} must be +1 or -1")
    return int(sign)


def _cutoff_tuple(
    maximum_twice_levels: int | Sequence[int],
    size: int,
) -> tuple[int, ...]:
    if isinstance(maximum_twice_levels, bool):
        raise TypeError("descendant cutoffs must be integers")
    if isinstance(maximum_twice_levels, int):
        cutoffs = (maximum_twice_levels,) * size
    else:
        cutoffs = tuple(maximum_twice_levels)
    if len(cutoffs) != size:
        raise ValueError("one descendant cutoff is required per edge")
    if any(
        isinstance(cutoff, bool) or not isinstance(cutoff, int)
        for cutoff in cutoffs
    ):
        raise TypeError("descendant cutoffs must be integers")
    if any(cutoff < 0 for cutoff in cutoffs):
        raise ValueError("descendant cutoffs must be nonnegative")
    return cutoffs


def _component_name(middle_word: NSWord) -> str | None:
    if not middle_word:
        return "PP"
    if (
        len(middle_word) == 1
        and middle_word[0].kind == "G"
        and middle_word[0].twice_index == -1
    ):
        return "GG"
    return None


def _has_generated_gram(twice_descendant_level: int) -> bool:
    if (
        _evaluate_base_generated_gram is not None
        and twice_descendant_level <= _BASE_GENERATED_MAX_TWICE_LEVEL
    ):
        return True
    return bool(
        _sparse_has_gram is not None
        and _sparse_has_gram(twice_descendant_level)
    )


def _has_generated_vertex(
    component: str,
    left_twice_level: int,
    right_twice_level: int,
) -> bool:
    if (
        _base_has_vertex is not None
        and _base_has_vertex(
            component,
            left_twice_level,
            right_twice_level,
        )
    ):
        return True
    return bool(
        _sparse_has_vertex is not None
        and _sparse_has_vertex(
            component,
            left_twice_level,
            right_twice_level,
        )
    )


@lru_cache(maxsize=None)
def _b1_ns_gram_expressions(
    twice_descendant_level: int,
) -> tuple[tuple[NSWord, ...], sp.Matrix]:
    """Return the exact ``b=1`` NS Gram matrix at one twice-level."""

    basis, matrix = ns_gram_matrix(
        twice_descendant_level,
        h=_B1_H,
        c=_B1_C_EXACT,
    )
    matrix = matrix.applyfunc(sp.cancel)
    return tuple(basis), matrix


@lru_cache(maxsize=None)
def _dynamic_gram_template(
    twice_descendant_level: int,
) -> tuple[tuple[NSWord, ...], Callable[[float], object]]:
    basis, matrix = _b1_ns_gram_expressions(twice_descendant_level)
    return basis, sp.lambdify(
        _B1_P,
        matrix.tolist(),
        modules="numpy",
        cse=True,
        docstring_limit=0,
    )


def _evaluate_gram(
    twice_descendant_level: int,
    momentum: float,
) -> tuple[tuple[NSWord, ...], np.ndarray]:
    basis = tuple(pbw_basis(twice_descendant_level))
    if (
        _evaluate_base_generated_gram is not None
        and twice_descendant_level <= _BASE_GENERATED_MAX_TWICE_LEVEL
    ):
        evaluated = _evaluate_base_generated_gram(
            twice_descendant_level,
            momentum,
        )
    elif (
        _evaluate_sparse_generated_gram is not None
        and _sparse_has_gram is not None
        and _sparse_has_gram(twice_descendant_level)
    ):
        evaluated = _evaluate_sparse_generated_gram(
            twice_descendant_level,
            momentum,
        )
    else:
        dynamic_basis, evaluator = _dynamic_gram_template(
            twice_descendant_level
        )
        if dynamic_basis != basis:
            raise AssertionError("dynamic NS Gram basis ordering changed")
        evaluated = evaluator(momentum)
    matrix = np.asarray(evaluated, dtype=np.float64)
    expected_shape = (len(basis), len(basis))
    if matrix.shape != expected_shape:
        matrix = np.reshape(matrix, expected_shape)
    return basis, matrix


@lru_cache(maxsize=16384)
def _fast_edge_factor(
    twice_descendant_level: int,
    momentum: float,
    lift_sign: int,
    condition_limit: float,
) -> FastNSEdgeFactor:
    """Evaluate, equilibrate, and Cholesky factor one NS Gram matrix."""

    p = _validate_real_momentum(momentum, "internal momentum")
    eta = _validate_sign(lift_sign, "edge lift sign")
    basis, gram = _evaluate_gram(twice_descendant_level, p)
    if not np.all(np.isfinite(gram)):
        raise ArithmeticError("the NS Gram matrix is non-finite")
    scale_norm = max(1.0, float(np.max(np.abs(gram))))
    symmetry_error = float(np.max(np.abs(gram - gram.T)))
    if symmetry_error > 5.0e-13 * scale_norm:
        raise ArithmeticError(
            f"the NS Gram matrix is not symmetric: {symmetry_error:.3e}"
        )
    diagonal = np.diag(gram)
    if np.any(diagonal <= 0.0):
        raise np.linalg.LinAlgError("the NS Gram matrix has nonpositive diagonal")
    equilibrium = 1.0 / np.sqrt(diagonal)
    equilibrated = equilibrium[:, None] * gram * equilibrium[None, :]
    lower = np.linalg.cholesky(equilibrated)
    condition = float(np.linalg.cond(equilibrated))
    if not math.isfinite(condition) or condition > condition_limit:
        raise np.linalg.LinAlgError(
            "equilibrated NS Gram condition number "
            f"{condition:.3e} exceeds {condition_limit:.3e}"
        )
    residual = float(
        np.linalg.norm(lower @ lower.T - equilibrated, ord=np.inf)
        / max(1.0, np.linalg.norm(equilibrated, ord=np.inf))
    )
    if residual > 5.0e-13:
        raise np.linalg.LinAlgError(
            f"NS Gram Cholesky residual exceeds tolerance: {residual:.3e}"
        )
    lift_factor = float(eta**twice_descendant_level)
    for array in (gram, equilibrium, equilibrated, lower):
        array.setflags(write=False)
    return FastNSEdgeFactor(
        basis=basis,
        equilibrium_scale=equilibrium,
        cholesky_lower=lower,
        lift_factor=lift_factor,
        condition_number=condition,
        factorization_residual=residual,
        normalized_gram=gram,
        equilibrated_gram=equilibrated,
    )


@lru_cache(maxsize=None)
def _b1_ns_vertex_expressions(
    left_basis: tuple[NSWord, ...],
    middle_word: NSWord,
    right_basis: tuple[NSWord, ...],
) -> sp.Matrix:
    """Return the exact ``b=1`` NS Ward matrix for offline generation."""

    h_left = (1 + _VERTEX_P_LEFT**2) / 2
    h_right = (1 + _VERTEX_P_RIGHT**2) / 2
    return sp.Matrix(
        [
            [
                ns_three_point(
                    left_word,
                    middle_word,
                    right_word,
                    h_infinity=h_left,
                    h_middle=_VERTEX_H_NS,
                    h_zero=h_right,
                    c=_B1_C_EXACT,
                    simplify=False,
                )
                for right_word in right_basis
            ]
            for left_word in left_basis
        ]
    )


@lru_cache(maxsize=None)
def _dynamic_vertex_coefficient_template(
    left_basis: tuple[NSWord, ...],
    middle_word: NSWord,
    right_basis: tuple[NSWord, ...],
) -> Callable[[float, float], object]:
    """Compile exact external-weight coefficients for an audit fallback."""

    matrix = _b1_ns_vertex_expressions(
        left_basis,
        middle_word,
        right_basis,
    )
    polynomials = [sp.Poly(sp.cancel(entry), _VERTEX_H_NS) for entry in matrix]
    degree = max(
        (int(polynomial.degree()) for polynomial in polynomials if not polynomial.is_zero),
        default=0,
    )
    coefficient_matrices = tuple(
        sp.Matrix(matrix.rows, matrix.cols, [
            polynomial.nth(power) for polynomial in polynomials
        ])
        for power in range(degree + 1)
    )
    return sp.lambdify(
        (_VERTEX_P_LEFT, _VERTEX_P_RIGHT),
        tuple(coefficient.tolist() for coefficient in coefficient_matrices),
        modules="numpy",
        cse=True,
        docstring_limit=0,
    )


@lru_cache(maxsize=32768)
def _evaluate_vertex_coefficients(
    left_basis: tuple[NSWord, ...],
    middle_word: NSWord,
    right_basis: tuple[NSWord, ...],
    left_momentum: float,
    right_momentum: float,
) -> np.ndarray:
    """Return exact power coefficients in the external NS weight."""

    p_left = _validate_real_momentum(left_momentum, "left momentum")
    p_right = _validate_real_momentum(right_momentum, "right momentum")
    left_level = twice_level(left_basis[0])
    right_level = twice_level(right_basis[0])
    component = _component_name(middle_word)
    if (
        _evaluate_base_generated_vertex_coefficients is not None
        and component is not None
        and left_level <= _BASE_GENERATED_MAX_TWICE_LEVEL
        and right_level <= _BASE_GENERATED_MAX_TWICE_LEVEL
    ):
        evaluated = _evaluate_base_generated_vertex_coefficients(
            component,
            left_level,
            right_level,
            p_left,
            p_right,
        )
    elif (
        _evaluate_sparse_generated_vertex_coefficients is not None
        and _sparse_has_vertex is not None
        and component is not None
        and _sparse_has_vertex(component, left_level, right_level)
    ):
        evaluated = _evaluate_sparse_generated_vertex_coefficients(
            component,
            left_level,
            right_level,
            p_left,
            p_right,
        )
    else:
        evaluator = _dynamic_vertex_coefficient_template(
            left_basis,
            middle_word,
            right_basis,
        )
        evaluated = evaluator(p_left, p_right)
    coefficients = np.asarray(evaluated, dtype=np.complex128)
    expected_shape = (len(left_basis), len(right_basis))
    if coefficients.ndim < 3 or coefficients.shape[1:] != expected_shape:
        coefficients = np.reshape(coefficients, (-1, *expected_shape))
    if not np.all(np.isfinite(coefficients)):
        raise ArithmeticError("the normalized NS vertex coefficients are non-finite")
    coefficients.setflags(write=False)
    return coefficients


def _evaluate_matrix_polynomial(
    coefficients: np.ndarray,
    external_weight: complex,
) -> np.ndarray:
    """Evaluate a power-major matrix polynomial by Horner's rule."""

    result = np.zeros(coefficients.shape[1:], dtype=np.complex128)
    weight = complex(external_weight)
    for coefficient in coefficients[::-1]:
        result = result * weight + coefficient
    return result


def _evaluate_vertex(
    left_basis: tuple[NSWord, ...],
    middle_word: NSWord,
    right_basis: tuple[NSWord, ...],
    left_momentum: float,
    external_weight: complex,
    right_momentum: float,
) -> np.ndarray:
    """Evaluate one normalized Ward matrix from exact polynomial data."""

    return _evaluate_matrix_polynomial(
        _evaluate_vertex_coefficients(
            left_basis,
            middle_word,
            right_basis,
            left_momentum,
            right_momentum,
        ),
        external_weight,
    )


def _whiten_vertex(
    vertex: np.ndarray,
    left_edge: FastNSEdgeFactor,
    right_edge: FastNSEdgeFactor,
) -> np.ndarray:
    value = (
        left_edge.equilibrium_scale[:, None]
        * vertex
        * right_edge.equilibrium_scale[None, :]
        * right_edge.lift_factor
    )
    value = solve_triangular(
        left_edge.cholesky_lower,
        value,
        lower=True,
        check_finite=False,
    )
    return solve_triangular(
        right_edge.cholesky_lower,
        value.T,
        lower=True,
        check_finite=False,
    ).T


def _whiten_vertex_coefficients(
    coefficients: np.ndarray,
    left_edge: FastNSEdgeFactor,
    right_edge: FastNSEdgeFactor,
) -> np.ndarray:
    """Whiten every external-weight coefficient in two batched solves."""

    count, left_dimension, right_dimension = coefficients.shape
    scaled = (
        left_edge.equilibrium_scale[None, :, None]
        * coefficients
        * right_edge.equilibrium_scale[None, None, :]
        * right_edge.lift_factor
    )
    left_rhs = scaled.transpose(1, 0, 2).reshape(left_dimension, -1)
    left_solved = solve_triangular(
        left_edge.cholesky_lower,
        left_rhs,
        lower=True,
        check_finite=False,
    ).reshape(left_dimension, count, right_dimension).transpose(1, 0, 2)
    right_rhs = left_solved.transpose(2, 0, 1).reshape(right_dimension, -1)
    whitened = solve_triangular(
        right_edge.cholesky_lower,
        right_rhs,
        lower=True,
        check_finite=False,
    ).reshape(right_dimension, count, left_dimension).transpose(1, 2, 0)
    whitened.setflags(write=False)
    return whitened


@lru_cache(maxsize=32768)
def _prewhitened_vertex_coefficients(
    left_twice_level: int,
    middle_word: NSWord,
    right_twice_level: int,
    left_momentum: float,
    right_momentum: float,
    left_lift_sign: int,
    right_lift_sign: int,
    condition_limit: float,
) -> np.ndarray:
    """Return geometry-only whitened coefficients reusable across energies."""

    left_edge = _fast_edge_factor(
        left_twice_level,
        left_momentum,
        left_lift_sign,
        condition_limit,
    )
    right_edge = _fast_edge_factor(
        right_twice_level,
        right_momentum,
        right_lift_sign,
        condition_limit,
    )
    coefficients = _evaluate_vertex_coefficients(
        left_edge.basis,
        middle_word,
        right_edge.basis,
        left_momentum,
        right_momentum,
    )
    return _whiten_vertex_coefficients(coefficients, left_edge, right_edge)


def _whitened_cyclic_trace(vertices: Sequence[np.ndarray]) -> complex:
    """Contract an already-whitened necklace without further solves."""

    if len(vertices) == 2:
        return complex(np.einsum("ij,ji->", *vertices, optimize=True))
    product_matrix = vertices[0]
    for matrix in vertices[1:]:
        product_matrix = product_matrix @ matrix
    return complex(np.trace(product_matrix))


def _two_point_polynomial_trace(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    """Contract two power-major matrix polynomials without choosing a weight."""

    result = np.zeros(left.shape[0] + right.shape[0] - 1, dtype=np.complex128)
    for left_power, left_matrix in enumerate(left):
        for right_power, right_matrix in enumerate(right):
            result[left_power + right_power] += np.einsum(
                "ij,ji->",
                left_matrix,
                right_matrix,
                optimize=True,
            )
    result.setflags(write=False)
    return result


def _cyclic_trace(
    vertices: Sequence[np.ndarray],
    edges: Sequence[FastNSEdgeFactor],
) -> complex:
    whitened = tuple(
        _whiten_vertex(
            matrix,
            edges[(vertex - 1) % len(edges)],
            edges[vertex],
        )
        for vertex, matrix in enumerate(vertices)
    )
    if len(whitened) == 2:
        return complex(np.einsum("ij,ji->", *whitened, optimize=True))
    product_matrix = whitened[0]
    for matrix in whitened[1:]:
        product_matrix = product_matrix @ matrix
    return complex(np.trace(product_matrix))


def generated_ns_rectangle_available(
    maximum_twice_levels: int | Sequence[int],
    external_words: Sequence[NSWord],
) -> bool:
    """Return whether every Gram and vertex kernel is statically generated."""

    words = tuple(tuple(word) for word in external_words)
    if not words or GENERATED_MAX_TWICE_LEVEL < 0:
        return False
    cutoffs = _cutoff_tuple(maximum_twice_levels, len(words))
    if any(
        not all(_has_generated_gram(level) for level in range(cutoff + 1))
        for cutoff in cutoffs
    ):
        return False
    for vertex, word in enumerate(words):
        component = _component_name(word)
        if component is None:
            return False
        previous = (vertex - 1) % len(words)
        for left_level in range(cutoffs[previous] + 1):
            for right_level in range(cutoffs[vertex] + 1):
                if not _has_generated_vertex(
                    component,
                    left_level,
                    right_level,
                ):
                    return False
    return True


def prewarm_direct_fast_b1_ns(
    maximum_twice_levels: int | Sequence[int],
    external_words: Sequence[NSWord],
) -> None:
    """Compile dynamic templates before forking when static kernels are absent."""

    words = tuple(tuple(word) for word in external_words)
    cutoffs = _cutoff_tuple(maximum_twice_levels, len(words))
    for cutoff in cutoffs:
        for level in range(cutoff + 1):
            _dynamic_gram_template(level)
    for vertex, word in enumerate(words):
        previous = (vertex - 1) % len(words)
        for left_level in range(cutoffs[previous] + 1):
            left_basis = tuple(pbw_basis(left_level))
            for right_level in range(cutoffs[vertex] + 1):
                right_basis = tuple(pbw_basis(right_level))
                _dynamic_vertex_coefficient_template(
                    left_basis,
                    word,
                    right_basis,
                )


def direct_fast_b1_ns_necklace_series(
    *,
    internal_momenta: Sequence[complex],
    external_ns_momenta: Sequence[complex],
    maximum_twice_levels: int | Sequence[int],
    edge_lift_signs: Sequence[int] | None = None,
    external_words: Sequence[NSWord] | None = None,
    form_weights: Sequence[Sequence[complex]] | None = None,
    condition_limit: float = 1.0e11,
) -> TorusNecklaceBlockSeries:
    r"""Construct the conditioned direct ``b=1`` NS necklace series."""

    internal = tuple(
        _validate_real_momentum(momentum, f"internal_momenta[{index}]")
        for index, momentum in enumerate(internal_momenta)
    )
    external = tuple(complex(momentum) for momentum in external_ns_momenta)
    if not internal or len(internal) != len(external):
        raise ValueError("internal and external momenta must have equal length")
    size = len(internal)
    cutoffs = _cutoff_tuple(maximum_twice_levels, size)
    lifts = (1,) * size if edge_lift_signs is None else tuple(edge_lift_signs)
    if len(lifts) != size:
        raise ValueError("one edge lift sign is required per edge")
    lifts = tuple(
        _validate_sign(sign, f"edge_lift_signs[{index}]")
        for index, sign in enumerate(lifts)
    )
    words = (
        ((),) * size
        if external_words is None
        else tuple(tuple(word) for word in external_words)
    )
    if len(words) != size:
        raise ValueError("one external word is required per vertex")
    weights = (
        ((1.0 + 0.0j, 1.0 + 0.0j),) * size
        if form_weights is None
        else tuple(tuple(complex(value) for value in pair) for pair in form_weights)
    )
    if len(weights) != size or any(len(pair) != 2 for pair in weights):
        raise ValueError("form_weights must contain one even/odd pair per vertex")

    coefficients: dict[tuple[int, ...], complex] = {}
    conditions: dict[tuple[int, int], float] = {}
    external_weights = tuple(ns_weight(momentum) for momentum in external)
    levels_per_edge = tuple(range(cutoff + 1) for cutoff in cutoffs)
    for levels in product(*levels_per_edge):
        edges = tuple(
            _fast_edge_factor(level, momentum, lift, float(condition_limit))
            for level, momentum, lift in zip(levels, internal, lifts)
        )
        for edge_index, (level, edge) in enumerate(zip(levels, edges)):
            conditions[(edge_index, level)] = edge.condition_number

        vertices = []
        for vertex in range(size):
            previous = (vertex - 1) % size
            coefficients_for_weight = _prewhitened_vertex_coefficients(
                levels[previous],
                words[vertex],
                levels[vertex],
                internal[previous],
                internal[vertex],
                lifts[previous],
                lifts[vertex],
                float(condition_limit),
            )
            form_parity = (
                levels[previous]
                + fermion_parity(words[vertex])
                + levels[vertex]
            ) % 2
            vertices.append(
                _evaluate_matrix_polynomial(
                    coefficients_for_weight,
                    external_weights[vertex],
                )
                * weights[vertex][form_parity]
            )
        coefficients[tuple(levels)] = _whitened_cyclic_trace(vertices)

    return TorusNecklaceBlockSeries(
        coefficients=coefficients,
        sector="NS",
        c=complex(_B1_C),
        internal_weights=tuple(ns_weight(momentum) for momentum in internal),
        external_weights=external_weights,
        external_words=words,
        edge_lift_signs=lifts,
        maximum_twice_levels=cutoffs,
        gram_condition_numbers=conditions,
    )


def direct_fast_b1_ns_two_point_polynomial_series(
    *,
    internal_momenta: Sequence[complex],
    maximum_twice_levels: int | Sequence[int],
    edge_lift_signs: Sequence[int] | None = None,
    external_words: Sequence[NSWord] | None = None,
    form_weights: Sequence[Sequence[complex]] | None = None,
    condition_limit: float = 1.0e11,
) -> ExternalWeightPolynomialBlockSeries:
    r"""Construct an exact two-point block polynomial in common ``h_ext``."""

    internal = tuple(
        _validate_real_momentum(momentum, f"internal_momenta[{index}]")
        for index, momentum in enumerate(internal_momenta)
    )
    if len(internal) != 2:
        raise ValueError("the polynomial sweep backend requires two edges")
    cutoffs = _cutoff_tuple(maximum_twice_levels, 2)
    lifts = (1, 1) if edge_lift_signs is None else tuple(edge_lift_signs)
    if len(lifts) != 2:
        raise ValueError("two edge lift signs are required")
    lifts = tuple(
        _validate_sign(sign, f"edge_lift_signs[{index}]")
        for index, sign in enumerate(lifts)
    )
    words = (
        ((), ())
        if external_words is None
        else tuple(tuple(word) for word in external_words)
    )
    if len(words) != 2:
        raise ValueError("two external words are required")
    weights = (
        ((1.0 + 0.0j, 1.0 + 0.0j),) * 2
        if form_weights is None
        else tuple(tuple(complex(value) for value in pair) for pair in form_weights)
    )
    if len(weights) != 2 or any(len(pair) != 2 for pair in weights):
        raise ValueError("form_weights must contain two even/odd pairs")

    coefficient_polynomials: dict[tuple[int, ...], np.ndarray] = {}
    conditions: dict[tuple[int, int], float] = {}
    for levels in product(*(range(cutoff + 1) for cutoff in cutoffs)):
        edges = tuple(
            _fast_edge_factor(level, momentum, lift, float(condition_limit))
            for level, momentum, lift in zip(levels, internal, lifts)
        )
        for edge_index, (level, edge) in enumerate(zip(levels, edges)):
            conditions[(edge_index, level)] = edge.condition_number

        vertices: list[np.ndarray] = []
        for vertex in range(2):
            previous = (vertex - 1) % 2
            polynomial = _prewhitened_vertex_coefficients(
                levels[previous],
                words[vertex],
                levels[vertex],
                internal[previous],
                internal[vertex],
                lifts[previous],
                lifts[vertex],
                float(condition_limit),
            )
            form_parity = (
                levels[previous]
                + fermion_parity(words[vertex])
                + levels[vertex]
            ) % 2
            vertices.append(polynomial * weights[vertex][form_parity])
        coefficient_polynomials[tuple(levels)] = _two_point_polynomial_trace(
            vertices[0],
            vertices[1],
        )

    return ExternalWeightPolynomialBlockSeries(
        coefficient_polynomials=coefficient_polynomials,
        sector="NS",
        c=complex(_B1_C),
        internal_weights=tuple(ns_weight(momentum) for momentum in internal),
        external_words=words,
        edge_lift_signs=lifts,
        maximum_twice_levels=cutoffs,
        gram_condition_numbers=conditions,
    )


__all__ = [
    "GENERATED_MAX_TWICE_LEVEL",
    "direct_fast_b1_ns_necklace_series",
    "direct_fast_b1_ns_two_point_polynomial_series",
    "generated_ns_rectangle_available",
    "prewarm_direct_fast_b1_ns",
]
