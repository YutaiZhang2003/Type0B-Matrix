#!/usr/bin/env python3
r"""Conditioned direct :math:`b=1` Ramond necklace blocks.

This module implements the production low-level backend independently of the
generic direct oracle in :mod:`spin23_genus1_blocks`.  It makes only exact
basis changes:

* the polynomial Ramond ground doublet is normalized by ``ground_parity``;
* total-even and total-odd states are grouped into exact Gram blocks;
* each normalized Gram block is diagonally equilibrated and Cholesky
  factored; and
* necklace vertices are whitened with triangular solves, so no inverse Gram
  matrix is formed.

The finite descendant rectangle and all Ward identities are unchanged.  The
module is specialized to real nonnegative continuum momenta at ``b=1``.  The
generic direct implementation remains the audit oracle for complex momenta
and for regression tests.

Generated kernels with the coefficient API return the exact polynomial
coefficients in the external NS weight.  Older generated kernels are handled
by a unit-circle DFT at the Ward-identity degree bound and are accepted only
after two independent complex holdout checks at ``5e-11`` relative scale.
That compatibility path adds controlled complex128 roundoff but no fitted or
truncated dependence on the external weight.
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

from ns_algebra.ns_sca import Word as NSWord
from ramond_algebra.nrr_three_point_tensor import (
    rr_three_point_from_ground_tensor,
)
from ramond_algebra.ramond_sca import (
    PBWState,
    gram_matrix as ramond_gram_matrix,
    pbw_basis as ramond_pbw_basis,
)
from spin23_genus1_blocks import (
    ExternalWeightPolynomialBlockSeries,
    TorusNecklaceBlockSeries,
)
from spin23_super_liouville_data import (
    ns_weight,
    ramond_liouville_weight,
    rr_ns_chiral_structure_constant,
)


_B1_C_EXACT = sp.Rational(27, 2)
_B1_C = float(_B1_C_EXACT)
_B1_P = sp.symbols("ramond_b1_P", real=True)
_B1_H = _B1_C_EXACT / 24 + _B1_P**2 / 2
_VERTEX_P_LEFT, _VERTEX_P_RIGHT, _VERTEX_H_NS, _VERTEX_SIGN = sp.symbols(
    "ramond_b1_P_left ramond_b1_P_right ramond_b1_h_ns ramond_b1_sign",
    real=True,
)

try:
    from spin23_generated.ramond_b1_kernels import (
        GENERATED_MAX_TWICE_LEVEL as _BASE_GENERATED_MAX_TWICE_LEVEL,
        evaluate_gram as _evaluate_base_generated_gram,
        evaluate_vertex_coefficients as _evaluate_base_generated_vertex_coefficients,
        evaluate_vertex_parts as _evaluate_base_generated_vertex_parts,
    )
except ImportError:
    try:
        from spin23_generated.ramond_b1_kernels import (
            GENERATED_MAX_TWICE_LEVEL as _BASE_GENERATED_MAX_TWICE_LEVEL,
            evaluate_gram as _evaluate_base_generated_gram,
            evaluate_vertex_parts as _evaluate_base_generated_vertex_parts,
        )
    except ImportError:
        _BASE_GENERATED_MAX_TWICE_LEVEL = -1
        _evaluate_base_generated_gram = None
        _evaluate_base_generated_vertex_parts = None
    _evaluate_base_generated_vertex_coefficients = None

try:
    from spin23_generated.ramond_b1_level5_x3_kernels import (
        GENERATED_MAX_TWICE_LEVEL as _SPARSE_GENERATED_MAX_TWICE_LEVEL,
        evaluate_gram as _evaluate_sparse_generated_gram,
        evaluate_vertex_coefficients as _evaluate_sparse_generated_vertex_coefficients,
        evaluate_vertex_parts as _evaluate_sparse_generated_vertex_parts,
        has_gram as _sparse_has_gram,
        has_vertex as _sparse_has_vertex,
    )
except ImportError:
    try:
        from spin23_generated.ramond_b1_level5_x3_kernels import (
            GENERATED_MAX_TWICE_LEVEL as _SPARSE_GENERATED_MAX_TWICE_LEVEL,
            evaluate_gram as _evaluate_sparse_generated_gram,
            evaluate_vertex_parts as _evaluate_sparse_generated_vertex_parts,
            has_gram as _sparse_has_gram,
            has_vertex as _sparse_has_vertex,
        )
    except ImportError:
        _SPARSE_GENERATED_MAX_TWICE_LEVEL = -1
        _evaluate_sparse_generated_gram = None
        _evaluate_sparse_generated_vertex_parts = None
        _sparse_has_gram = None
        _sparse_has_vertex = None
    _evaluate_sparse_generated_vertex_coefficients = None


GENERATED_MAX_TWICE_LEVEL = max(
    _BASE_GENERATED_MAX_TWICE_LEVEL,
    _SPARSE_GENERATED_MAX_TWICE_LEVEL,
)


@dataclass(frozen=True)
class FastRamondEdgeFactor:
    """One conditioned sewing edge at a fixed momentum and level."""

    basis: tuple[PBWState, ...]
    equilibrium_scale: np.ndarray
    cholesky_lower: np.ndarray
    parity_lift: np.ndarray
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
        raise ValueError(
            f"{name} must be nonnegative in the direct-fast backend"
        )
    return float(value.real)


def _validate_sign(sign: int, name: str) -> int:
    if sign not in (-1, 1):
        raise ValueError(f"{name} must be +1 or -1")
    return int(sign)


def _generated_component_name(middle_word: NSWord) -> str | None:
    """Return the generated-kernel component label for PP or GG data."""

    if not middle_word:
        return "PP"
    if (
        len(middle_word) == 1
        and middle_word[0].kind == "G"
        and middle_word[0].twice_index == -1
    ):
        return "GG"
    return None


def _has_generated_gram(twice_level: int) -> bool:
    if (
        _evaluate_base_generated_gram is not None
        and twice_level <= _BASE_GENERATED_MAX_TWICE_LEVEL
    ):
        return True
    return bool(
        _sparse_has_gram is not None and _sparse_has_gram(twice_level)
    )


def _has_generated_vertex(
    component: str,
    left_twice_level: int,
    right_twice_level: int,
) -> bool:
    if (
        (
            _evaluate_base_generated_vertex_coefficients is not None
            or _evaluate_base_generated_vertex_parts is not None
        )
        and left_twice_level <= _BASE_GENERATED_MAX_TWICE_LEVEL
        and right_twice_level <= _BASE_GENERATED_MAX_TWICE_LEVEL
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


def _cutoff_tuple(
    maximum_twice_levels: int | Sequence[int],
    size: int,
) -> tuple[int, ...]:
    if isinstance(maximum_twice_levels, bool):
        raise TypeError("descendant cutoffs must be integers")
    if isinstance(maximum_twice_levels, int):
        values = (maximum_twice_levels,) * size
    else:
        values = tuple(maximum_twice_levels)
    if len(values) != size:
        raise ValueError("one descendant cutoff is required per edge")
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in values
    ):
        raise TypeError("descendant cutoffs must be integers")
    if any(value < 0 for value in values):
        raise ValueError("descendant cutoffs must be nonnegative")
    return tuple(value - value % 2 for value in values)


def _parity_ordered_basis(twice_level: int) -> tuple[PBWState, ...]:
    """Return the runtime PBW basis in generated-kernel order."""

    return tuple(
        sorted(
            ramond_pbw_basis(twice_level),
            key=lambda state: state.parity,
        )
    )


@lru_cache(maxsize=None)
def _normalized_gram_expressions(
    twice_level: int,
) -> tuple[tuple[PBWState, ...], sp.Matrix]:
    """Return the exact normalized Gram matrix used by offline generation."""

    basis, gram = ramond_gram_matrix(
        twice_level,
        h=_B1_H,
        c=_B1_C_EXACT,
    )
    ground_scale = sp.diag(
        *(
            sp.sqrt(2) / _B1_P if state.ground_parity else sp.S.One
            for state in basis
        )
    )
    normalized = (ground_scale * gram * ground_scale).applyfunc(sp.cancel)
    for entry in normalized:
        if sp.denom(entry).has(_B1_P):
            raise AssertionError(
                f"uncancelled P pole in normalized level-{twice_level / 2:g} Gram"
            )

    permutation = tuple(
        sorted(range(len(basis)), key=lambda index: basis[index].parity)
    )
    ordered_basis = tuple(basis[index] for index in permutation)
    normalized = normalized.extract(permutation, permutation)
    split = sum(state.parity == 0 for state in ordered_basis)
    if any(
        normalized[row, column] != 0
        for row in range(split)
        for column in range(split, len(ordered_basis))
    ):
        raise AssertionError("Ramond Gram matrix is not exactly parity block diagonal")
    if any(
        normalized[row, column] != 0
        for row in range(split, len(ordered_basis))
        for column in range(split)
    ):
        raise AssertionError("Ramond Gram matrix is not exactly parity block diagonal")

    return ordered_basis, normalized


@lru_cache(maxsize=None)
def _dynamic_normalized_gram_template(
    twice_level: int,
) -> tuple[tuple[PBWState, ...], Callable[[float], object]]:
    """Compile an audit fallback when no static generated kernel exists."""

    ordered_basis, normalized = _normalized_gram_expressions(twice_level)
    evaluator = sp.lambdify(
        _B1_P,
        normalized.tolist(),
        modules="numpy",
        cse=True,
        docstring_limit=0,
    )
    return ordered_basis, evaluator


def _evaluate_normalized_gram(
    twice_level: int,
    momentum: float,
) -> tuple[tuple[PBWState, ...], np.ndarray]:
    """Evaluate a static Gram kernel, with symbolic fallback for audits."""

    if (
        _evaluate_base_generated_gram is not None
        and twice_level <= _BASE_GENERATED_MAX_TWICE_LEVEL
    ):
        basis = _parity_ordered_basis(twice_level)
        gram = _evaluate_base_generated_gram(twice_level, momentum)
        return basis, np.asarray(gram, dtype=np.float64)
    if (
        _evaluate_sparse_generated_gram is not None
        and _sparse_has_gram is not None
        and _sparse_has_gram(twice_level)
    ):
        basis = _parity_ordered_basis(twice_level)
        gram = _evaluate_sparse_generated_gram(twice_level, momentum)
        return basis, np.asarray(gram, dtype=np.float64)
    basis, evaluator = _dynamic_normalized_gram_template(twice_level)
    return basis, np.asarray(evaluator(momentum), dtype=np.float64)


@lru_cache(maxsize=16384)
def _fast_edge_factor(
    twice_level: int,
    momentum: float,
    lift_sign: int,
    condition_limit: float,
) -> FastRamondEdgeFactor:
    """Evaluate, equilibrate, and factor one normalized Ramond Gram matrix."""

    p = _validate_real_momentum(momentum, "internal momentum")
    eta = _validate_sign(lift_sign, "edge lift sign")
    basis, gram = _evaluate_normalized_gram(twice_level, p)
    expected_shape = (len(basis), len(basis))
    if gram.shape != expected_shape:
        gram = np.reshape(gram, expected_shape)
    if not np.all(np.isfinite(gram)):
        raise ArithmeticError("normalized Ramond Gram matrix is non-finite")
    scale_norm = max(1.0, float(np.max(np.abs(gram))))
    symmetry_error = float(np.max(np.abs(gram - gram.T)))
    if symmetry_error > 5.0e-13 * scale_norm:
        raise ArithmeticError(
            f"normalized Ramond Gram matrix is not symmetric: {symmetry_error:.3e}"
        )

    diagonal = np.diag(gram)
    if np.any(diagonal <= 0.0):
        raise np.linalg.LinAlgError(
            "normalized Ramond Gram matrix has nonpositive diagonal"
        )
    equilibrium = 1.0 / np.sqrt(diagonal)
    equilibrated = equilibrium[:, None] * gram * equilibrium[None, :]

    split = sum(state.parity == 0 for state in basis)
    slices = (slice(0, split), slice(split, len(basis)))
    cholesky = np.zeros_like(equilibrated)
    conditions: list[float] = []
    for parity_slice in slices:
        block = equilibrated[parity_slice, parity_slice]
        if block.size == 0:
            continue
        lower = np.linalg.cholesky(block)
        cholesky[parity_slice, parity_slice] = lower
        conditions.append(float(np.linalg.cond(block)))
    condition = max(conditions, default=1.0)
    if not math.isfinite(condition) or condition > condition_limit:
        raise np.linalg.LinAlgError(
            "equilibrated Ramond Gram condition number "
            f"{condition:.3e} exceeds {condition_limit:.3e}"
        )
    reconstruction = cholesky @ cholesky.T
    factorization_residual = float(
        np.linalg.norm(reconstruction - equilibrated, ord=np.inf)
        / max(1.0, np.linalg.norm(equilibrated, ord=np.inf))
    )
    if factorization_residual > 5.0e-13:
        raise np.linalg.LinAlgError(
            "Ramond Gram Cholesky residual exceeds tolerance: "
            f"{factorization_residual:.3e}"
        )

    parity_lift = np.asarray(
        [eta**state.parity for state in basis],
        dtype=np.float64,
    )
    for array in (
        gram,
        equilibrium,
        equilibrated,
        cholesky,
        parity_lift,
    ):
        array.setflags(write=False)
    return FastRamondEdgeFactor(
        basis=basis,
        equilibrium_scale=equilibrium,
        cholesky_lower=cholesky,
        parity_lift=parity_lift,
        condition_number=condition,
        factorization_residual=factorization_residual,
        normalized_gram=gram,
        equilibrated_gram=equilibrated,
    )


@lru_cache(maxsize=None)
def _normalized_vertex_sign_expressions(
    left_basis: tuple[PBWState, ...],
    middle_word: NSWord,
    right_basis: tuple[PBWState, ...],
) -> tuple[sp.Matrix, sp.Matrix]:
    r"""Return exact normalized matrices ``(A,B)`` in :math:`V(s)=A+sB`."""

    polynomial_ground_tensor = (
        (
            sp.S.One,
            sp.Rational(1, 2) * (-1 + sp.I) * _VERTEX_P_RIGHT,
        ),
        (
            sp.Rational(1, 2)
            * _VERTEX_SIGN
            * (1 - sp.I)
            * _VERTEX_P_LEFT,
            sp.Rational(1, 2)
            * _VERTEX_SIGN
            * _VERTEX_P_LEFT
            * _VERTEX_P_RIGHT,
        ),
    )
    h_left = _B1_C_EXACT / 24 + _VERTEX_P_LEFT**2 / 2
    h_right = _B1_C_EXACT / 24 + _VERTEX_P_RIGHT**2 / 2
    sign_even_rows: list[list[sp.Expr]] = []
    sign_odd_rows: list[list[sp.Expr]] = []
    for left_state in left_basis:
        even_row: list[sp.Expr] = []
        odd_row: list[sp.Expr] = []
        left_scale = (
            sp.sqrt(2) / _VERTEX_P_LEFT
            if left_state.ground_parity
            else sp.S.One
        )
        for right_state in right_basis:
            right_scale = (
                sp.sqrt(2) / _VERTEX_P_RIGHT
                if right_state.ground_parity
                else sp.S.One
            )
            polynomial = rr_three_point_from_ground_tensor(
                left_state,
                middle_word,
                right_state,
                h_infinity=h_left,
                h_ns=_VERTEX_H_NS,
                h_zero=h_right,
                c=_B1_C_EXACT,
                ground_tensor=polynomial_ground_tensor,
                simplify=False,
            )
            normalized = sp.cancel(left_scale * polynomial * right_scale)
            if sp.denom(normalized).has(_VERTEX_P_LEFT, _VERTEX_P_RIGHT):
                raise AssertionError(
                    "normalized Ramond vertex retains a momentum pole"
                )
            sign_even = sp.expand(normalized.subs(_VERTEX_SIGN, 0))
            sign_odd = sp.expand(
                normalized.subs(_VERTEX_SIGN, 1) - sign_even
            )
            even_row.append(sign_even)
            odd_row.append(sign_odd)
        sign_even_rows.append(even_row)
        sign_odd_rows.append(odd_row)
    return sp.Matrix(sign_even_rows), sp.Matrix(sign_odd_rows)


@lru_cache(maxsize=None)
def _dynamic_normalized_vertex_coefficient_template(
    left_basis: tuple[PBWState, ...],
    middle_word: NSWord,
    right_basis: tuple[PBWState, ...],
) -> Callable[[float, float], object]:
    """Compile exact external-weight coefficients for an audit fallback."""

    sign_even, sign_odd = _normalized_vertex_sign_expressions(
        left_basis,
        middle_word,
        right_basis,
    )
    expressions = list(sign_even) + list(sign_odd)
    polynomials = [sp.Poly(sp.cancel(entry), _VERTEX_H_NS) for entry in expressions]
    degree = max(
        (int(polynomial.degree()) for polynomial in polynomials if not polynomial.is_zero),
        default=0,
    )
    row_count = len(left_basis)
    column_count = len(right_basis)
    coefficients = tuple(
        (
            sp.Matrix(
                row_count,
                column_count,
                [polynomial.nth(power) for polynomial in polynomials[: row_count * column_count]],
            ).tolist(),
            sp.Matrix(
                row_count,
                column_count,
                [polynomial.nth(power) for polynomial in polynomials[row_count * column_count :]],
            ).tolist(),
        )
        for power in range(degree + 1)
    )
    return sp.lambdify(
        (_VERTEX_P_LEFT, _VERTEX_P_RIGHT),
        coefficients,
        modules="numpy",
        cse=True,
        docstring_limit=0,
    )


def _reconstruct_generated_vertex_coefficients(
    evaluator: Callable[..., object],
    *,
    component: str,
    left_twice_level: int,
    right_twice_level: int,
    left_momentum: float,
    right_momentum: float,
    degree_bound: int,
) -> np.ndarray:
    r"""Recover a generated Ward polynomial by a unit-circle DFT.

    This compatibility path is used only by generated modules predating the
    explicit coefficient API.  The Ward identities bound the degree by the
    total descendant level (plus one for a ``G_-1/2`` insertion).  A
    roots-of-unity transform is unit-conditioned, and two off-grid holdouts
    fail closed if floating-point reconstruction exceeds the stated
    tolerance.
    """

    if degree_bound < 0:
        raise ValueError("degree_bound must be nonnegative")
    # Oversampling preserves the unit-circle conditioning while averaging
    # evaluation roundoff from the legacy generated float64 kernels.  Taking
    # only the frequencies allowed by the exact Ward-identity degree bound
    # does not change or truncate the represented polynomial.
    sample_count = 16 * (degree_bound + 1)
    roots = np.exp(
        2j * math.pi * np.arange(sample_count, dtype=np.float64)
        / sample_count
    )
    samples = np.stack(
        tuple(
            np.asarray(
                evaluator(
                    component,
                    left_twice_level,
                    right_twice_level,
                    left_momentum,
                    complex(root),
                    right_momentum,
                ),
                dtype=np.complex128,
            )
            for root in roots
        ),
        axis=0,
    )
    coefficients = (
        np.fft.fft(samples, axis=0)[: degree_bound + 1] / sample_count
    )
    for holdout in (0.37 + 0.19j, -0.23 + 0.11j):
        reconstructed = np.zeros(coefficients.shape[1:], dtype=np.complex128)
        for coefficient in coefficients[::-1]:
            reconstructed = reconstructed * holdout + coefficient
        expected = np.asarray(
            evaluator(
                component,
                left_twice_level,
                right_twice_level,
                left_momentum,
                holdout,
                right_momentum,
            ),
            dtype=np.complex128,
        )
        scale = max(1.0, float(np.max(np.abs(expected))))
        discrepancy = float(np.max(np.abs(reconstructed - expected)))
        if discrepancy > 5.0e-11 * scale:
            raise ArithmeticError(
                "generated Ramond external-weight polynomial failed its "
                f"holdout check for {component} levels "
                f"({left_twice_level},{right_twice_level}) at momenta "
                f"({left_momentum:.17g},{right_momentum:.17g}): "
                f"{discrepancy / scale:.3e}"
            )
    return coefficients


@lru_cache(maxsize=32768)
def _normalized_vertex_sign_coefficients(
    left_basis: tuple[PBWState, ...],
    middle_word: NSWord,
    right_basis: tuple[PBWState, ...],
    left_momentum: float,
    right_momentum: float,
) -> np.ndarray:
    r"""Return external-weight coefficients of :math:`V(s)=A+sB`."""

    p_left = _validate_real_momentum(left_momentum, "left momentum")
    p_right = _validate_real_momentum(right_momentum, "right momentum")
    left_level = left_basis[0].twice_descendant_level
    right_level = right_basis[0].twice_descendant_level
    component = _generated_component_name(middle_word)
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
        _evaluate_base_generated_vertex_parts is not None
        and component is not None
        and left_level <= _BASE_GENERATED_MAX_TWICE_LEVEL
        and right_level <= _BASE_GENERATED_MAX_TWICE_LEVEL
    ):
        evaluated = _reconstruct_generated_vertex_coefficients(
            _evaluate_base_generated_vertex_parts,
            component=component,
            left_twice_level=left_level,
            right_twice_level=right_level,
            left_momentum=p_left,
            right_momentum=p_right,
            degree_bound=(
                (left_level + right_level) // 2
                + (1 if middle_word else 0)
            ),
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
    elif (
        _evaluate_sparse_generated_vertex_parts is not None
        and _sparse_has_vertex is not None
        and component is not None
        and _sparse_has_vertex(component, left_level, right_level)
    ):
        evaluated = _reconstruct_generated_vertex_coefficients(
            _evaluate_sparse_generated_vertex_parts,
            component=component,
            left_twice_level=left_level,
            right_twice_level=right_level,
            left_momentum=p_left,
            right_momentum=p_right,
            degree_bound=(
                (left_level + right_level) // 2
                + (1 if middle_word else 0)
            ),
        )
    else:
        evaluator = _dynamic_normalized_vertex_coefficient_template(
            left_basis,
            middle_word,
            right_basis,
        )
        evaluated = evaluator(p_left, p_right)
    expected_shape = (len(left_basis), len(right_basis))
    coefficients = np.asarray(evaluated, dtype=np.complex128)
    if coefficients.ndim < 4 or coefficients.shape[1:] != (2, *expected_shape):
        coefficients = np.reshape(coefficients, (-1, 2, *expected_shape))
    if not np.all(np.isfinite(coefficients)):
        raise ArithmeticError("normalized Ramond vertex coefficients are non-finite")
    coefficients.setflags(write=False)
    return coefficients


def _evaluate_matrix_polynomial(
    coefficients: np.ndarray,
    external_weight: complex,
) -> np.ndarray:
    """Evaluate a power-major tensor polynomial by Horner's rule."""

    result = np.zeros(coefficients.shape[1:], dtype=np.complex128)
    weight = complex(external_weight)
    for coefficient in coefficients[::-1]:
        result = result * weight + coefficient
    return result


def _normalized_vertex_sign_parts(
    left_basis: tuple[PBWState, ...],
    middle_word: NSWord,
    right_basis: tuple[PBWState, ...],
    left_momentum: float,
    external_weight: complex,
    right_momentum: float,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Return ``(A,B)`` in the normalized vertex :math:`V(s)=A+sB`."""

    evaluated = _evaluate_matrix_polynomial(
        _normalized_vertex_sign_coefficients(
            left_basis,
            middle_word,
            right_basis,
            left_momentum,
            right_momentum,
        ),
        external_weight,
    )
    return evaluated[0], evaluated[1]


def _whiten_vertex(
    normalized_vertex: np.ndarray,
    left_edge: FastRamondEdgeFactor,
    right_edge: FastRamondEdgeFactor,
) -> np.ndarray:
    """Apply exact equilibrium scalings, the lift, and two triangular solves."""

    value = (
        left_edge.equilibrium_scale[:, None]
        * normalized_vertex
        * right_edge.equilibrium_scale[None, :]
    )
    value = value * right_edge.parity_lift[None, :]
    value = solve_triangular(
        left_edge.cholesky_lower,
        value,
        lower=True,
        check_finite=False,
    )
    value = solve_triangular(
        right_edge.cholesky_lower,
        value.T,
        lower=True,
        check_finite=False,
    ).T
    return value


def _whiten_vertex_sign_pair(
    sign_even: np.ndarray,
    sign_odd: np.ndarray,
    left_edge: FastRamondEdgeFactor,
    right_edge: FastRamondEdgeFactor,
) -> tuple[np.ndarray, np.ndarray]:
    """Whiten ``A-B`` and ``A+B`` in two batched triangular solves."""

    right_dimension = sign_even.shape[1]
    scaled = []
    for vertex in (sign_even - sign_odd, sign_even + sign_odd):
        value = (
            left_edge.equilibrium_scale[:, None]
            * vertex
            * right_edge.equilibrium_scale[None, :]
        )
        scaled.append(value * right_edge.parity_lift[None, :])
    left_batch = np.concatenate(scaled, axis=1)
    left_batch = solve_triangular(
        left_edge.cholesky_lower,
        left_batch,
        lower=True,
        check_finite=False,
    )
    left_solved = (
        left_batch[:, :right_dimension],
        left_batch[:, right_dimension:],
    )

    left_dimension = sign_even.shape[0]
    right_batch = np.concatenate(
        (left_solved[0].T, left_solved[1].T),
        axis=1,
    )
    right_batch = solve_triangular(
        right_edge.cholesky_lower,
        right_batch,
        lower=True,
        check_finite=False,
    )
    return (
        right_batch[:, :left_dimension].T,
        right_batch[:, left_dimension:].T,
    )


def _whiten_vertex_tensor(
    vertices: np.ndarray,
    left_edge: FastRamondEdgeFactor,
    right_edge: FastRamondEdgeFactor,
) -> np.ndarray:
    """Whiten a leading stack of matrices in two batched solves."""

    leading_shape = vertices.shape[:-2]
    left_dimension, right_dimension = vertices.shape[-2:]
    count = math.prod(leading_shape)
    flattened = np.reshape(vertices, (count, left_dimension, right_dimension))
    scaled = (
        left_edge.equilibrium_scale[None, :, None]
        * flattened
        * right_edge.equilibrium_scale[None, None, :]
        * right_edge.parity_lift[None, None, :]
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
    result = np.reshape(whitened, (*leading_shape, left_dimension, right_dimension))
    result.setflags(write=False)
    return result


@lru_cache(maxsize=32768)
def _prewhitened_vertex_sign_coefficients(
    left_twice_level: int,
    middle_word: NSWord,
    right_twice_level: int,
    left_momentum: float,
    right_momentum: float,
    left_lift_sign: int,
    right_lift_sign: int,
    condition_limit: float,
) -> np.ndarray:
    """Return whitened sign-pair polynomials reusable across energies."""

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
    coefficients = _normalized_vertex_sign_coefficients(
        left_edge.basis,
        middle_word,
        right_edge.basis,
        left_momentum,
        right_momentum,
    )
    sign_pairs = np.stack(
        (
            coefficients[:, 0] - coefficients[:, 1],
            coefficients[:, 0] + coefficients[:, 1],
        ),
        axis=1,
    )
    return _whiten_vertex_tensor(sign_pairs, left_edge, right_edge)


def _whitened_cyclic_trace(
    normalized_vertices: Sequence[np.ndarray],
    edges: Sequence[FastRamondEdgeFactor],
) -> complex:
    if not normalized_vertices or len(normalized_vertices) != len(edges):
        raise ValueError(
            "one vertex and one edge factor are required per necklace site"
        )
    whitened = []
    for vertex, matrix in enumerate(normalized_vertices):
        previous = (vertex - 1) % len(edges)
        whitened.append(_whiten_vertex(matrix, edges[previous], edges[vertex]))
    if len(whitened) == 2:
        return complex(
            np.einsum(
                "ij,ji->",
                whitened[0],
                whitened[1],
                optimize=True,
            )
        )
    product_matrix = whitened[0]
    for matrix in whitened[1:]:
        product_matrix = product_matrix @ matrix
    return complex(np.trace(product_matrix))


def _two_point_trace(left: np.ndarray, right: np.ndarray) -> complex:
    """Return ``trace(left @ right)`` without materializing the product."""

    return complex(np.einsum("ij,ji->", left, right, optimize=True))


def _already_whitened_cyclic_trace(vertices: Sequence[np.ndarray]) -> complex:
    """Contract prewhitened vertices without additional triangular solves."""

    if len(vertices) == 2:
        return _two_point_trace(vertices[0], vertices[1])
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


def direct_fast_b1_ramond_necklace_series(
    *,
    internal_momenta: Sequence[complex],
    external_ns_momenta: Sequence[complex],
    structure_signs: Sequence[int],
    maximum_twice_levels: int | Sequence[int],
    edge_lift_signs: Sequence[int] | None = None,
    external_words: Sequence[NSWord] | None = None,
    include_structure_constants: bool = False,
    structure_precision: int = 32,
    condition_limit: float = 1.0e11,
) -> TorusNecklaceBlockSeries:
    r"""Construct a conditioned direct long-R necklace series at ``b=1``.

    The returned finite coefficient table is mathematically identical to
    :func:`spin23_genus1_blocks.b1_ramond_liouville_necklace_series` on its
    real positive-momentum domain.  ``structure_signs`` are retained as an
    explicit branch label; sign-independent vertex work is cached and reused
    between the two allowed two-point assignments.
    """

    internal = tuple(
        _validate_real_momentum(value, f"internal_momenta[{index}]")
        for index, value in enumerate(internal_momenta)
    )
    external = tuple(complex(value) for value in external_ns_momenta)
    if not internal or len(external) != len(internal):
        raise ValueError(
            "internal and external momentum lists must have equal positive "
            "length"
        )
    size = len(internal)
    signs = tuple(
        _validate_sign(value, f"structure_signs[{index}]")
        for index, value in enumerate(structure_signs)
    )
    if len(signs) != size:
        raise ValueError("one structure sign is required per vertex")
    cutoffs = _cutoff_tuple(maximum_twice_levels, size)
    lifts = (1,) * size if edge_lift_signs is None else tuple(edge_lift_signs)
    if len(lifts) != size:
        raise ValueError("one edge lift sign is required per edge")
    lifts = tuple(
        _validate_sign(value, f"edge_lift_signs[{index}]")
        for index, value in enumerate(lifts)
    )
    words = (
        ((),) * size
        if external_words is None
        else tuple(tuple(word) for word in external_words)
    )
    if len(words) != size:
        raise ValueError("one external word is required per vertex")

    levels_per_edge = tuple(
        tuple(range(0, cutoff + 1, 2)) for cutoff in cutoffs
    )
    coefficients: dict[tuple[int, ...], complex] = {}
    conditions: dict[tuple[int, int], float] = {}
    external_weights = tuple(ns_weight(momentum) for momentum in external)
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
            sign_pair_coefficients = _prewhitened_vertex_sign_coefficients(
                levels[previous],
                words[vertex],
                levels[vertex],
                internal[previous],
                internal[vertex],
                lifts[previous],
                lifts[vertex],
                float(condition_limit),
            )
            sign_pair = _evaluate_matrix_polynomial(
                sign_pair_coefficients,
                external_weights[vertex],
            )
            vertices.append(sign_pair[0 if signs[vertex] == -1 else 1])
        coefficients[tuple(levels)] = _already_whitened_cyclic_trace(vertices)

    if include_structure_constants:
        scalar = math.prod(
            rr_ns_chiral_structure_constant(
                internal[(vertex - 1) % size],
                internal[vertex],
                external[vertex],
                structure_sign=signs[vertex],
                precision=structure_precision,
            )
            for vertex in range(size)
        )
        coefficients = {
            levels: scalar * coefficient
            for levels, coefficient in coefficients.items()
        }

    return TorusNecklaceBlockSeries(
        coefficients=coefficients,
        sector="R",
        c=_B1_C,
        internal_weights=tuple(ramond_liouville_weight(p) for p in internal),
        external_weights=external_weights,
        external_words=words,
        edge_lift_signs=lifts,
        maximum_twice_levels=cutoffs,
        gram_condition_numbers=conditions,
    )


def direct_fast_b1_ramond_two_sign_series(
    *,
    internal_momenta: Sequence[complex],
    external_ns_momenta: Sequence[complex],
    temporal_lift_sign: int,
    maximum_twice_levels: int | Sequence[int],
    edge_lift_signs: Sequence[int] | None = None,
    external_words: Sequence[NSWord] | None = None,
    condition_limit: float = 1.0e11,
) -> dict[int, TorusNecklaceBlockSeries]:
    r"""Construct both allowed two-point HJS-sign series in one level loop.

    The dictionary key is the sign at vertex zero.  The sign at vertex one
    is ``temporal_lift_sign * key``.  This uses the exact affine identity
    :math:`V(s)=A+sB`; both sign matrices are grouped as multiple right-hand
    sides in the same triangular solves.
    """

    internal = tuple(
        _validate_real_momentum(value, f"internal_momenta[{index}]")
        for index, value in enumerate(internal_momenta)
    )
    external = tuple(complex(value) for value in external_ns_momenta)
    if len(internal) != 2 or len(external) != 2:
        raise ValueError("the fused HJS evaluator requires exactly two edges")
    eta = _validate_sign(temporal_lift_sign, "temporal_lift_sign")
    cutoffs = _cutoff_tuple(maximum_twice_levels, 2)
    lifts = (eta, 1) if edge_lift_signs is None else tuple(edge_lift_signs)
    if len(lifts) != 2:
        raise ValueError("two edge lift signs are required")
    lifts = tuple(
        _validate_sign(value, f"edge_lift_signs[{index}]")
        for index, value in enumerate(lifts)
    )
    words = (
        ((), ())
        if external_words is None
        else tuple(tuple(word) for word in external_words)
    )
    if len(words) != 2:
        raise ValueError("two external words are required")

    levels_per_edge = tuple(
        tuple(range(0, cutoff + 1, 2)) for cutoff in cutoffs
    )
    coefficients = {-1: {}, 1: {}}
    conditions: dict[tuple[int, int], float] = {}
    external_weights = tuple(ns_weight(momentum) for momentum in external)
    for levels in product(*levels_per_edge):
        edges = tuple(
            _fast_edge_factor(level, momentum, lift, float(condition_limit))
            for level, momentum, lift in zip(levels, internal, lifts)
        )
        for edge_index, (level, edge) in enumerate(zip(levels, edges)):
            conditions[(edge_index, level)] = edge.condition_number

        vertex_signs: list[np.ndarray] = []
        for vertex in range(2):
            previous = (vertex - 1) % 2
            sign_pair_coefficients = _prewhitened_vertex_sign_coefficients(
                levels[previous],
                words[vertex],
                levels[vertex],
                internal[previous],
                internal[vertex],
                lifts[previous],
                lifts[vertex],
                float(condition_limit),
            )
            vertex_signs.append(
                _evaluate_matrix_polynomial(
                    sign_pair_coefficients,
                    external_weights[vertex],
                )
            )
        for first_sign in (-1, 1):
            first_index = 0 if first_sign == -1 else 1
            second_sign = eta * first_sign
            second_index = 0 if second_sign == -1 else 1
            coefficients[first_sign][tuple(levels)] = _two_point_trace(
                vertex_signs[0][first_index],
                vertex_signs[1][second_index],
            )

    shared = dict(
        sector="R",
        c=_B1_C,
        internal_weights=tuple(ramond_liouville_weight(p) for p in internal),
        external_weights=external_weights,
        external_words=words,
        edge_lift_signs=lifts,
        maximum_twice_levels=cutoffs,
        gram_condition_numbers=conditions,
    )
    return {
        first_sign: TorusNecklaceBlockSeries(
            coefficients=coefficient_table,
            **shared,
        )
        for first_sign, coefficient_table in coefficients.items()
    }


def direct_fast_b1_ramond_two_sign_polynomial_series(
    *,
    internal_momenta: Sequence[complex],
    temporal_lift_sign: int,
    maximum_twice_levels: int | Sequence[int],
    edge_lift_signs: Sequence[int] | None = None,
    external_words: Sequence[NSWord] | None = None,
    condition_limit: float = 1.0e11,
) -> dict[int, ExternalWeightPolynomialBlockSeries]:
    r"""Construct both exact two-point HJS-sign polynomials in ``h_ext``."""

    internal = tuple(
        _validate_real_momentum(value, f"internal_momenta[{index}]")
        for index, value in enumerate(internal_momenta)
    )
    if len(internal) != 2:
        raise ValueError("the polynomial sweep backend requires two edges")
    eta = _validate_sign(temporal_lift_sign, "temporal_lift_sign")
    cutoffs = _cutoff_tuple(maximum_twice_levels, 2)
    lifts = (eta, 1) if edge_lift_signs is None else tuple(edge_lift_signs)
    if len(lifts) != 2:
        raise ValueError("two edge lift signs are required")
    lifts = tuple(
        _validate_sign(value, f"edge_lift_signs[{index}]")
        for index, value in enumerate(lifts)
    )
    words = (
        ((), ())
        if external_words is None
        else tuple(tuple(word) for word in external_words)
    )
    if len(words) != 2:
        raise ValueError("two external words are required")

    coefficient_polynomials: dict[int, dict[tuple[int, ...], np.ndarray]] = {
        -1: {},
        1: {},
    }
    conditions: dict[tuple[int, int], float] = {}
    levels_per_edge = tuple(
        tuple(range(0, cutoff + 1, 2)) for cutoff in cutoffs
    )
    for levels in product(*levels_per_edge):
        edges = tuple(
            _fast_edge_factor(level, momentum, lift, float(condition_limit))
            for level, momentum, lift in zip(levels, internal, lifts)
        )
        for edge_index, (level, edge) in enumerate(zip(levels, edges)):
            conditions[(edge_index, level)] = edge.condition_number

        vertex_signs: list[np.ndarray] = []
        for vertex in range(2):
            previous = (vertex - 1) % 2
            vertex_signs.append(
                _prewhitened_vertex_sign_coefficients(
                    levels[previous],
                    words[vertex],
                    levels[vertex],
                    internal[previous],
                    internal[vertex],
                    lifts[previous],
                    lifts[vertex],
                    float(condition_limit),
                )
            )
        for first_sign in (-1, 1):
            first_index = 0 if first_sign == -1 else 1
            second_sign = eta * first_sign
            second_index = 0 if second_sign == -1 else 1
            coefficient_polynomials[first_sign][tuple(levels)] = (
                _two_point_polynomial_trace(
                    vertex_signs[0][:, first_index],
                    vertex_signs[1][:, second_index],
                )
            )

    shared = dict(
        sector="R",
        c=complex(_B1_C),
        internal_weights=tuple(
            ramond_liouville_weight(momentum) for momentum in internal
        ),
        external_words=words,
        edge_lift_signs=lifts,
        maximum_twice_levels=cutoffs,
        gram_condition_numbers=conditions,
    )
    return {
        first_sign: ExternalWeightPolynomialBlockSeries(
            coefficient_polynomials=polynomials,
            **shared,
        )
        for first_sign, polynomials in coefficient_polynomials.items()
    }


def generated_direct_rectangle_available(
    maximum_twice_levels: int | Sequence[int],
    external_words: Sequence[NSWord],
) -> bool:
    """Return whether one rectangle has no runtime-symbolic dependencies."""

    words = tuple(tuple(word) for word in external_words)
    cutoffs = _cutoff_tuple(maximum_twice_levels, len(words))
    levels_by_edge = tuple(
        tuple(range(0, cutoff + 1, 2)) for cutoff in cutoffs
    )
    if any(
        not _has_generated_gram(level)
        for levels in levels_by_edge
        for level in levels
    ):
        return False
    for vertex, word in enumerate(words):
        component = _generated_component_name(word)
        if component is None:
            return False
        previous = (vertex - 1) % len(words)
        if any(
            not _has_generated_vertex(component, left_level, right_level)
            for left_level in levels_by_edge[previous]
            for right_level in levels_by_edge[vertex]
        ):
            return False
    return True


def prewarm_direct_fast_b1_ramond(
    maximum_twice_levels: int | Sequence[int],
    external_words: Sequence[NSWord],
) -> None:
    """Load static kernels or compile audit fallbacks before a POSIX fork."""

    words = tuple(tuple(word) for word in external_words)
    cutoffs = _cutoff_tuple(maximum_twice_levels, len(words))
    bases_by_edge = []
    for cutoff in cutoffs:
        bases_by_edge.append(
            {
                level: _evaluate_normalized_gram(level, 0.5)[0]
                for level in range(0, cutoff + 1, 2)
            }
        )
    for vertex, word in enumerate(words):
        previous = (vertex - 1) % len(words)
        for left_basis in bases_by_edge[previous].values():
            for right_basis in bases_by_edge[vertex].values():
                component = _generated_component_name(word)
                left_level = left_basis[0].twice_descendant_level
                right_level = right_basis[0].twice_descendant_level
                if not (
                    component is not None
                    and _has_generated_vertex(
                        component,
                        left_level,
                        right_level,
                    )
                ):
                    _dynamic_normalized_vertex_coefficient_template(
                        left_basis,
                        word,
                        right_basis,
                    )


def q_weighted_series_discrepancy(
    left: TorusNecklaceBlockSeries,
    right: TorusNecklaceBlockSeries,
    plumbing_parameters: Sequence[complex],
) -> float:
    r"""Return the absolute :math:`q`-weighted coefficient discrepancy."""

    q_values = tuple(complex(value) for value in plumbing_parameters)
    if len(q_values) != len(left.internal_weights) or len(q_values) != len(
        right.internal_weights
    ):
        raise ValueError("one plumbing parameter is required per necklace edge")
    keys = set(left.coefficients) | set(right.coefficients)
    return float(
        sum(
            math.prod(
                abs(q) ** (level / 2.0)
                for q, level in zip(q_values, levels)
            )
            * abs(
                left.coefficients.get(levels, 0.0j)
                - right.coefficients.get(levels, 0.0j)
            )
            for levels in keys
        )
    )


__all__ = [
    "FastRamondEdgeFactor",
    "direct_fast_b1_ramond_necklace_series",
    "direct_fast_b1_ramond_two_sign_series",
    "direct_fast_b1_ramond_two_sign_polynomial_series",
    "generated_direct_rectangle_available",
    "prewarm_direct_fast_b1_ramond",
    "q_weighted_series_discrepancy",
]
