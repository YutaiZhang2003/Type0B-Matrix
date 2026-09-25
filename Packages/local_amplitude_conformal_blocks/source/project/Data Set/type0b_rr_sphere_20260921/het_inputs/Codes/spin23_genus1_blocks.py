#!/usr/bin/env python3
r"""Direct finite-level genus-one N=1 super-Virasoro necklace blocks.

For ``n`` external NS insertions, label the internal necklace edges by
``e=0,...,n-1`` and let vertex ``v`` join edge ``v-1`` at infinity to edge
``v`` at zero.  At fixed descendant levels the stripped chiral block is

.. math::

   \mathcal F_{\boldsymbol N}
   =\operatorname{Tr}\!\left(
      T_0 K_0 T_1 K_1\cdots T_{n-1}K_{n-1}
     \right),

where :math:`T_v` is the ordered descendant three-point tensor and
:math:`K_e` is the inverse Gram matrix, including the chosen spin lift.
The full plumbing term is

.. math::

   \prod_e q_e^{h_e-c/24+N_e}\,
   \mathcal F_{\boldsymbol N}.

Both NS and long-R handle sectors are implemented.  The calculation is
exact at each symbolic descendant level before conversion to floating point;
it is a low-level oracle, not a replacement for the faster elliptic
recursions needed in production.  In the Ramond sector the complete
two-dimensional ground fiber is retained on every edge.
"""

from __future__ import annotations

import cmath
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
import math
from typing import Callable, Literal, Mapping, Sequence

import numpy as np
import sympy as sp

from ns_algebra.ns_sca import (
    Word as NSWord,
    fermion_parity as ns_fermion_parity,
    gram_matrix as ns_gram_matrix,
)
from ns_algebra.ns_three_point_tensor import ns_three_point
from ramond_algebra.nrr_three_point_tensor import (
    hjs_to_polynomial_ground_tensor,
    rr_three_point_from_ground_tensor,
)
from ramond_algebra.ramond_sca import (
    PBWState,
    gram_matrix as ramond_gram_matrix,
)
from spin23_super_liouville_data import (
    ns_weight,
    ramond_liouville_weight,
    rr_ns_chiral_structure_constant,
)


Sector = Literal["NS", "R"]
VertexBackend = Literal["direct", "template"]
GroundTensor = tuple[tuple[sp.Expr, sp.Expr], tuple[sp.Expr, sp.Expr]]

(
    _NS_TEMPLATE_H_INFINITY,
    _NS_TEMPLATE_H_MIDDLE,
    _NS_TEMPLATE_H_ZERO,
    _NS_TEMPLATE_C,
) = sp.symbols("ns_h_infinity ns_h_middle ns_h_zero ns_c")

_R_TEMPLATE_H_INFINITY, _R_TEMPLATE_H_NS, _R_TEMPLATE_H_ZERO, _R_TEMPLATE_C = (
    sp.symbols("h_infinity h_ns h_zero c")
)
_R_TEMPLATE_GROUND = sp.symbols("t_00 t_01 t_10 t_11")
_R_TEMPLATE_GROUND_MATRIX = (
    (_R_TEMPLATE_GROUND[0], _R_TEMPLATE_GROUND[1]),
    (_R_TEMPLATE_GROUND[2], _R_TEMPLATE_GROUND[3]),
)


def _as_numeric(expression: sp.Expr | complex, digits: int) -> complex:
    value = complex(sp.N(sp.sympify(expression), digits))
    if not math.isfinite(value.real) or not math.isfinite(value.imag):
        raise ArithmeticError(f"non-finite block entry {value!r}")
    return value


def _validate_sign(value: int, name: str) -> int:
    if value not in (-1, 1):
        raise ValueError(f"{name} must be +1 or -1")
    return value


def _validate_vertex_backend(value: str) -> VertexBackend:
    if value not in ("direct", "template"):
        raise ValueError("vertex_backend must be 'direct' or 'template'")
    return value


def _validate_parallel_inputs(
    internal_weights: Sequence[complex],
    external_weights: Sequence[complex],
    maximum_twice_levels: int | Sequence[int],
    edge_lift_signs: Sequence[int] | None,
    external_words: Sequence[NSWord] | None,
) -> tuple[
    tuple[complex, ...],
    tuple[complex, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[NSWord, ...],
]:
    internal = tuple(complex(value) for value in internal_weights)
    external = tuple(complex(value) for value in external_weights)
    if not internal or len(internal) != len(external):
        raise ValueError(
            "internal_weights and external_weights must have the same "
            "positive length"
        )
    n_vertices = len(internal)
    if isinstance(maximum_twice_levels, int):
        cutoffs = (maximum_twice_levels,) * n_vertices
    else:
        cutoffs = tuple(maximum_twice_levels)
    if len(cutoffs) != n_vertices:
        raise ValueError("maximum_twice_levels must have one entry per edge")
    if any(not isinstance(value, int) for value in cutoffs):
        raise TypeError("every descendant cutoff must be an integer")
    if any(value < 0 for value in cutoffs):
        raise ValueError("descendant cutoffs must be nonnegative")

    lifts = (
        (1,) * n_vertices
        if edge_lift_signs is None
        else tuple(edge_lift_signs)
    )
    if len(lifts) != n_vertices:
        raise ValueError("edge_lift_signs must have one entry per edge")
    lifts = tuple(_validate_sign(value, "edge lift sign") for value in lifts)

    words = (
        ((),) * n_vertices
        if external_words is None
        else tuple(tuple(word) for word in external_words)
    )
    if len(words) != n_vertices:
        raise ValueError("external_words must have one entry per vertex")
    return internal, external, cutoffs, lifts, words


def _validate_form_weights(
    form_weights: Sequence[Sequence[complex]] | None,
    n_vertices: int,
) -> tuple[tuple[complex, complex], ...]:
    if form_weights is None:
        return ((1.0 + 0.0j, 1.0 + 0.0j),) * n_vertices
    result = tuple(tuple(complex(value) for value in pair) for pair in form_weights)
    if len(result) != n_vertices or any(len(pair) != 2 for pair in result):
        raise ValueError("form_weights must contain one (even,odd) pair per vertex")
    return result  # type: ignore[return-value]


def _as_ground_tensors(
    ground_tensors: Sequence[Sequence[Sequence[sp.Expr | complex]] | sp.MatrixBase],
    n_vertices: int,
) -> tuple[GroundTensor, ...]:
    if len(ground_tensors) != n_vertices:
        raise ValueError("ground_tensors must contain one matrix per vertex")
    result: list[GroundTensor] = []
    for tensor in ground_tensors:
        matrix = sp.Matrix(tensor)
        if matrix.shape != (2, 2):
            raise ValueError("every Ramond ground tensor must be two-by-two")
        result.append(
            (
                (sp.sympify(matrix[0, 0]), sp.sympify(matrix[0, 1])),
                (sp.sympify(matrix[1, 0]), sp.sympify(matrix[1, 1])),
            )
        )
    return tuple(result)


@lru_cache(maxsize=None)
def _ns_edge_data(
    twice_level: int,
    h: complex,
    c: complex,
    lift_sign: int,
    digits: int,
) -> tuple[tuple[NSWord, ...], np.ndarray, float]:
    basis, gram = ns_gram_matrix(
        twice_level,
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
    inverse = np.linalg.inv(matrix)
    # In an NS Verma module, fermion parity equals twice-level modulo two.
    inverse *= lift_sign**twice_level
    return tuple(basis), inverse, condition


@lru_cache(maxsize=None)
def _ramond_edge_data(
    twice_level: int,
    h: complex,
    c: complex,
    lift_sign: int,
    digits: int,
) -> tuple[tuple[PBWState, ...], np.ndarray, float]:
    basis, gram = ramond_gram_matrix(
        twice_level,
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
    inverse = np.linalg.inv(matrix)
    parity_lift = np.asarray(
        [lift_sign**state.parity for state in basis],
        dtype=np.complex128,
    )
    # K_even = eta**F B^{-1}; left multiplication by the diagonal parity
    # lift scales rows of the inverse Gram matrix.
    kernel = parity_lift[:, np.newaxis] * inverse
    kernel.setflags(write=False)
    return tuple(basis), kernel, condition


def _check_condition(
    condition: float,
    *,
    edge: int,
    twice_level: int,
    condition_limit: float,
) -> None:
    if not math.isfinite(condition) or condition > condition_limit:
        raise np.linalg.LinAlgError(
            f"Gram matrix on edge {edge} at level {twice_level}/2 has "
            f"condition number {condition:.3e}, above {condition_limit:.3e}"
        )


def _cyclic_trace(vertices: Sequence[np.ndarray], kernels: Sequence[np.ndarray]) -> complex:
    if not vertices or len(vertices) != len(kernels):
        raise ValueError("cyclic contraction requires equally many vertices and edges")
    product_matrix = vertices[0] @ kernels[0]
    for vertex, kernel in zip(vertices[1:], kernels[1:]):
        product_matrix = product_matrix @ vertex @ kernel
    if product_matrix.shape[0] != product_matrix.shape[1]:
        raise AssertionError("the necklace contraction did not close")
    return complex(np.trace(product_matrix))


@lru_cache(maxsize=32768)
def _ns_vertex_data(
    infinity_basis: tuple[NSWord, ...],
    middle_word: NSWord,
    zero_basis: tuple[NSWord, ...],
    h_infinity: complex,
    h_middle: complex,
    h_zero: complex,
    c: complex,
    digits: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return reusable unweighted NS vertex entries and form parities."""

    entries = np.empty(
        (len(infinity_basis), len(zero_basis)),
        dtype=np.complex128,
    )
    parities = np.empty(entries.shape, dtype=np.int8)
    for row, infinity_word in enumerate(infinity_basis):
        for column, zero_word in enumerate(zero_basis):
            parities[row, column] = (
                ns_fermion_parity(infinity_word)
                + ns_fermion_parity(middle_word)
                + ns_fermion_parity(zero_word)
            ) % 2
            entry = ns_three_point(
                infinity_word,
                middle_word,
                zero_word,
                h_infinity=h_infinity,
                h_middle=h_middle,
                h_zero=h_zero,
                c=c,
                simplify=False,
            )
            entries[row, column] = _as_numeric(entry, digits)
    entries.setflags(write=False)
    parities.setflags(write=False)
    return entries, parities


@lru_cache(maxsize=None)
def _ns_vertex_template(
    infinity_basis: tuple[NSWord, ...],
    middle_word: NSWord,
    zero_basis: tuple[NSWord, ...],
) -> tuple[Callable[..., object], np.ndarray]:
    """Compile one weight-independent NS descendant vertex template."""

    expressions: list[list[sp.Expr]] = []
    parities = np.empty(
        (len(infinity_basis), len(zero_basis)),
        dtype=np.int8,
    )
    for row, infinity_word in enumerate(infinity_basis):
        expression_row: list[sp.Expr] = []
        for column, zero_word in enumerate(zero_basis):
            parities[row, column] = (
                ns_fermion_parity(infinity_word)
                + ns_fermion_parity(middle_word)
                + ns_fermion_parity(zero_word)
            ) % 2
            expression_row.append(
                ns_three_point(
                    infinity_word,
                    middle_word,
                    zero_word,
                    h_infinity=_NS_TEMPLATE_H_INFINITY,
                    h_middle=_NS_TEMPLATE_H_MIDDLE,
                    h_zero=_NS_TEMPLATE_H_ZERO,
                    c=_NS_TEMPLATE_C,
                    simplify=False,
                )
            )
        expressions.append(expression_row)
    parities.setflags(write=False)
    evaluator = sp.lambdify(
        (
            _NS_TEMPLATE_H_INFINITY,
            _NS_TEMPLATE_H_MIDDLE,
            _NS_TEMPLATE_H_ZERO,
            _NS_TEMPLATE_C,
        ),
        expressions,
        modules="numpy",
        # Global CSE scales poorly for the large descendant matrices that
        # first appear at level three.  The Ward expressions are already
        # exact; compiling them directly avoids a large one-time symbolic
        # optimization without changing their numerical value.
        cse=False,
        docstring_limit=0,
    )
    return evaluator, parities


@lru_cache(maxsize=32768)
def _ns_vertex_data_template(
    infinity_basis: tuple[NSWord, ...],
    middle_word: NSWord,
    zero_basis: tuple[NSWord, ...],
    h_infinity: complex,
    h_middle: complex,
    h_zero: complex,
    c: complex,
    digits: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate one precompiled NS descendant vertex template."""

    del digits  # The public block is returned as complex128 in both backends.
    evaluator, parities = _ns_vertex_template(
        infinity_basis,
        middle_word,
        zero_basis,
    )
    entries = np.asarray(
        evaluator(h_infinity, h_middle, h_zero, c),
        dtype=np.complex128,
    )
    expected_shape = (len(infinity_basis), len(zero_basis))
    if entries.shape != expected_shape:
        entries = np.reshape(entries, expected_shape)
    if not np.all(np.isfinite(entries)):
        raise ArithmeticError(
            "the compiled NS vertex returned a non-finite entry"
        )
    entries.setflags(write=False)
    return entries, parities


@lru_cache(maxsize=16384)
def _ramond_vertex_matrix_direct(
    infinity_basis: tuple[PBWState, ...],
    middle_word: NSWord,
    zero_basis: tuple[PBWState, ...],
    h_infinity: complex,
    h_ns: complex,
    h_zero: complex,
    c: complex,
    ground_tensor: GroundTensor,
    digits: int,
) -> np.ndarray:
    """Return one reusable numerical R--NS--R descendant vertex matrix."""

    matrix = np.empty(
        (len(infinity_basis), len(zero_basis)),
        dtype=np.complex128,
    )
    terminal = sp.Matrix(ground_tensor)
    for row, infinity_state in enumerate(infinity_basis):
        for column, zero_state in enumerate(zero_basis):
            entry = rr_three_point_from_ground_tensor(
                infinity_state,
                middle_word,
                zero_state,
                h_infinity=h_infinity,
                h_ns=h_ns,
                h_zero=h_zero,
                c=c,
                ground_tensor=terminal,
                simplify=False,
            )
            matrix[row, column] = _as_numeric(entry, digits)
    matrix.setflags(write=False)
    return matrix


@lru_cache(maxsize=None)
def _ramond_vertex_template(
    infinity_basis: tuple[PBWState, ...],
    middle_word: NSWord,
    zero_basis: tuple[PBWState, ...],
) -> Callable[..., object]:
    """Compile a weight-independent R--NS--R descendant tensor template."""

    expressions = []
    for infinity_state in infinity_basis:
        row = []
        for zero_state in zero_basis:
            row.append(
                rr_three_point_from_ground_tensor(
                    infinity_state,
                    middle_word,
                    zero_state,
                    h_infinity=_R_TEMPLATE_H_INFINITY,
                    h_ns=_R_TEMPLATE_H_NS,
                    h_zero=_R_TEMPLATE_H_ZERO,
                    c=_R_TEMPLATE_C,
                    ground_tensor=_R_TEMPLATE_GROUND_MATRIX,
                    simplify=False,
                )
            )
        expressions.append(row)
    return sp.lambdify(
        (
            _R_TEMPLATE_H_INFINITY,
            _R_TEMPLATE_H_NS,
            _R_TEMPLATE_H_ZERO,
            _R_TEMPLATE_C,
            *_R_TEMPLATE_GROUND,
        ),
        expressions,
        modules="numpy",
        # As in the NS template above, whole-matrix CSE costs more than the
        # subsequent numerical contractions at the levels used here.
        cse=False,
        docstring_limit=0,
    )


@lru_cache(maxsize=16384)
def _ramond_vertex_matrix_template(
    infinity_basis: tuple[PBWState, ...],
    middle_word: NSWord,
    zero_basis: tuple[PBWState, ...],
    h_infinity: complex,
    h_ns: complex,
    h_zero: complex,
    c: complex,
    ground_tensor: GroundTensor,
    digits: int,
) -> np.ndarray:
    """Evaluate one precompiled R--NS--R descendant tensor template."""

    evaluator = _ramond_vertex_template(
        infinity_basis,
        middle_word,
        zero_basis,
    )
    ground_values = tuple(
        _as_numeric(ground_tensor[row][column], digits)
        for row in range(2)
        for column in range(2)
    )
    matrix = np.asarray(
        evaluator(
            h_infinity,
            h_ns,
            h_zero,
            c,
            *ground_values,
        ),
        dtype=np.complex128,
    )
    expected_shape = (len(infinity_basis), len(zero_basis))
    if matrix.shape != expected_shape:
        matrix = np.reshape(matrix, expected_shape)
    if not np.all(np.isfinite(matrix)):
        raise ArithmeticError(
            "the compiled Ramond vertex returned a non-finite entry"
        )
    matrix.setflags(write=False)
    return matrix


@dataclass(frozen=True)
class TorusNecklaceBlockSeries:
    """A multivariate direct torus necklace block through finite levels."""

    coefficients: Mapping[tuple[int, ...], complex]
    sector: Sector
    c: complex
    internal_weights: tuple[complex, ...]
    external_weights: tuple[complex, ...]
    external_words: tuple[NSWord, ...]
    edge_lift_signs: tuple[int, ...]
    maximum_twice_levels: tuple[int, ...]
    gram_condition_numbers: Mapping[tuple[int, int], float]

    def descendant_value(
        self,
        plumbing_parameters: Sequence[complex],
        *,
        maximum_twice_levels: int | Sequence[int] | None = None,
    ) -> complex:
        """Evaluate the stripped series, optionally at a lower rectangle."""

        q_values = tuple(complex(value) for value in plumbing_parameters)
        if len(q_values) != len(self.internal_weights):
            raise ValueError("one plumbing parameter is required per edge")
        if maximum_twice_levels is None:
            cutoffs = self.maximum_twice_levels
        elif isinstance(maximum_twice_levels, int) and not isinstance(
            maximum_twice_levels,
            bool,
        ):
            cutoffs = (maximum_twice_levels,) * len(self.internal_weights)
        else:
            cutoffs = tuple(maximum_twice_levels)  # type: ignore[arg-type]
        if len(cutoffs) != len(self.internal_weights):
            raise ValueError("one evaluation cutoff is required per edge")
        if any(
            isinstance(cutoff, bool)
            or not isinstance(cutoff, int)
            or cutoff < 0
            for cutoff in cutoffs
        ):
            raise ValueError("evaluation cutoffs must be nonnegative integers")
        if any(
            cutoff > available
            for cutoff, available in zip(cutoffs, self.maximum_twice_levels)
        ):
            raise ValueError(
                "an evaluation cutoff exceeds the constructed block cutoff"
            )
        value = 0.0j
        for levels, coefficient in self.coefficients.items():
            if any(level > cutoff for level, cutoff in zip(levels, cutoffs)):
                continue
            term = coefficient
            for q_value, twice_level in zip(q_values, levels):
                if q_value == 0 and twice_level == 0:
                    continue
                if q_value == 0:
                    term = 0.0j
                    break
                term *= cmath.exp(0.5 * twice_level * cmath.log(q_value))
            value += term
        return value

    def value(
        self,
        plumbing_parameters: Sequence[complex],
        *,
        include_primary_powers: bool = True,
        maximum_twice_levels: int | Sequence[int] | None = None,
    ) -> complex:
        r"""Evaluate the block, optionally including :math:`q_e^{h_e-c/24}`."""

        q_values = tuple(complex(value) for value in plumbing_parameters)
        descendant = self.descendant_value(
            q_values,
            maximum_twice_levels=maximum_twice_levels,
        )
        if not include_primary_powers:
            return descendant
        primary = 1.0 + 0.0j
        for q_value, weight in zip(q_values, self.internal_weights):
            if q_value == 0:
                raise ValueError("primary plumbing powers are singular at q=0")
            primary *= cmath.exp((weight - self.c / 24.0) * cmath.log(q_value))
        return primary * descendant

    @property
    def leading_coefficient(self) -> complex:
        """Return the coefficient with every necklace edge at ground level."""

        return complex(self.coefficients[(0,) * len(self.internal_weights)])


@dataclass(frozen=True)
class ExternalWeightPolynomialBlockSeries:
    r"""A finite necklace block polynomial in one common external weight.

    ``coefficient_polynomials[levels][r]`` is the coefficient multiplying
    :math:`h_{\rm ext}^r` at the indicated descendant levels.  This form is
    used for exact equal-energy two-point sweeps: all inverse-Gram and Ward
    algebra is performed once, before any external energy is selected.
    """

    coefficient_polynomials: Mapping[tuple[int, ...], np.ndarray]
    sector: Sector
    c: complex
    internal_weights: tuple[complex, ...]
    external_words: tuple[NSWord, ...]
    edge_lift_signs: tuple[int, ...]
    maximum_twice_levels: tuple[int, ...]
    gram_condition_numbers: Mapping[tuple[int, int], float]

    def descendant_polynomial(
        self,
        plumbing_parameters: Sequence[complex],
        *,
        maximum_twice_levels: int | Sequence[int] | None = None,
    ) -> np.ndarray:
        """Return power-major coefficients of the stripped block value."""

        q_values = tuple(complex(value) for value in plumbing_parameters)
        if len(q_values) != len(self.internal_weights):
            raise ValueError("one plumbing parameter is required per edge")
        if maximum_twice_levels is None:
            cutoffs = self.maximum_twice_levels
        elif isinstance(maximum_twice_levels, int) and not isinstance(
            maximum_twice_levels,
            bool,
        ):
            cutoffs = (maximum_twice_levels,) * len(self.internal_weights)
        else:
            cutoffs = tuple(maximum_twice_levels)  # type: ignore[arg-type]
        if len(cutoffs) != len(self.internal_weights):
            raise ValueError("one evaluation cutoff is required per edge")
        if any(
            isinstance(cutoff, bool)
            or not isinstance(cutoff, int)
            or cutoff < 0
            for cutoff in cutoffs
        ):
            raise ValueError("evaluation cutoffs must be nonnegative integers")
        if any(
            cutoff > available
            for cutoff, available in zip(cutoffs, self.maximum_twice_levels)
        ):
            raise ValueError(
                "an evaluation cutoff exceeds the constructed block cutoff"
            )

        degree = max(
            (len(coefficients) for coefficients in self.coefficient_polynomials.values()),
            default=1,
        )
        result = np.zeros(degree, dtype=np.complex128)
        for levels, coefficients in self.coefficient_polynomials.items():
            if any(level > cutoff for level, cutoff in zip(levels, cutoffs)):
                continue
            plumbing_factor = 1.0 + 0.0j
            for q_value, twice_level in zip(q_values, levels):
                if q_value == 0 and twice_level == 0:
                    continue
                if q_value == 0:
                    plumbing_factor = 0.0j
                    break
                plumbing_factor *= cmath.exp(
                    0.5 * twice_level * cmath.log(q_value)
                )
            result[: len(coefficients)] += plumbing_factor * coefficients
        result.setflags(write=False)
        return result

    def at_external_weight(
        self,
        external_weight: complex,
    ) -> TorusNecklaceBlockSeries:
        """Evaluate the exact polynomial and return the ordinary series."""

        weight = complex(external_weight)
        coefficients = {
            levels: complex(np.polynomial.polynomial.polyval(weight, polynomial))
            for levels, polynomial in self.coefficient_polynomials.items()
        }
        return TorusNecklaceBlockSeries(
            coefficients=coefficients,
            sector=self.sector,
            c=self.c,
            internal_weights=self.internal_weights,
            external_weights=(weight,) * len(self.internal_weights),
            external_words=self.external_words,
            edge_lift_signs=self.edge_lift_signs,
            maximum_twice_levels=self.maximum_twice_levels,
            gram_condition_numbers=self.gram_condition_numbers,
        )


def direct_ns_torus_necklace_series(
    *,
    c: complex,
    internal_weights: Sequence[complex],
    external_weights: Sequence[complex],
    maximum_twice_levels: int | Sequence[int],
    edge_lift_signs: Sequence[int] | None = None,
    external_words: Sequence[NSWord] | None = None,
    form_weights: Sequence[Sequence[complex]] | None = None,
    digits: int = 50,
    condition_limit: float = 1.0e13,
    vertex_backend: VertexBackend = "direct",
) -> TorusNecklaceBlockSeries:
    r"""Construct a direct NS-handle torus necklace block.

    ``form_weights[v]`` is ``(w_even,w_odd)`` and multiplies the normalized
    even or odd NS three-point form at vertex ``v``.  Keeping it equal to
    ``(1,1)`` returns the algebraic block oracle; supplying the corresponding
    super-Liouville constants assembles a physical chiral spectral kernel.
    """

    internal, external, cutoffs, lifts, words = _validate_parallel_inputs(
        internal_weights,
        external_weights,
        maximum_twice_levels,
        edge_lift_signs,
        external_words,
    )
    vertex_backend = _validate_vertex_backend(vertex_backend)
    weights = _validate_form_weights(form_weights, len(internal))
    levels_per_edge = tuple(tuple(range(cutoff + 1)) for cutoff in cutoffs)
    coefficients: dict[tuple[int, ...], complex] = {}
    conditions: dict[tuple[int, int], float] = {}
    for levels in product(*levels_per_edge):
        bases: list[tuple[NSWord, ...]] = []
        kernels: list[np.ndarray] = []
        for edge, (level, h, lift) in enumerate(zip(levels, internal, lifts)):
            basis, kernel, condition = _ns_edge_data(
                level,
                h,
                complex(c),
                lift,
                int(digits),
            )
            _check_condition(
                condition,
                edge=edge,
                twice_level=level,
                condition_limit=condition_limit,
            )
            bases.append(basis)
            kernels.append(kernel)
            conditions[(edge, level)] = condition

        vertices: list[np.ndarray] = []
        for vertex in range(len(internal)):
            previous = (vertex - 1) % len(internal)
            vertex_builder = (
                _ns_vertex_data
                if vertex_backend == "direct"
                else _ns_vertex_data_template
            )
            unweighted, parities = vertex_builder(
                tuple(bases[previous]),
                words[vertex],
                tuple(bases[vertex]),
                internal[previous],
                external[vertex],
                internal[vertex],
                complex(c),
                int(digits),
            )
            selected_weights = np.asarray(weights[vertex], dtype=np.complex128)
            matrix = unweighted * selected_weights[parities]
            vertices.append(matrix)
        coefficients[tuple(levels)] = _cyclic_trace(vertices, kernels)

    return TorusNecklaceBlockSeries(
        coefficients=coefficients,
        sector="NS",
        c=complex(c),
        internal_weights=internal,
        external_weights=external,
        external_words=words,
        edge_lift_signs=lifts,
        maximum_twice_levels=cutoffs,
        gram_condition_numbers=conditions,
    )


def direct_ramond_torus_necklace_series(
    *,
    c: complex,
    internal_weights: Sequence[complex],
    external_weights: Sequence[complex],
    ground_tensors: Sequence[
        Sequence[Sequence[sp.Expr | complex]] | sp.MatrixBase
    ],
    maximum_twice_levels: int | Sequence[int],
    edge_lift_signs: Sequence[int] | None = None,
    external_words: Sequence[NSWord] | None = None,
    digits: int = 50,
    condition_limit: float = 1.0e13,
    vertex_backend: VertexBackend = "direct",
) -> TorusNecklaceBlockSeries:
    """Construct a direct long-R-handle torus necklace block.

    Each ground tensor is in the polynomial Ramond basis.  Its rows label
    the ground parity on edge ``v-1`` at infinity and its columns label the
    ground parity on edge ``v`` at zero.
    """

    internal, external, cutoffs, lifts, words = _validate_parallel_inputs(
        internal_weights,
        external_weights,
        maximum_twice_levels,
        edge_lift_signs,
        external_words,
    )
    vertex_backend = _validate_vertex_backend(vertex_backend)
    cutoffs = tuple(cutoff - cutoff % 2 for cutoff in cutoffs)
    tensors = _as_ground_tensors(ground_tensors, len(internal))
    levels_per_edge = tuple(
        tuple(range(0, cutoff + 1, 2)) for cutoff in cutoffs
    )
    coefficients: dict[tuple[int, ...], complex] = {}
    conditions: dict[tuple[int, int], float] = {}
    for levels in product(*levels_per_edge):
        bases: list[tuple[PBWState, ...]] = []
        kernels: list[np.ndarray] = []
        for edge, (level, h, lift) in enumerate(zip(levels, internal, lifts)):
            basis, kernel, condition = _ramond_edge_data(
                level,
                h,
                complex(c),
                lift,
                int(digits),
            )
            _check_condition(
                condition,
                edge=edge,
                twice_level=level,
                condition_limit=condition_limit,
            )
            bases.append(basis)
            kernels.append(kernel)
            conditions[(edge, level)] = condition

        vertices: list[np.ndarray] = []
        for vertex in range(len(internal)):
            previous = (vertex - 1) % len(internal)
            vertex_builder = (
                _ramond_vertex_matrix_direct
                if vertex_backend == "direct"
                else _ramond_vertex_matrix_template
            )
            matrix = vertex_builder(
                tuple(bases[previous]),
                words[vertex],
                tuple(bases[vertex]),
                internal[previous],
                external[vertex],
                internal[vertex],
                complex(c),
                tensors[vertex],
                int(digits),
            )
            vertices.append(matrix)
        coefficients[tuple(levels)] = _cyclic_trace(vertices, kernels)

    return TorusNecklaceBlockSeries(
        coefficients=coefficients,
        sector="R",
        c=complex(c),
        internal_weights=internal,
        external_weights=external,
        external_words=words,
        edge_lift_signs=lifts,
        maximum_twice_levels=cutoffs,
        gram_condition_numbers=conditions,
    )


def b1_ramond_liouville_necklace_series(
    *,
    internal_momenta: Sequence[complex],
    external_ns_momenta: Sequence[complex],
    structure_signs: Sequence[int],
    maximum_twice_levels: int | Sequence[int],
    edge_lift_signs: Sequence[int] | None = None,
    external_words: Sequence[NSWord] | None = None,
    include_structure_constants: bool = False,
    precision: int = 40,
    digits: int = 50,
    condition_limit: float = 1.0e13,
    vertex_backend: VertexBackend = "template",
) -> TorusNecklaceBlockSeries:
    r"""Construct one chiral :math:`b=1` Ramond Liouville necklace branch.

    At vertex ``v`` the ordered tensor is
    ``(R(P[v-1]), NS(Pext[v]), R(P[v]))``.  ``structure_signs[v]`` selects
    the HJS ``+`` or ``-`` chiral form.  If ``include_structure_constants``
    is true, the corresponding delta-normalized R--R--NS coefficient is
    multiplied into that vertex.  This remains one chiral branch; the
    nonchiral spectral integral and heterotic GSO sum are separate.
    """

    internal_p = tuple(complex(value) for value in internal_momenta)
    external_p = tuple(complex(value) for value in external_ns_momenta)
    signs = tuple(structure_signs)
    if not internal_p or len(internal_p) != len(external_p):
        raise ValueError("internal and external momentum lists must match")
    if len(signs) != len(internal_p):
        raise ValueError("structure_signs must have one entry per vertex")
    for sign in signs:
        _validate_sign(sign, "structure sign")

    tensors: list[sp.Matrix] = []
    for vertex, sign in enumerate(signs):
        previous = (vertex - 1) % len(internal_p)
        beta_left = 1j * internal_p[previous] / math.sqrt(2.0)
        beta_right = 1j * internal_p[vertex] / math.sqrt(2.0)
        tensor = hjs_to_polynomial_ground_tensor(
            sp.sympify(beta_left),
            sp.sympify(beta_right),
            structure_sign=sign,
        )
        if include_structure_constants:
            coefficient = rr_ns_chiral_structure_constant(
                internal_p[previous],
                internal_p[vertex],
                external_p[vertex],
                structure_sign=sign,
                precision=precision,
            )
            tensor = sp.sympify(coefficient) * tensor
        tensors.append(tensor)

    return direct_ramond_torus_necklace_series(
        c=13.5,
        internal_weights=tuple(ramond_liouville_weight(p) for p in internal_p),
        external_weights=tuple(ns_weight(p) for p in external_p),
        ground_tensors=tensors,
        maximum_twice_levels=maximum_twice_levels,
        edge_lift_signs=edge_lift_signs,
        external_words=external_words,
        digits=digits,
        condition_limit=condition_limit,
        vertex_backend=vertex_backend,
    )


__all__ = [
    "GroundTensor",
    "Sector",
    "TorusNecklaceBlockSeries",
    "VertexBackend",
    "b1_ramond_liouville_necklace_series",
    "direct_ns_torus_necklace_series",
    "direct_ramond_torus_necklace_series",
]
