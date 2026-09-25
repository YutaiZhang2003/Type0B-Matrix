#!/usr/bin/env python3
r"""Direct Ramond super-Virasoro blocks for the Spin(23) code.

This module implements finite-level block definitions rather than a
conjectural scalar recursion.  For sphere insertions ordered as
``(R_1, NS_2, NS_3, R_4)`` at ``(0,z,1,infinity)``, the internal channel is
Ramond and

.. math::

   \mathcal F_{\mathrm R}(z)
   =z^{h_p-h_1-h_2}\sum_{N\geq0}z^N
     \rho_L(R_4,NS_3,A)
     (B_{p,N}^{-1})^{AB}
     \rho_R(B,NS_2,R_1).

The Ramond basis contains both ground parities at every integer level.
Consequently this is not obtained by copying the two scalar NS components:
the full two-dimensional ground representation and its Gram matrix are
retained in every contraction.

The torus one-point evaluator uses the same ordered R--NS--R tensor and the
normalized Ramond sewing operator

.. math::

   q^{L_0-h_p}\eta^F
   \left(1-\nu q^{-1/2}G_0\right).

It returns the coefficient of either ``1`` or the odd modulus ``nu`` as a
separate series.  No commuting numerical value is assigned to ``nu``.

The fundamental direct API accepts arbitrary two-by-two ground tensors in
the polynomial Ramond basis.  The HJS convenience wrappers convert the
published Ramond three-point structures, including the anti-linear
infinity-slot phase, before evaluating the same direct definition.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Literal, Mapping, Sequence

import numpy as np
import sympy as sp
from scipy.linalg import lu_factor, lu_solve

from ns_algebra.ns_sca import Word as NSWord
from ns_algebra.ns_sca import fermion_parity as ns_fermion_parity
from ns_algebra.ns_sca import twice_level as ns_twice_level
from ramond_algebra.nrr_three_point_tensor import (
    hjs_to_polynomial_ground_tensor,
    rr_three_point_from_ground_tensor,
)
from ramond_algebra.ramond_plumbing import ramond_plumbing_kernels
from ramond_algebra.ramond_sca import (
    PBWState,
    gram_matrix as ramond_gram_matrix,
    ground_state,
    twice_level as ramond_twice_level,
)


GroundBasis = Literal["polynomial", "hjs"]
RamondKernelKind = Literal["even", "odd"]
GroundTensor = tuple[tuple[sp.Expr, sp.Expr], tuple[sp.Expr, sp.Expr]]

EMPTY_NS_WORD: NSWord = ()
_HJS_PHASE = (1 + 1j) / math.sqrt(2.0)


def ramond_weight(c: complex, beta: complex) -> complex:
    r"""Return the Ramond weight :math:`h=c/24-\beta^2`."""

    return complex(c) / 24.0 - complex(beta) ** 2


def primary_ramond_external_states() -> tuple[PBWState, PBWState]:
    """Return even Ramond ground states for external legs ``(1,4)``."""

    return ground_state(0), ground_state(0)


def primary_ns_external_words() -> tuple[NSWord, NSWord]:
    """Return primary NS words for external legs ``(2,3)``."""

    return EMPTY_NS_WORD, EMPTY_NS_WORD


def _validate_cutoff(maximum_twice_level: int) -> int:
    if not isinstance(maximum_twice_level, int):
        raise TypeError("maximum_twice_level must be an integer")
    if maximum_twice_level < 0:
        raise ValueError("maximum_twice_level must be nonnegative")
    return maximum_twice_level


def _validate_sign(value: int, name: str) -> int:
    if value not in (-1, 1):
        raise ValueError(f"{name} must be +1 or -1")
    return value


def _validate_kernel_kind(kind: str) -> RamondKernelKind:
    if kind not in ("even", "odd"):
        raise ValueError("kernel_kind must be 'even' or 'odd'")
    return kind  # type: ignore[return-value]


def _validate_form_parities(values: Sequence[int]) -> tuple[int, int]:
    parities = tuple(values)
    if len(parities) != 2 or any(parity not in (0, 1) for parity in parities):
        raise ValueError("form_parities must contain two entries, each zero or one")
    return parities  # type: ignore[return-value]


def _rnsr_form_parity(
    infinity_state: PBWState,
    ns_word: NSWord,
    zero_state: PBWState,
) -> int:
    return (
        infinity_state.parity
        + ns_fermion_parity(tuple(ns_word))
        + zero_state.parity
    ) % 2


def _as_numeric(expression: sp.Expr | complex, digits: int) -> complex:
    value = complex(sp.N(sp.sympify(expression), digits))
    if not math.isfinite(value.real) or not math.isfinite(value.imag):
        raise ArithmeticError(f"non-finite Ramond block value {value!r}")
    return value


def _as_ground_tensor(
    tensor: Sequence[Sequence[sp.Expr | complex]] | sp.MatrixBase,
) -> GroundTensor:
    matrix = sp.Matrix(tensor)
    if matrix.shape != (2, 2):
        raise ValueError("a Ramond ground tensor must be two-by-two")
    return (
        (sp.sympify(matrix[0, 0]), sp.sympify(matrix[0, 1])),
        (sp.sympify(matrix[1, 0]), sp.sympify(matrix[1, 1])),
    )


def _ground_tensor_matrix(tensor: GroundTensor) -> sp.Matrix:
    return sp.Matrix(tensor)


def _validate_external_data(
    external_weights: Sequence[complex],
    ramond_states: Sequence[PBWState],
    ns_words: Sequence[NSWord],
) -> tuple[
    tuple[complex, complex, complex, complex],
    tuple[PBWState, PBWState],
    tuple[NSWord, NSWord],
]:
    weights = tuple(complex(weight) for weight in external_weights)
    states = tuple(ramond_states)
    words = tuple(tuple(word) for word in ns_words)
    if len(weights) != 4:
        raise ValueError("external_weights must contain (h1,h2,h3,h4)")
    if len(states) != 2 or any(not isinstance(state, PBWState) for state in states):
        raise ValueError("ramond_states must contain PBW states for legs (1,4)")
    if len(words) != 2:
        raise ValueError("ns_words must contain descendant words for legs (2,3)")
    return weights, states, words  # type: ignore[return-value]


@lru_cache(maxsize=None)
def _factored_ramond_gram_data(
    twice_descendant_level: int,
    c: complex,
    h_internal: complex,
    digits: int,
) -> tuple[tuple[PBWState, ...], np.ndarray, np.ndarray, np.ndarray, float]:
    basis, gram = ramond_gram_matrix(
        twice_descendant_level,
        h=sp.sympify(h_internal),
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
    return tuple(basis), matrix, lu, pivots, condition


def _sphere_three_point_vectors(
    basis: Sequence[PBWState],
    *,
    c: complex,
    h_internal: complex,
    external_weights: tuple[complex, complex, complex, complex],
    ramond_states: tuple[PBWState, PBWState],
    ns_words: tuple[NSWord, NSWord],
    left_ground_tensor: GroundTensor,
    right_ground_tensor: GroundTensor,
    form_parities: tuple[int, int],
    digits: int,
) -> tuple[np.ndarray, np.ndarray]:
    h1, h2, h3, h4 = external_weights
    state1, state4 = ramond_states
    word2, word3 = ns_words
    left_form_parity, right_form_parity = form_parities
    left: list[complex] = []
    right: list[complex] = []
    for internal_state in basis:
        left.append(
            0.0j
            if _rnsr_form_parity(state4, word3, internal_state)
            != left_form_parity
            else _as_numeric(
                rr_three_point_from_ground_tensor(
                    state4,
                    word3,
                    internal_state,
                    h_infinity=h4,
                    h_ns=h3,
                    h_zero=h_internal,
                    c=c,
                    ground_tensor=_ground_tensor_matrix(left_ground_tensor),
                ),
                digits,
            )
        )
        right.append(
            0.0j
            if _rnsr_form_parity(internal_state, word2, state1)
            != right_form_parity
            else _as_numeric(
                rr_three_point_from_ground_tensor(
                    internal_state,
                    word2,
                    state1,
                    h_infinity=h_internal,
                    h_ns=h2,
                    h_zero=h1,
                    c=c,
                    ground_tensor=_ground_tensor_matrix(right_ground_tensor),
                ),
                digits,
            )
        )
    return np.asarray(left, dtype=np.complex128), np.asarray(right, dtype=np.complex128)


@dataclass(frozen=True)
class RamondSphereBlockSeries:
    """A finite direct sphere block with one internal Ramond module."""

    coefficients: Mapping[int, complex]
    c: complex
    h_internal: complex
    external_weights: tuple[complex, complex, complex, complex]
    ramond_states: tuple[PBWState, PBWState]
    ns_words: tuple[NSWord, NSWord]
    left_ground_tensor: GroundTensor
    right_ground_tensor: GroundTensor
    form_parities: tuple[int, int]
    external_ground_basis: GroundBasis
    maximum_twice_level: int
    gram_condition_numbers: Mapping[int, float]

    def descendant_value(self, z: complex) -> complex:
        """Evaluate the descendant series without its leading OPE power."""

        z = complex(z)
        if z == 0:
            return complex(self.coefficients.get(0, 0.0j))
        log_z = cmath.log(z)
        return sum(
            coefficient * cmath.exp(0.5 * twice_level * log_z)
            for twice_level, coefficient in self.coefficients.items()
        )

    def value(self, z: complex, *, include_primary_power: bool = True) -> complex:
        """Evaluate the block on the principal branch of ``log(z)``."""

        descendant = self.descendant_value(z)
        if not include_primary_power:
            return descendant
        if z == 0:
            raise ValueError("the primary Ramond block power is singular at z=0")
        h1, h2, _, _ = self.external_weights
        state1, _ = self.ramond_states
        word2, _ = self.ns_words
        effective_h1 = h1 + 0.5 * ramond_twice_level(state1.word)
        effective_h2 = h2 + 0.5 * ns_twice_level(word2) if word2 else h2
        return cmath.exp(
            (self.h_internal - effective_h1 - effective_h2) * cmath.log(z)
        ) * descendant


def direct_ramond_sphere_block_series(
    *,
    c: complex,
    h_internal: complex,
    external_weights: Sequence[complex],
    left_ground_tensor: Sequence[Sequence[sp.Expr | complex]] | sp.MatrixBase,
    right_ground_tensor: Sequence[Sequence[sp.Expr | complex]] | sp.MatrixBase,
    maximum_twice_level: int,
    form_parities: Sequence[int] = (0, 0),
    ramond_states: Sequence[PBWState] | None = None,
    ns_words: Sequence[NSWord] | None = None,
    digits: int = 50,
    condition_limit: float = 1.0e13,
) -> RamondSphereBlockSeries:
    r"""Construct the inverse-Gram Ramond sphere block.

    ``left_ground_tensor`` and ``right_ground_tensor`` are the terminal
    values of the oriented R--NS--R forms in the polynomial ground basis.
    Rows label the infinity Ramond parity and columns label the zero Ramond
    parity.  They are independent data: BPZ sewing does not justify replacing
    one by a complex conjugate of the other.
    """

    maximum_twice_level = _validate_cutoff(maximum_twice_level)
    if maximum_twice_level % 2:
        maximum_twice_level -= 1
    states_input = (
        primary_ramond_external_states() if ramond_states is None else ramond_states
    )
    words_input = primary_ns_external_words() if ns_words is None else ns_words
    weights, states, words = _validate_external_data(
        external_weights,
        states_input,
        words_input,
    )
    left_tensor = _as_ground_tensor(left_ground_tensor)
    right_tensor = _as_ground_tensor(right_ground_tensor)
    forms = _validate_form_parities(form_parities)
    coefficients: dict[int, complex] = {}
    conditions: dict[int, float] = {}
    for twice_level in range(0, maximum_twice_level + 1, 2):
        basis, _, lu, pivots, condition = _factored_ramond_gram_data(
            twice_level,
            complex(c),
            complex(h_internal),
            int(digits),
        )
        if not math.isfinite(condition) or condition > condition_limit:
            raise np.linalg.LinAlgError(
                f"Ramond Gram matrix at level {twice_level}/2 has condition "
                f"number {condition:.3e}, above {condition_limit:.3e}"
            )
        left, right = _sphere_three_point_vectors(
            basis,
            c=complex(c),
            h_internal=complex(h_internal),
            external_weights=weights,
            ramond_states=states,
            ns_words=words,
            left_ground_tensor=left_tensor,
            right_ground_tensor=right_tensor,
            form_parities=forms,
            digits=int(digits),
        )
        coefficients[twice_level] = complex(
            left @ lu_solve((lu, pivots), right, check_finite=False)
        )
        conditions[twice_level] = condition
    return RamondSphereBlockSeries(
        coefficients=coefficients,
        c=complex(c),
        h_internal=complex(h_internal),
        external_weights=weights,
        ramond_states=states,
        ns_words=words,
        left_ground_tensor=left_tensor,
        right_ground_tensor=right_tensor,
        form_parities=forms,
        external_ground_basis="polynomial",
        maximum_twice_level=maximum_twice_level,
        gram_condition_numbers=conditions,
    )


def _hjs_external_state_factor(
    state: PBWState,
    beta: complex,
    *,
    slot: Literal["infinity", "zero"],
) -> complex:
    """Return the polynomial/HJS factor of one external Ramond state."""

    if state.ground_parity == 0:
        return 1.0 + 0.0j
    beta = complex(beta)
    if beta == 0:
        raise ValueError("an odd HJS ground state is singular at beta=0")
    if slot == "infinity":
        return 1j * _HJS_PHASE * beta
    return _HJS_PHASE * beta


def hjs_ramond_sphere_block_series(
    *,
    c: complex,
    beta_internal: complex,
    beta_one: complex,
    beta_four: complex,
    h_two: complex,
    h_three: complex,
    maximum_twice_level: int,
    form_parities: Sequence[int] = (0, 0),
    left_structure_sign: int = 1,
    right_structure_sign: int = 1,
    ramond_states: Sequence[PBWState] | None = None,
    ns_words: Sequence[NSWord] | None = None,
    digits: int = 50,
    condition_limit: float = 1.0e13,
) -> RamondSphereBlockSeries:
    r"""Construct an HJS-normalized mixed Ramond sphere block.

    The right trinion is ``(R_p,NS_2,R_1)`` and the left trinion is
    ``(R_4,NS_3,R_p)``.  ``left_structure_sign`` and
    ``right_structure_sign`` independently select the two published HJS
    R--NS--R chiral structures.  External odd ground states are converted
    back from the polynomial basis after the internal sewing contraction.
    """

    _validate_sign(left_structure_sign, "left_structure_sign")
    _validate_sign(right_structure_sign, "right_structure_sign")
    states = (
        primary_ramond_external_states()
        if ramond_states is None
        else tuple(ramond_states)
    )
    if len(states) != 2:
        raise ValueError("ramond_states must contain states for legs (1,4)")
    h_internal = ramond_weight(c, beta_internal)
    h_one = ramond_weight(c, beta_one)
    h_four = ramond_weight(c, beta_four)
    right_tensor = hjs_to_polynomial_ground_tensor(
        sp.sympify(beta_internal),
        sp.sympify(beta_one),
        structure_sign=right_structure_sign,
    )
    left_tensor = hjs_to_polynomial_ground_tensor(
        sp.sympify(beta_four),
        sp.sympify(beta_internal),
        structure_sign=left_structure_sign,
    )
    series = direct_ramond_sphere_block_series(
        c=c,
        h_internal=h_internal,
        external_weights=(h_one, h_two, h_three, h_four),
        left_ground_tensor=left_tensor,
        right_ground_tensor=right_tensor,
        maximum_twice_level=maximum_twice_level,
        form_parities=form_parities,
        ramond_states=states,
        ns_words=ns_words,
        digits=digits,
        condition_limit=condition_limit,
    )
    external_factor = _hjs_external_state_factor(
        states[0], beta_one, slot="zero"
    ) * _hjs_external_state_factor(states[1], beta_four, slot="infinity")
    return replace(
        series,
        coefficients={
            level: coefficient / external_factor
            for level, coefficient in series.coefficients.items()
        },
        external_ground_basis="hjs",
    )


@dataclass(frozen=True)
class RamondTorusOnePointBlockSeries:
    """One coefficient of a normalized Ramond torus one-point block."""

    coefficients: Mapping[int, complex]
    kernel_kind: RamondKernelKind
    lift_sign: int
    c: complex
    h_internal: complex
    h_external: complex
    external_word: NSWord
    ground_tensor: GroundTensor
    maximum_twice_level: int
    gram_condition_numbers: Mapping[int, float]

    def value(self, q: complex) -> complex:
        r"""Evaluate the stripped series with exponents stored as twice-powers.

        The even kernel uses keys ``2N`` and the odd-modulus coefficient uses
        keys ``2N-1``, corresponding respectively to :math:`q^N` and
        :math:`q^{N-1/2}`.
        """

        q = complex(q)
        if q == 0 and any(power < 0 for power in self.coefficients):
            raise ValueError("the odd Ramond ground term contains q**(-1/2)")
        if q == 0:
            return complex(self.coefficients.get(0, 0.0j))
        log_q = cmath.log(q)
        return sum(
            coefficient * cmath.exp(0.5 * twice_power * log_q)
            for twice_power, coefficient in self.coefficients.items()
        )


def direct_ramond_torus_one_point_series(
    *,
    c: complex,
    h_internal: complex,
    h_external: complex,
    ground_tensor: Sequence[Sequence[sp.Expr | complex]] | sp.MatrixBase,
    maximum_twice_level: int,
    kernel_kind: RamondKernelKind = "even",
    lift_sign: int = 1,
    external_word: NSWord = EMPTY_NS_WORD,
    digits: int = 50,
    condition_limit: float = 1.0e13,
) -> RamondTorusOnePointBlockSeries:
    """Construct a direct finite-level Ramond torus one-point series."""

    maximum_twice_level = _validate_cutoff(maximum_twice_level)
    if maximum_twice_level % 2:
        maximum_twice_level -= 1
    kind = _validate_kernel_kind(kernel_kind)
    eta = _validate_sign(lift_sign, "lift_sign")
    tensor = _as_ground_tensor(ground_tensor)
    word = tuple(external_word)
    coefficients: dict[int, complex] = {}
    conditions: dict[int, float] = {}
    for twice_level in range(0, maximum_twice_level + 1, 2):
        basis, gram_matrix, _, _, condition = _factored_ramond_gram_data(
            twice_level,
            complex(c),
            complex(h_internal),
            int(digits),
        )
        if not math.isfinite(condition) or condition > condition_limit:
            raise np.linalg.LinAlgError(
                f"Ramond Gram matrix at level {twice_level}/2 has condition "
                f"number {condition:.3e}, above {condition_limit:.3e}"
            )
        exact_kernels = ramond_plumbing_kernels(
            twice_level,
            h=sp.sympify(h_internal),
            c=sp.sympify(c),
            lift_sign=eta,
        )
        exact_kernel = (
            exact_kernels.even_kernel if kind == "even" else exact_kernels.odd_kernel
        )
        kernel = np.asarray(
            [
                [
                    _as_numeric(exact_kernel[row, column], digits)
                    for column in range(exact_kernel.cols)
                ]
                for row in range(exact_kernel.rows)
            ],
            dtype=np.complex128,
        )
        if kernel.shape != gram_matrix.shape:
            raise AssertionError("Ramond sewing kernel changed basis dimension")
        three_point = np.empty(kernel.shape, dtype=np.complex128)
        for row, infinity_state in enumerate(basis):
            for column, zero_state in enumerate(basis):
                three_point[row, column] = _as_numeric(
                    rr_three_point_from_ground_tensor(
                        infinity_state,
                        word,
                        zero_state,
                        h_infinity=h_internal,
                        h_ns=h_external,
                        h_zero=h_internal,
                        c=c,
                        ground_tensor=_ground_tensor_matrix(tensor),
                    ),
                    digits,
                )
        twice_power = twice_level if kind == "even" else twice_level - 1
        coefficients[twice_power] = complex(np.sum(kernel * three_point))
        conditions[twice_level] = condition
    return RamondTorusOnePointBlockSeries(
        coefficients=coefficients,
        kernel_kind=kind,
        lift_sign=eta,
        c=complex(c),
        h_internal=complex(h_internal),
        h_external=complex(h_external),
        external_word=word,
        ground_tensor=tensor,
        maximum_twice_level=maximum_twice_level,
        gram_condition_numbers=conditions,
    )


def hjs_ramond_torus_one_point_series(
    *,
    c: complex,
    beta_internal: complex,
    h_external: complex,
    maximum_twice_level: int,
    structure_sign: int = 1,
    kernel_kind: RamondKernelKind = "even",
    lift_sign: int = 1,
    external_word: NSWord = EMPTY_NS_WORD,
    digits: int = 50,
    condition_limit: float = 1.0e13,
) -> RamondTorusOnePointBlockSeries:
    """Construct an HJS-normalized Ramond torus one-point series."""

    _validate_sign(structure_sign, "structure_sign")
    tensor = hjs_to_polynomial_ground_tensor(
        sp.sympify(beta_internal),
        sp.sympify(beta_internal),
        structure_sign=structure_sign,
    )
    return direct_ramond_torus_one_point_series(
        c=c,
        h_internal=ramond_weight(c, beta_internal),
        h_external=h_external,
        ground_tensor=tensor,
        maximum_twice_level=maximum_twice_level,
        kernel_kind=kernel_kind,
        lift_sign=lift_sign,
        external_word=external_word,
        digits=digits,
        condition_limit=condition_limit,
    )


__all__ = [
    "EMPTY_NS_WORD",
    "GroundBasis",
    "GroundTensor",
    "RamondKernelKind",
    "RamondSphereBlockSeries",
    "RamondTorusOnePointBlockSeries",
    "direct_ramond_sphere_block_series",
    "direct_ramond_torus_one_point_series",
    "hjs_ramond_sphere_block_series",
    "hjs_ramond_torus_one_point_series",
    "primary_ns_external_words",
    "primary_ramond_external_states",
    "ramond_weight",
]
