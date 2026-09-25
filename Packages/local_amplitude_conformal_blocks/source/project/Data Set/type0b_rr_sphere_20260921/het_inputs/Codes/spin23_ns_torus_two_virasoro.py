#!/usr/bin/env python3
r"""Two-Virasoro construction of the NS torus two-point components.

This is the NS analogue of :mod:`spin23_ramond_torus_recursion`.  It is kept
separate because it is a new descendant extension, not part of the scalar
primary h-recursion.  The implementation decomposes an auxiliary Majorana
module tensored with each physical NS module into two ordinary Virasoro
modules, evaluates the resulting ordinary necklace blocks, and divides out
the auxiliary block as a formal bivariate series.

The module is only promoted to the physical backend after its PP and GG
tables agree coefficient by coefficient with the independent super-Virasoro
Ward/Gram oracle.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import math
from typing import Callable

import numpy as np
import sympy as sp

from ns_algebra.ns_sca import Word
from ns_algebra.ns_sca import fermion_parity as ns_fermion_parity
from ns_algebra.ns_three_point_tensor import ns_three_point
from spin23_genus1_recursion import (
    FinitePartDiagnostics,
    dictionary_finite_part,
    ns_liouville_weight,
)
from spin23_ramond_torus_recursion import (
    CoefficientTable,
    LevelPair,
    _convolve_tables,
    _divide_series,
    _ordinary_block_coefficients,
    _subtract_tables,
)
from spin23_two_virasoro_ramond import (
    AuxiliaryFermionState,
    BranchingProduct,
    EmbeddedBranchState,
    TensorBasisState,
    auxiliary_fermion_basis,
    auxiliary_fermion_inner_product,
    auxiliary_ns_ns_ns_three_point,
    embedded_branch_state,
)


_NS_TEMPLATE_H_INFINITY, _NS_TEMPLATE_H_MIDDLE, _NS_TEMPLATE_H_ZERO = (
    sp.symbols("h_infinity h_middle h_zero")
)
_NS_TEMPLATE_C = sp.Symbol("c")


def _ns_branch_twice_grade(branch_number: Fraction) -> int:
    doubled = 2 * Fraction(branch_number)
    if doubled.denominator != 1:
        raise ValueError("NS branch numbers lie in one-half integers")
    return int(doubled) ** 2


def ns_branch_numbers(maximum_twice_grade: int) -> tuple[Fraction, ...]:
    r"""Return all :math:`n\in\frac12\mathbb Z` through a twice-grade."""

    if not isinstance(maximum_twice_grade, int) or maximum_twice_grade < 0:
        raise ValueError("maximum_twice_grade must be a nonnegative integer")
    bound = math.isqrt(maximum_twice_grade)
    return tuple(Fraction(integer, 2) for integer in range(-bound, bound + 1))


@lru_cache(maxsize=32768)
def _auxiliary_ns_edge_data(
    twice_level: int,
    lift_sign: int,
) -> tuple[tuple[AuxiliaryFermionState, ...], np.ndarray]:
    if not isinstance(twice_level, int) or twice_level < 0:
        raise ValueError("twice_level must be a nonnegative integer")
    if lift_sign not in (-1, 1):
        raise ValueError("lift_sign must be +1 or -1")
    basis = auxiliary_fermion_basis("NS", twice_level)
    if not basis:
        empty = np.zeros((0, 0), dtype=np.complex128)
        empty.setflags(write=False)
        return basis, empty
    gram = np.asarray(
        [
            [
                auxiliary_fermion_inner_product(left, right)
                for right in basis
            ]
            for left in basis
        ],
        dtype=np.complex128,
    )
    inverse = np.linalg.inv(gram)
    inverse *= lift_sign**twice_level
    inverse.setflags(write=False)
    return basis, inverse


@lru_cache(maxsize=8192)
def auxiliary_ns_necklace_coefficients(
    *,
    cut_state: AuxiliaryFermionState,
    maximum_previous_twice_level: int,
    maximum_current_twice_level: int,
    previous_lift_sign: int = 1,
    current_lift_sign: int = 1,
) -> CoefficientTable:
    """Return the sewn auxiliary-NS necklace coefficient table."""

    if cut_state.sector != "NS":
        raise ValueError("cut_state must be an NS auxiliary state")
    if cut_state.twice_modes not in ((), (1,)):
        raise NotImplementedError("only the vacuum and f_-1/2 cut states are used")
    cut_norm = auxiliary_fermion_inner_product(cut_state, cut_state)
    if cut_norm == 0:
        raise ArithmeticError("the auxiliary cut state has zero BPZ norm")
    result: CoefficientTable = {}
    for previous_level in range(maximum_previous_twice_level + 1):
        previous_basis, previous_kernel = _auxiliary_ns_edge_data(
            previous_level,
            previous_lift_sign,
        )
        for current_level in range(maximum_current_twice_level + 1):
            current_basis, current_kernel = _auxiliary_ns_edge_data(
                current_level,
                current_lift_sign,
            )
            left = np.asarray(
                [
                    [
                        auxiliary_ns_ns_ns_three_point(
                            previous,
                            cut_state,
                            current,
                        )
                        for current in current_basis
                    ]
                    for previous in previous_basis
                ],
                dtype=np.complex128,
            ).reshape((len(previous_basis), len(current_basis)))
            right = np.asarray(
                [
                    [
                        auxiliary_ns_ns_ns_three_point(
                            current,
                            cut_state,
                            previous,
                        )
                        for previous in previous_basis
                    ]
                    for current in current_basis
                ],
                dtype=np.complex128,
            ).reshape((len(current_basis), len(previous_basis)))
            result[(previous_level, current_level)] = complex(
                np.trace(
                    left
                    @ current_kernel
                    @ right
                    @ previous_kernel
                )
                / cut_norm
            )
    return result


@lru_cache(maxsize=32768)
def _ns_three_point_numeric_template(
    infinity_word: Word,
    middle_word: Word,
    zero_word: Word,
) -> Callable[..., object]:
    """Compile the exact NS Ward polynomial for one state triple."""

    expression = ns_three_point(
        infinity_word,
        middle_word,
        zero_word,
        h_infinity=_NS_TEMPLATE_H_INFINITY,
        h_middle=_NS_TEMPLATE_H_MIDDLE,
        h_zero=_NS_TEMPLATE_H_ZERO,
        c=_NS_TEMPLATE_C,
    )
    return sp.lambdify(
        (
            _NS_TEMPLATE_H_INFINITY,
            _NS_TEMPLATE_H_MIDDLE,
            _NS_TEMPLATE_H_ZERO,
            _NS_TEMPLATE_C,
        ),
        expression,
        modules="numpy",
        cse=True,
    )


def _ns_three_point_numeric(
    infinity_word: Word,
    middle_word: Word,
    zero_word: Word,
    *,
    h_infinity: complex,
    h_middle: complex,
    h_zero: complex,
    c: complex,
) -> complex:
    """Evaluate one cached exact NS Ward polynomial numerically."""

    evaluator = _ns_three_point_numeric_template(
        infinity_word,
        middle_word,
        zero_word,
    )
    return complex(evaluator(h_infinity, h_middle, h_zero, c))


@lru_cache(maxsize=32768)
def _elementary_ns_product_form(
    infinity: TensorBasisState,
    middle: TensorBasisState,
    zero: TensorBasisState,
    *,
    orientation: str,
    h_infinity: complex,
    h_middle: complex,
    h_zero: complex,
    c: complex,
    super_form_parity: int,
) -> complex:
    """Evaluate one graded product-theory NS three-point form."""

    if any(
        not isinstance(state.super_state, tuple)
        for state in (infinity, middle, zero)
    ):
        raise TypeError("all three product states must contain NS states")
    if super_form_parity not in (0, 1):
        raise ValueError("super_form_parity must be zero or one")
    mu = (
        infinity.auxiliary.parity,
        middle.auxiliary.parity,
        zero.auxiliary.parity,
    )
    nu = (
        ns_fermion_parity(infinity.super_state),
        ns_fermion_parity(middle.super_state),
        ns_fermion_parity(zero.super_state),
    )
    if sum(mu) % 2:
        return 0.0j
    if sum(nu) % 2 != super_form_parity:
        return 0.0j
    auxiliary = auxiliary_ns_ns_ns_three_point(
        infinity.auxiliary,
        middle.auxiliary,
        zero.auxiliary,
    )
    if auxiliary == 0:
        return 0.0j
    super_value = _ns_three_point_numeric(
        infinity.super_state,
        middle.super_state,
        zero.super_state,
        h_infinity=h_infinity,
        h_middle=h_middle,
        h_zero=h_zero,
        c=c,
    )
    form_parity_difference = super_form_parity
    if orientation == "left":
        exponent = (
            mu[0] * nu[1]
            + form_parity_difference * (mu[0] + mu[1])
        )
    elif orientation == "right":
        exponent = (
            mu[0] * nu[1]
            + form_parity_difference * (mu[0] + nu[1])
        )
    else:
        raise ValueError("orientation must be 'left' or 'right'")
    return complex(
        (-1) ** exponent
        * auxiliary
        * super_value
    )


def _oriented_ns_branching_coefficient(
    infinity: EmbeddedBranchState,
    middle: EmbeddedBranchState,
    zero: EmbeddedBranchState,
    *,
    orientation: str,
    super_form_parity: int,
) -> complex:
    if (infinity.sector, middle.sector, zero.sector) != ("NS", "NS", "NS"):
        raise ValueError("all three branches must be NS branches")
    total = 0.0j
    for coefficient_infinity, state_infinity in zip(
        infinity.coefficients,
        infinity.basis,
    ):
        for coefficient_middle, state_middle in zip(
            middle.coefficients,
            middle.basis,
        ):
            for coefficient_zero, state_zero in zip(
                zero.coefficients,
                zero.basis,
            ):
                total += (
                    coefficient_infinity
                    * coefficient_middle
                    * coefficient_zero
                    * _elementary_ns_product_form(
                        state_infinity,
                        state_middle,
                        state_zero,
                        orientation=orientation,
                        h_infinity=infinity.super_weight,
                        h_middle=middle.super_weight,
                        h_zero=zero.super_weight,
                        c=infinity.super_central_charge,
                        super_form_parity=super_form_parity,
                    )
                )
    return complex(total)


def ns_necklace_branching_coefficient_product(
    previous: EmbeddedBranchState,
    middle: EmbeddedBranchState,
    current: EmbeddedBranchState,
    *,
    super_form_parity: int,
) -> BranchingProduct:
    """Return the normalized two-vertex NS branch product."""

    left = _oriented_ns_branching_coefficient(
        previous,
        middle,
        current,
        orientation="left",
        super_form_parity=super_form_parity,
    )
    right = _oriented_ns_branching_coefficient(
        current,
        middle,
        previous,
        orientation="right",
        super_form_parity=super_form_parity,
    )
    norm_product = previous.norm * middle.norm * current.norm
    if norm_product == 0:
        raise ArithmeticError("a branch norm vanished")
    return BranchingProduct(
        value=left * right / norm_product,
        left=left,
        right=right,
        norm_product=norm_product,
        max_eigen_residual=max(
            previous.eigen_residual,
            middle.eigen_residual,
            current.eigen_residual,
        ),
        min_spectral_gap=min(
            previous.spectral_gap,
            middle.spectral_gap,
            current.spectral_gap,
        ),
    )


@dataclass(frozen=True)
class GenericNSNecklaceCoefficientResult:
    """Generic-b PP and GG NS necklace coefficients."""

    primary_components: dict[int, CoefficientTable]
    superdescendant_components: dict[int, CoefficientTable]
    tensor_cut_level_zero: dict[int, CoefficientTable]
    tensor_cut_level_half: dict[int, CoefficientTable]
    auxiliary_cut_level_zero: CoefficientTable
    auxiliary_cut_level_half: CoefficientTable
    b: complex
    maximum_previous_twice_level: int
    maximum_current_twice_level: int
    virasoro_recursion_order: int
    branch_term_count: int
    max_branch_eigen_residual: float
    min_branch_spectral_gap: float


@dataclass(frozen=True)
class SelfDualNSNecklaceCoefficientResult:
    """Coefficientwise b=1 PP and GG NS necklace components."""

    primary_components: dict[int, CoefficientTable]
    superdescendant_components: dict[int, CoefficientTable]
    primary_diagnostics: dict[int, dict[LevelPair, FinitePartDiagnostics]]
    superdescendant_diagnostics: dict[
        int,
        dict[LevelPair, FinitePartDiagnostics],
    ]
    maximum_previous_twice_level: int
    maximum_current_twice_level: int
    virasoro_recursion_order: int
    radius: float
    check_radius: float
    samples: int
    generic_evaluations: int
    max_branch_eigen_residual: float
    min_branch_spectral_gap: float


def generic_ns_torus_two_point_necklace_coefficients(
    *,
    b: complex,
    previous_internal_momentum: complex,
    current_internal_momentum: complex,
    external_ns_momentum: complex,
    previous_lift_sign: int = 1,
    current_lift_sign: int = 1,
    maximum_previous_twice_level: int,
    maximum_current_twice_level: int,
    virasoro_recursion_order: int | None = None,
) -> GenericNSNecklaceCoefficientResult:
    """Return generic-b PP and GG tables from two Virasoro recursions."""

    b = complex(b)
    if b == 0 or b * b == 1:
        raise ValueError("assemble at generic b before taking the finite part")
    for value, name in (
        (maximum_previous_twice_level, "maximum_previous_twice_level"),
        (maximum_current_twice_level, "maximum_current_twice_level"),
    ):
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    for value, name in (
        (previous_lift_sign, "previous_lift_sign"),
        (current_lift_sign, "current_lift_sign"),
    ):
        if value not in (-1, 1):
            raise ValueError(f"{name} must be +1 or -1")
    complete_order = (
        maximum_previous_twice_level // 2
        + maximum_current_twice_level // 2
    )
    recursion_order = (
        complete_order
        if virasoro_recursion_order is None
        else int(virasoro_recursion_order)
    )
    if recursion_order < complete_order:
        raise ValueError(
            "virasoro_recursion_order is below the requested rectangle"
        )

    previous_momentum = complex(previous_internal_momentum)
    current_momentum = complex(current_internal_momentum)
    external_momentum = complex(external_ns_momentum)
    previous_branches = ns_branch_numbers(maximum_previous_twice_level)
    current_branches = ns_branch_numbers(maximum_current_twice_level)
    max_residual = 0.0
    min_gap = math.inf
    term_count = 0

    @lru_cache(maxsize=None)
    def branch_state(momentum: complex, branch_number: Fraction):
        return embedded_branch_state(
            b=b,
            sector="NS",
            physical_momentum=momentum,
            branch_number=branch_number,
        )

    def tensor_table(
        cut_branches: tuple[Fraction, ...],
        super_form_parity: int,
    ) -> CoefficientTable:
        nonlocal max_residual, min_gap, term_count
        result: CoefficientTable = {
            (previous_level, current_level): 0.0j
            for previous_level in range(maximum_previous_twice_level + 1)
            for current_level in range(maximum_current_twice_level + 1)
        }
        for cut_number in cut_branches:
            cut = branch_state(external_momentum, cut_number)
            for previous_number in previous_branches:
                previous_onset = _ns_branch_twice_grade(previous_number)
                residual_previous = (
                    maximum_previous_twice_level - previous_onset
                ) // 2
                for current_number in current_branches:
                    current_onset = _ns_branch_twice_grade(current_number)
                    residual_current = (
                        maximum_current_twice_level - current_onset
                    ) // 2
                    previous = branch_state(
                        previous_momentum,
                        previous_number,
                    )
                    current = branch_state(
                        current_momentum,
                        current_number,
                    )
                    if (
                        previous.parity
                        + cut.parity
                        + current.parity
                        - super_form_parity
                    ) % 2:
                        continue
                    branching = ns_necklace_branching_coefficient_product(
                        previous,
                        cut,
                        current,
                        super_form_parity=super_form_parity,
                    )
                    if branching.value == 0:
                        continue
                    first = _ordinary_block_coefficients(
                        previous.parameters.c_1,
                        cut.parameters.h_1,
                        previous.parameters.h_1,
                        current.parameters.h_1,
                        residual_previous,
                        residual_current,
                        recursion_order,
                        "necklace",
                    )
                    second = _ordinary_block_coefficients(
                        previous.parameters.c_2,
                        cut.parameters.h_2,
                        previous.parameters.h_2,
                        current.parameters.h_2,
                        residual_previous,
                        residual_current,
                        recursion_order,
                        "necklace",
                    )
                    ordinary_product = _convolve_tables(
                        first,
                        second,
                        residual_previous,
                        residual_current,
                    )
                    scale = (
                        previous_lift_sign**previous.parity
                        * current_lift_sign**current.parity
                        * branching.value
                    )
                    for (ordinary_previous, ordinary_current), value in (
                        ordinary_product.items()
                    ):
                        key = (
                            previous_onset + 2 * ordinary_previous,
                            current_onset + 2 * ordinary_current,
                        )
                        result[key] = result.get(key, 0.0j) + scale * value
                    term_count += 1
                    max_residual = max(
                        max_residual,
                        branching.max_eigen_residual,
                    )
                    min_gap = min(min_gap, branching.min_spectral_gap)
        return result

    auxiliary_zero = auxiliary_ns_necklace_coefficients(
        cut_state=AuxiliaryFermionState("NS"),
        maximum_previous_twice_level=maximum_previous_twice_level,
        maximum_current_twice_level=maximum_current_twice_level,
        previous_lift_sign=previous_lift_sign,
        current_lift_sign=current_lift_sign,
    )
    auxiliary_half = auxiliary_ns_necklace_coefficients(
        cut_state=AuxiliaryFermionState("NS", (1,)),
        maximum_previous_twice_level=maximum_previous_twice_level,
        maximum_current_twice_level=maximum_current_twice_level,
        previous_lift_sign=previous_lift_sign,
        current_lift_sign=current_lift_sign,
    )
    tensor_zero: dict[int, CoefficientTable] = {}
    tensor_half: dict[int, CoefficientTable] = {}
    primary: dict[int, CoefficientTable] = {}
    superdescendant: dict[int, CoefficientTable] = {}
    external_weight = ns_liouville_weight(external_momentum, b)
    for form_parity in (0, 1):
        tensor_zero[form_parity] = tensor_table(
            (Fraction(0),),
            form_parity,
        )
        tensor_half[form_parity] = tensor_table(
            (Fraction(-1, 2), Fraction(1, 2)),
            form_parity,
        )
        primary[form_parity] = _divide_series(
            tensor_zero[form_parity],
            auxiliary_zero,
            maximum_previous_twice_level,
            maximum_current_twice_level,
        )
        mixed = _convolve_tables(
            primary[form_parity],
            auxiliary_half,
            maximum_previous_twice_level,
            maximum_current_twice_level,
        )
        cut_half = _divide_series(
            _subtract_tables(
                tensor_half[form_parity],
                mixed,
                maximum_previous_twice_level,
                maximum_current_twice_level,
            ),
            auxiliary_zero,
            maximum_previous_twice_level,
            maximum_current_twice_level,
        )
        superdescendant[form_parity] = {
            key: (-1) ** form_parity * 2.0 * external_weight * value
            for key, value in cut_half.items()
        }
    return GenericNSNecklaceCoefficientResult(
        primary_components=primary,
        superdescendant_components=superdescendant,
        tensor_cut_level_zero=tensor_zero,
        tensor_cut_level_half=tensor_half,
        auxiliary_cut_level_zero=auxiliary_zero,
        auxiliary_cut_level_half=auxiliary_half,
        b=b,
        maximum_previous_twice_level=maximum_previous_twice_level,
        maximum_current_twice_level=maximum_current_twice_level,
        virasoro_recursion_order=recursion_order,
        branch_term_count=term_count,
        max_branch_eigen_residual=max_residual,
        min_branch_spectral_gap=min_gap,
    )


def self_dual_ns_torus_two_point_necklace_coefficients(
    *,
    previous_internal_momentum: complex,
    current_internal_momentum: complex,
    external_ns_momentum: complex,
    previous_lift_sign: int = 1,
    current_lift_sign: int = 1,
    maximum_previous_twice_level: int,
    maximum_current_twice_level: int,
    virasoro_recursion_order: int | None = None,
    radius: float = 0.04,
    check_radius: float = 0.05,
    samples: int = 24,
) -> SelfDualNSNecklaceCoefficientResult:
    r"""Take the coefficientwise constant Laurent term at :math:`b=1`."""

    for value, name in (
        (maximum_previous_twice_level, "maximum_previous_twice_level"),
        (maximum_current_twice_level, "maximum_current_twice_level"),
    ):
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    complete_order = (
        maximum_previous_twice_level // 2
        + maximum_current_twice_level // 2
    )
    recursion_order = (
        complete_order
        if virasoro_recursion_order is None
        else int(virasoro_recursion_order)
    )
    levels = tuple(
        (previous_level, current_level)
        for previous_level in range(maximum_previous_twice_level + 1)
        for current_level in range(maximum_current_twice_level + 1)
    )
    keys = tuple(
        (component, form_parity, previous_level, current_level)
        for component in ("PP", "GG")
        for form_parity in (0, 1)
        for previous_level, current_level in levels
    )
    max_residual = 0.0
    min_gap = math.inf
    generic_evaluations = 0

    @lru_cache(maxsize=None)
    def evaluate_at_b(
        b: complex,
    ) -> tuple[tuple[tuple[str, int, int, int], complex], ...]:
        nonlocal max_residual, min_gap, generic_evaluations
        result = generic_ns_torus_two_point_necklace_coefficients(
            b=b,
            previous_internal_momentum=previous_internal_momentum,
            current_internal_momentum=current_internal_momentum,
            external_ns_momentum=external_ns_momentum,
            previous_lift_sign=previous_lift_sign,
            current_lift_sign=current_lift_sign,
            maximum_previous_twice_level=maximum_previous_twice_level,
            maximum_current_twice_level=maximum_current_twice_level,
            virasoro_recursion_order=recursion_order,
        )
        generic_evaluations += 1
        max_residual = max(max_residual, result.max_branch_eigen_residual)
        min_gap = min(min_gap, result.min_branch_spectral_gap)
        values: dict[tuple[str, int, int, int], complex] = {}
        for form_parity in (0, 1):
            values.update(
                {
                    ("PP", form_parity, *level): value
                    for level, value in result.primary_components[
                        form_parity
                    ].items()
                }
            )
            values.update(
                {
                    ("GG", form_parity, *level): value
                    for level, value in result.superdescendant_components[
                        form_parity
                    ].items()
                }
            )
        return tuple(sorted(values.items()))

    def evaluator(b: complex) -> dict[tuple[str, int, int, int], complex]:
        return dict(evaluate_at_b(complex(b)))

    values, diagnostics = dictionary_finite_part(
        evaluator,
        keys=keys,
        radius=radius,
        check_radius=check_radius,
        samples=samples,
    )
    primary = {
        form_parity: {
            level: values[("PP", form_parity, *level)] for level in levels
        }
        for form_parity in (0, 1)
    }
    superdescendant = {
        form_parity: {
            level: values[("GG", form_parity, *level)] for level in levels
        }
        for form_parity in (0, 1)
    }
    return SelfDualNSNecklaceCoefficientResult(
        primary_components=primary,
        superdescendant_components=superdescendant,
        primary_diagnostics={
            form_parity: {
                level: diagnostics[("PP", form_parity, *level)]
                for level in levels
            }
            for form_parity in (0, 1)
        },
        superdescendant_diagnostics={
            form_parity: {
                level: diagnostics[("GG", form_parity, *level)]
                for level in levels
            }
            for form_parity in (0, 1)
        },
        maximum_previous_twice_level=maximum_previous_twice_level,
        maximum_current_twice_level=maximum_current_twice_level,
        virasoro_recursion_order=recursion_order,
        radius=float(radius),
        check_radius=float(check_radius),
        samples=int(samples),
        generic_evaluations=generic_evaluations,
        max_branch_eigen_residual=max_residual,
        min_branch_spectral_gap=min_gap,
    )


__all__ = [
    "GenericNSNecklaceCoefficientResult",
    "SelfDualNSNecklaceCoefficientResult",
    "auxiliary_ns_necklace_coefficients",
    "generic_ns_torus_two_point_necklace_coefficients",
    "ns_branch_numbers",
    "ns_necklace_branching_coefficient_product",
    "self_dual_ns_torus_two_point_necklace_coefficients",
]
