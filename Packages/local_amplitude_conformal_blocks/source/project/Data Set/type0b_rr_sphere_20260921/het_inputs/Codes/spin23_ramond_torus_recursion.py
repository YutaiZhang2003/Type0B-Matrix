#!/usr/bin/env python3
r"""High-precision Ramond two-point blocks from two Virasoro recursions.

This module completes the exact strategy described in
``outside_tex/ramond_blocks_from_two_virasoro.pdf`` for the two-edge necklace
used by the genus-one SO(23) amplitude.

At generic :math:`b`, tensoring the Ramond super-Virasoro module with one
auxiliary Majorana fermion decomposes it into two commuting ordinary
Virasoro modules.  The implementation

1. constructs every locally allowed branch highest state by finite
   diagonalization of the embedded :math:`L^{(1)}_0`;
2. evaluates both oriented branching coefficients from the exact descendant
   three-point tensors, without assuming the conditional mixed finite-product
   formula;
3. evaluates the two ordinary blocks by CCY central-charge recursion;
4. sums both Ramond parity copies and all retained branch labels; and
5. divides the complete result by the auxiliary-fermion block as a formal
   cut-edge series.

For cut-edge levels zero and one half this gives, respectively, the torus
two-point blocks with external :math:`V,V` and
:math:`G_{-1/2}V,G_{-1/2}V`.  At the self-dual point the complete generic-b
answer is assembled first and only then is the constant Laurent coefficient
in :math:`t=\log b` projected.  This ordering is essential because individual
two-Virasoro branches are singular at :math:`b=1` although their sum is
finite.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import math

from spin23_genus1_recursion import (
    FinitePartDiagnostics,
    dictionary_finite_part,
    ns_liouville_weight,
)
from spin23_two_virasoro_ramond import (
    AuxiliaryFermionState,
    auxiliary_ramond_necklace_coefficients,
    auxiliary_ramond_theta_coefficients,
    embedded_branch_state,
    necklace_branching_coefficient_product,
    theta_branching_coefficient_product,
)
from spin23_virasoro_torus_recursion import (
    virasoro_torus_two_point_coefficients,
)


BranchNumber = Fraction
LevelPair = tuple[int, int]
CoefficientTable = dict[LevelPair, complex]


def _validate_sign(value: int, name: str) -> int:
    if value not in (-1, 1):
        raise ValueError(f"{name} must be +1 or -1")
    return value


def _ramond_branch_level(branch_number: BranchNumber) -> int:
    """Return the integer level ``2*n**2-1/8`` of one R branch."""

    n = Fraction(branch_number)
    k = 2 * (n - Fraction(1, 4))
    if k.denominator != 1:
        raise ValueError("Ramond branch numbers lie in 1/2 Z + 1/4")
    integer = int(k)
    numerator = integer * (integer + 1)
    if numerator % 2:
        raise AssertionError("the Ramond branch onset must be integral")
    return numerator // 2


def ramond_branch_numbers(maximum_level: int) -> tuple[BranchNumber, ...]:
    r"""Return every :math:`n\in\frac12\mathbb Z+\frac14` through a level."""

    if not isinstance(maximum_level, int) or maximum_level < 0:
        raise ValueError("maximum_level must be a nonnegative integer")
    bound = 2 * maximum_level + 3
    values = {
        Fraction(k, 2) + Fraction(1, 4)
        for k in range(-bound, bound + 1)
        if k * (k + 1) // 2 <= maximum_level
        and k * (k + 1) >= 0
    }
    return tuple(sorted(values))


@lru_cache(maxsize=131072)
def _ordinary_block_coefficients(
    c: complex,
    h_cut: complex,
    h_edge_2: complex,
    h_edge_3: complex,
    maximum_level_2: int,
    maximum_level_3: int,
    recursion_order: int,
    local_ordering: str,
) -> tuple[tuple[LevelPair, complex], ...]:
    """Return a hashable ordinary-Virasoro coefficient table."""

    result = virasoro_torus_two_point_coefficients(
        c=c,
        h_cut=h_cut,
        h_previous=h_edge_2,
        h_current=h_edge_3,
        maximum_previous_level=maximum_level_2,
        maximum_current_level=maximum_level_3,
        recursion_order=recursion_order,
        local_ordering=local_ordering,  # type: ignore[arg-type]
    )
    return tuple(sorted(result.coefficients.items()))


def _convolve_tables(
    left: dict[LevelPair, complex] | tuple[tuple[LevelPair, complex], ...],
    right: dict[LevelPair, complex] | tuple[tuple[LevelPair, complex], ...],
    maximum_level_2: int,
    maximum_level_3: int,
) -> CoefficientTable:
    """Multiply two bivariate series inside a rectangular cutoff."""

    left_items = left.items() if isinstance(left, dict) else left
    right_items = right.items() if isinstance(right, dict) else right
    result: CoefficientTable = {}
    for (left_2, left_3), left_value in left_items:
        if left_value == 0:
            continue
        for (right_2, right_3), right_value in right_items:
            level_2 = left_2 + right_2
            level_3 = left_3 + right_3
            if level_2 > maximum_level_2 or level_3 > maximum_level_3:
                continue
            key = (level_2, level_3)
            result[key] = result.get(key, 0.0j) + left_value * right_value
    return result


def _add_shifted_table(
    target: CoefficientTable,
    source: dict[LevelPair, complex],
    *,
    scale: complex,
    shift_2: int,
    shift_3: int,
    maximum_level_2: int,
    maximum_level_3: int,
) -> None:
    """Add a scaled and monomial-shifted bivariate series in place."""

    if scale == 0:
        return
    for (source_2, source_3), value in source.items():
        level_2 = source_2 + shift_2
        level_3 = source_3 + shift_3
        if level_2 > maximum_level_2 or level_3 > maximum_level_3:
            continue
        key = (level_2, level_3)
        target[key] = target.get(key, 0.0j) + scale * value


def _subtract_tables(
    left: dict[LevelPair, complex],
    right: dict[LevelPair, complex],
    maximum_level_2: int,
    maximum_level_3: int,
) -> CoefficientTable:
    """Subtract two finite rectangular coefficient tables."""

    return {
        (level_2, level_3): left.get((level_2, level_3), 0.0j)
        - right.get((level_2, level_3), 0.0j)
        for level_2 in range(maximum_level_2 + 1)
        for level_3 in range(maximum_level_3 + 1)
    }


def _divide_series(
    numerator: dict[LevelPair, complex],
    denominator: dict[LevelPair, complex],
    maximum_level_2: int,
    maximum_level_3: int,
) -> CoefficientTable:
    r"""Formally divide two bivariate series with nonzero constant term."""

    constant = denominator.get((0, 0), 0.0j)
    if abs(constant) <= 1.0e-14:
        raise ZeroDivisionError(
            "the selected auxiliary block has a vanishing constant term"
        )
    quotient: CoefficientTable = {}
    for total_level in range(maximum_level_2 + maximum_level_3 + 1):
        for level_2 in range(maximum_level_2 + 1):
            level_3 = total_level - level_2
            if level_3 < 0 or level_3 > maximum_level_3:
                continue
            residual = numerator.get((level_2, level_3), 0.0j)
            for denominator_2 in range(level_2 + 1):
                for denominator_3 in range(level_3 + 1):
                    if denominator_2 == 0 and denominator_3 == 0:
                        continue
                    residual -= denominator.get(
                        (denominator_2, denominator_3),
                        0.0j,
                    ) * quotient.get(
                        (level_2 - denominator_2, level_3 - denominator_3),
                        0.0j,
                    )
            quotient[(level_2, level_3)] = residual / constant
    return quotient


def _sum_component_tables(
    components: dict[int, CoefficientTable],
    maximum_level_2: int,
    maximum_level_3: int,
) -> CoefficientTable:
    """Sum the two homogeneous superconformal form components."""

    return {
        (level_2, level_3): sum(
            components[form_parity].get((level_2, level_3), 0.0j)
            for form_parity in (0, 1)
        )
        for level_2 in range(maximum_level_2 + 1)
        for level_3 in range(maximum_level_3 + 1)
    }


@dataclass(frozen=True)
class GenericRamondCoefficientResult:
    r"""Coefficient-level Ramond block obtained from the exact PDF identity.

    Levels in every table are integer excitation levels on Ramond theta
    edges two and three.  ``primary_components[f]`` and
    ``superdescendant_components[f]`` retain the two homogeneous
    superconformal form parities.  The unqualified tables are their sums,
    corresponding to the full HJS ground tensor used by the direct necklace
    oracle.
    """

    primary_primary: dict[LevelPair, complex]
    superdescendant_superdescendant: dict[LevelPair, complex]
    primary_components: dict[int, CoefficientTable]
    superdescendant_components: dict[int, CoefficientTable]
    tensor_cut_level_zero: dict[int, CoefficientTable]
    tensor_cut_level_half: dict[int, CoefficientTable]
    auxiliary_cut_level_zero: CoefficientTable
    auxiliary_cut_level_half: CoefficientTable
    b: complex
    maximum_level_2: int
    maximum_level_3: int
    virasoro_recursion_order: int
    auxiliary_form_parity: int
    branch_term_count: int
    max_branch_eigen_residual: float
    min_branch_spectral_gap: float

    @staticmethod
    def _evaluate_table(
        table: dict[LevelPair, complex],
        q_2: complex,
        q_3: complex,
    ) -> complex:
        return complex(
            sum(
                coefficient * complex(q_2) ** level_2 * complex(q_3) ** level_3
                for (level_2, level_3), coefficient in table.items()
            )
        )

    def primary_value(self, q_2: complex, q_3: complex) -> complex:
        """Evaluate the retained primary-primary descendant series."""

        return self._evaluate_table(self.primary_primary, q_2, q_3)

    def superdescendant_value(self, q_2: complex, q_3: complex) -> complex:
        """Evaluate the retained superdescendant-superdescendant series."""

        return self._evaluate_table(
            self.superdescendant_superdescendant,
            q_2,
            q_3,
        )


@dataclass(frozen=True)
class GenericRamondNecklaceCoefficientResult:
    r"""Ramond coefficients in the direct cyclic-necklace convention.

    Table keys are integer excitation levels ``(N_previous,N_current)``.
    Unlike :class:`GenericRamondCoefficientResult`, this result exchanges the
    two Ramond slots at the second local vertex and omits the fixed-global-
    theta gathering sign.  It can therefore be compared coefficient by
    coefficient with :func:`spin23_genus1_blocks.direct_ramond_torus_necklace_series`.
    """

    primary_primary: CoefficientTable
    superdescendant_superdescendant: CoefficientTable
    primary_components: dict[int, CoefficientTable]
    superdescendant_components: dict[int, CoefficientTable]
    tensor_cut_level_zero: dict[int, CoefficientTable]
    tensor_cut_level_half: dict[int, CoefficientTable]
    auxiliary_cut_level_zero: CoefficientTable
    auxiliary_cut_level_half: CoefficientTable
    b: complex
    maximum_previous_level: int
    maximum_current_level: int
    virasoro_recursion_order: int
    auxiliary_form_parity: int
    branch_term_count: int
    max_branch_eigen_residual: float
    min_branch_spectral_gap: float

    @staticmethod
    def _evaluate_table(
        table: CoefficientTable,
        q_previous: complex,
        q_current: complex,
    ) -> complex:
        return complex(
            sum(
                coefficient
                * complex(q_previous) ** previous_level
                * complex(q_current) ** current_level
                for (previous_level, current_level), coefficient in table.items()
            )
        )

    def primary_value(self, q_previous: complex, q_current: complex) -> complex:
        """Evaluate the retained primary-primary descendant series."""

        return self._evaluate_table(
            self.primary_primary,
            q_previous,
            q_current,
        )

    def superdescendant_value(
        self,
        q_previous: complex,
        q_current: complex,
    ) -> complex:
        """Evaluate the retained superdescendant-superdescendant series."""

        return self._evaluate_table(
            self.superdescendant_superdescendant,
            q_previous,
            q_current,
        )


def generic_ramond_torus_two_point_necklace_coefficients(
    *,
    b: complex,
    previous_internal_momentum: complex,
    current_internal_momentum: complex,
    external_ns_momentum: complex,
    left_structure_sign: int,
    right_structure_sign: int,
    previous_lift_sign: int = 1,
    current_lift_sign: int = 1,
    maximum_previous_level: int,
    maximum_current_level: int,
    virasoro_recursion_order: int | None = None,
    auxiliary_form_parity: int | None = None,
) -> GenericRamondNecklaceCoefficientResult:
    r"""Return recursive Ramond coefficients in cyclic necklace order.

    The complete tensor-product branch sum is assembled at generic ``b`` and
    divided coefficientwise by the auxiliary Majorana block.  The two local
    vertices use the same slot ordering as the direct inverse-Gram oracle,
    providing an independent finite-level validation of the recursion.

    The two homogeneous superconformal form parities are deconvolved
    separately.  At cut level one half, moving the odd physical cut state
    past a form of parity ``f`` contributes the graded factor
    :math:`(-1)^f`; omitting it reverses precisely the odd-form part of the
    ``G_-1/2 V`` block.

    This scalar deconvolution requires equal edge lifts.  For opposite
    lifts the auxiliary Ramond ground trace has zero constant term, so the
    formal quotient does not exist; resolving that trace would require an
    auxiliary ground-fiber matrix identity rather than scalar division.
    """

    b = complex(b)
    if b == 0 or b * b == 1:
        raise ValueError(
            "the branching generators require generic b; take the finite "
            "part only after assembling the complete coefficient table"
        )
    previous_momentum = complex(previous_internal_momentum)
    current_momentum = complex(current_internal_momentum)
    external_momentum = complex(external_ns_momentum)
    left_structure_sign = _validate_sign(
        left_structure_sign,
        "left_structure_sign",
    )
    right_structure_sign = _validate_sign(
        right_structure_sign,
        "right_structure_sign",
    )
    previous_lift_sign = _validate_sign(
        previous_lift_sign,
        "previous_lift_sign",
    )
    current_lift_sign = _validate_sign(
        current_lift_sign,
        "current_lift_sign",
    )
    if previous_lift_sign != current_lift_sign:
        raise ValueError(
            "the scalar auxiliary necklace block is not invertible for "
            "opposite edge lifts; use the direct finite-level oracle or a "
            "matrix-valued auxiliary deconvolution"
        )
    for value, name in (
        (maximum_previous_level, "maximum_previous_level"),
        (maximum_current_level, "maximum_current_level"),
    ):
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    complete_order = maximum_previous_level + maximum_current_level
    recursion_order = (
        complete_order
        if virasoro_recursion_order is None
        else int(virasoro_recursion_order)
    )
    if recursion_order < complete_order:
        raise ValueError(
            "virasoro_recursion_order must be at least the sum of the two "
            "Ramond-edge cutoffs"
        )
    if auxiliary_form_parity is None:
        auxiliary_parity = 0
    elif auxiliary_form_parity in (0, 1):
        auxiliary_parity = int(auxiliary_form_parity)
    else:
        raise ValueError("auxiliary_form_parity must be zero, one, or None")

    branches = ramond_branch_numbers(
        max(maximum_previous_level, maximum_current_level)
    )
    max_residual = 0.0
    min_gap = math.inf
    term_count = 0

    @lru_cache(maxsize=None)
    def branch_state(
        sector: str,
        momentum: complex,
        branch_number: BranchNumber,
        parity: int | None,
    ):
        return embedded_branch_state(
            b=b,
            sector=sector,  # type: ignore[arg-type]
            physical_momentum=momentum,
            branch_number=branch_number,
            parity=parity,
        )

    def tensor_table(
        cut_branches: tuple[BranchNumber, ...],
        super_form_parity: int,
    ) -> CoefficientTable:
        nonlocal max_residual, min_gap, term_count
        result: CoefficientTable = {
            (previous_level, current_level): 0.0j
            for previous_level in range(maximum_previous_level + 1)
            for current_level in range(maximum_current_level + 1)
        }
        for cut_number in cut_branches:
            cut_branch = branch_state(
                "NS",
                external_momentum,
                cut_number,
                None,
            )
            for previous_number in branches:
                previous_onset = _ramond_branch_level(previous_number)
                if previous_onset > maximum_previous_level:
                    continue
                for current_number in branches:
                    current_onset = _ramond_branch_level(current_number)
                    if current_onset > maximum_current_level:
                        continue
                    residual_previous = maximum_previous_level - previous_onset
                    residual_current = maximum_current_level - current_onset
                    for previous_parity in (0, 1):
                        previous = branch_state(
                            "R",
                            previous_momentum,
                            previous_number,
                            previous_parity,
                        )
                        for current_parity in (0, 1):
                            if (
                                cut_branch.parity
                                + previous_parity
                                + current_parity
                                - super_form_parity
                                - auxiliary_parity
                            ) % 2:
                                continue
                            current = branch_state(
                                "R",
                                current_momentum,
                                current_number,
                                current_parity,
                            )
                            branching = necklace_branching_coefficient_product(
                                previous,
                                cut_branch,
                                current,
                                left_structure_sign=left_structure_sign,
                                right_structure_sign=right_structure_sign,
                                super_form_parity=super_form_parity,
                                auxiliary_form_parity=auxiliary_parity,
                            )
                            if branching.value == 0:
                                continue
                            first = _ordinary_block_coefficients(
                                previous.parameters.c_1,
                                cut_branch.parameters.h_1,
                                previous.parameters.h_1,
                                current.parameters.h_1,
                                residual_previous,
                                residual_current,
                                recursion_order,
                                "necklace",
                            )
                            second = _ordinary_block_coefficients(
                                previous.parameters.c_2,
                                cut_branch.parameters.h_2,
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
                                previous_lift_sign**previous_parity
                                * current_lift_sign**current_parity
                                * branching.value
                            )
                            _add_shifted_table(
                                result,
                                ordinary_product,
                                scale=scale,
                                shift_2=previous_onset,
                                shift_3=current_onset,
                                maximum_level_2=maximum_previous_level,
                                maximum_level_3=maximum_current_level,
                            )
                            term_count += 1
                            max_residual = max(
                                max_residual,
                                branching.max_eigen_residual,
                            )
                            min_gap = min(
                                min_gap,
                                branching.min_spectral_gap,
                            )
        return result

    auxiliary_zero = auxiliary_ramond_necklace_coefficients(
        cut_state=AuxiliaryFermionState("NS"),
        maximum_previous_level=maximum_previous_level,
        maximum_current_level=maximum_current_level,
        previous_lift_sign=previous_lift_sign,
        current_lift_sign=current_lift_sign,
        form_parity=auxiliary_parity,
    )
    auxiliary_half = auxiliary_ramond_necklace_coefficients(
        cut_state=AuxiliaryFermionState("NS", (1,)),
        maximum_previous_level=maximum_previous_level,
        maximum_current_level=maximum_current_level,
        previous_lift_sign=previous_lift_sign,
        current_lift_sign=current_lift_sign,
        form_parity=auxiliary_parity,
    )
    if abs(auxiliary_zero.get((0, 0), 0.0j)) <= 1.0e-14:
        raise ZeroDivisionError(
            "the selected auxiliary form has an unsaturated ground zero mode"
        )

    tensor_zero: dict[int, CoefficientTable] = {}
    tensor_half: dict[int, CoefficientTable] = {}
    primary_components: dict[int, CoefficientTable] = {}
    superdescendant_components: dict[int, CoefficientTable] = {}
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
        primary = _divide_series(
            tensor_zero[form_parity],
            auxiliary_zero,
            maximum_previous_level,
            maximum_current_level,
        )
        mixed_auxiliary = _convolve_tables(
            primary,
            auxiliary_half,
            maximum_previous_level,
            maximum_current_level,
        )
        cut_half_numerator = _subtract_tables(
            tensor_half[form_parity],
            mixed_auxiliary,
            maximum_previous_level,
            maximum_current_level,
        )
        cut_half = _divide_series(
            cut_half_numerator,
            auxiliary_zero,
            maximum_previous_level,
            maximum_current_level,
        )
        primary_components[form_parity] = primary
        superdescendant_components[form_parity] = {
            key: (-1) ** form_parity * 2.0 * external_weight * value
            for key, value in cut_half.items()
        }
    return GenericRamondNecklaceCoefficientResult(
        primary_primary=_sum_component_tables(
            primary_components,
            maximum_previous_level,
            maximum_current_level,
        ),
        superdescendant_superdescendant=_sum_component_tables(
            superdescendant_components,
            maximum_previous_level,
            maximum_current_level,
        ),
        primary_components=primary_components,
        superdescendant_components=superdescendant_components,
        tensor_cut_level_zero=tensor_zero,
        tensor_cut_level_half=tensor_half,
        auxiliary_cut_level_zero=auxiliary_zero,
        auxiliary_cut_level_half=auxiliary_half,
        b=b,
        maximum_previous_level=maximum_previous_level,
        maximum_current_level=maximum_current_level,
        virasoro_recursion_order=recursion_order,
        auxiliary_form_parity=auxiliary_parity,
        branch_term_count=term_count,
        max_branch_eigen_residual=max_residual,
        min_branch_spectral_gap=min_gap,
    )


def generic_ramond_torus_two_point_coefficients(
    *,
    b: complex,
    momentum_2: complex,
    momentum_3: complex,
    external_ns_momentum: complex,
    left_structure_sign: int,
    right_structure_sign: int,
    lift_sign_2: int = 1,
    lift_sign_3: int = 1,
    maximum_level_2: int,
    maximum_level_3: int,
    virasoro_recursion_order: int | None = None,
    auxiliary_form_parity: int | None = None,
) -> GenericRamondCoefficientResult:
    r"""Implement equations (6.5)--(6.7) as finite formal series.

    This routine does not use a Ramond large-:math:`c` seed.  At generic
    ``b`` it constructs the locally finite two-Virasoro branch sum in the
    supplied PDF, using directly evaluated branch restrictions and two
    ordinary CCY Virasoro recursions.  It then divides by the auxiliary
    Majorana block coefficient by coefficient.

    The requested rectangle is complete only if the ordinary recursion can
    contain singular-vector insertions on both Ramond edges.  Accordingly,
    ``virasoro_recursion_order`` defaults to ``maximum_level_2 +
    maximum_level_3`` and smaller values are rejected rather than silently
    returning an incompletely recursive table.
    """

    b = complex(b)
    if b == 0 or b * b == 1:
        raise ValueError(
            "the branching generators require generic b; take the finite "
            "part only after assembling the complete coefficient table"
        )
    momentum_2 = complex(momentum_2)
    momentum_3 = complex(momentum_3)
    external_momentum = complex(external_ns_momentum)
    left_structure_sign = _validate_sign(
        left_structure_sign,
        "left_structure_sign",
    )
    right_structure_sign = _validate_sign(
        right_structure_sign,
        "right_structure_sign",
    )
    lift_sign_2 = _validate_sign(lift_sign_2, "lift_sign_2")
    lift_sign_3 = _validate_sign(lift_sign_3, "lift_sign_3")
    for value, name in (
        (maximum_level_2, "maximum_level_2"),
        (maximum_level_3, "maximum_level_3"),
    ):
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    complete_order = maximum_level_2 + maximum_level_3
    recursion_order = (
        complete_order
        if virasoro_recursion_order is None
        else int(virasoro_recursion_order)
    )
    if recursion_order < complete_order:
        raise ValueError(
            "virasoro_recursion_order must be at least the sum of the two "
            "Ramond-edge cutoffs"
        )
    if auxiliary_form_parity is None:
        # The even auxiliary form has a nonzero ground sewing for equal
        # Ramond lifts; the odd form soaks the zero mode for opposite lifts.
        auxiliary_parity = 0 if lift_sign_2 * lift_sign_3 == 1 else 1
    elif auxiliary_form_parity in (0, 1):
        auxiliary_parity = int(auxiliary_form_parity)
    else:
        raise ValueError("auxiliary_form_parity must be zero, one, or None")

    maximum_branch_level = max(maximum_level_2, maximum_level_3)
    branches = ramond_branch_numbers(maximum_branch_level)
    max_residual = 0.0
    min_gap = math.inf
    term_count = 0

    @lru_cache(maxsize=None)
    def branch_state(
        sector: str,
        momentum: complex,
        branch_number: BranchNumber,
        parity: int | None,
    ):
        return embedded_branch_state(
            b=b,
            sector=sector,  # type: ignore[arg-type]
            physical_momentum=momentum,
            branch_number=branch_number,
            parity=parity,
        )

    def tensor_table(
        *,
        cut_branches: tuple[BranchNumber, ...],
        super_form_parity: int,
    ) -> CoefficientTable:
        nonlocal max_residual, min_gap, term_count
        result: CoefficientTable = {
            (level_2, level_3): 0.0j
            for level_2 in range(maximum_level_2 + 1)
            for level_3 in range(maximum_level_3 + 1)
        }
        for cut_number in cut_branches:
            twice_cut_number = int(2 * cut_number)
            cut_branch = branch_state(
                "NS",
                external_momentum,
                cut_number,
                None,
            )
            for number_2 in branches:
                onset_2 = _ramond_branch_level(number_2)
                if onset_2 > maximum_level_2:
                    continue
                for number_3 in branches:
                    onset_3 = _ramond_branch_level(number_3)
                    if onset_3 > maximum_level_3:
                        continue
                    residual_maximum_2 = maximum_level_2 - onset_2
                    residual_maximum_3 = maximum_level_3 - onset_3
                    for parity_2 in (0, 1):
                        for parity_3 in (0, 1):
                            if (
                                twice_cut_number
                                + parity_2
                                + parity_3
                                - super_form_parity
                                - auxiliary_parity
                            ) % 2:
                                continue
                            branch_2 = branch_state(
                                "R",
                                momentum_2,
                                number_2,
                                parity_2,
                            )
                            branch_3 = branch_state(
                                "R",
                                momentum_3,
                                number_3,
                                parity_3,
                            )
                            branching = theta_branching_coefficient_product(
                                cut_branch,
                                branch_2,
                                branch_3,
                                left_structure_sign=left_structure_sign,
                                right_structure_sign=right_structure_sign,
                                super_form_parity=super_form_parity,
                                auxiliary_form_parity=auxiliary_parity,
                            )
                            if branching.value == 0:
                                continue
                            first = _ordinary_block_coefficients(
                                branch_2.parameters.c_1,
                                cut_branch.parameters.h_1,
                                branch_2.parameters.h_1,
                                branch_3.parameters.h_1,
                                residual_maximum_2,
                                residual_maximum_3,
                                recursion_order,
                                "theta",
                            )
                            second = _ordinary_block_coefficients(
                                branch_2.parameters.c_2,
                                cut_branch.parameters.h_2,
                                branch_2.parameters.h_2,
                                branch_3.parameters.h_2,
                                residual_maximum_2,
                                residual_maximum_3,
                                recursion_order,
                                "theta",
                            )
                            ordinary_product = _convolve_tables(
                                first,
                                second,
                                residual_maximum_2,
                                residual_maximum_3,
                            )
                            gathering = (-1) ** (
                                twice_cut_number * parity_2
                                + twice_cut_number * parity_3
                                + parity_2 * parity_3
                            )
                            lift = (
                                lift_sign_2**parity_2
                                * lift_sign_3**parity_3
                            )
                            scale = gathering * lift * branching.value
                            _add_shifted_table(
                                result,
                                ordinary_product,
                                scale=scale,
                                shift_2=onset_2,
                                shift_3=onset_3,
                                maximum_level_2=maximum_level_2,
                                maximum_level_3=maximum_level_3,
                            )
                            term_count += 1
                            max_residual = max(
                                max_residual,
                                branching.max_eigen_residual,
                            )
                            min_gap = min(
                                min_gap,
                                branching.min_spectral_gap,
                            )
        return result

    auxiliary_zero = auxiliary_ramond_theta_coefficients(
        cut_state=AuxiliaryFermionState("NS"),
        maximum_level_2=maximum_level_2,
        maximum_level_3=maximum_level_3,
        lift_sign_2=lift_sign_2,
        lift_sign_3=lift_sign_3,
        form_parity=auxiliary_parity,
    )
    auxiliary_half = auxiliary_ramond_theta_coefficients(
        cut_state=AuxiliaryFermionState("NS", (1,)),
        maximum_level_2=maximum_level_2,
        maximum_level_3=maximum_level_3,
        lift_sign_2=lift_sign_2,
        lift_sign_3=lift_sign_3,
        form_parity=auxiliary_parity,
    )
    if abs(auxiliary_zero.get((0, 0), 0.0j)) <= 1.0e-14:
        raise ZeroDivisionError(
            "the selected auxiliary form has an unsaturated ground zero mode"
        )

    tensor_zero: dict[int, CoefficientTable] = {}
    tensor_half: dict[int, CoefficientTable] = {}
    primary_components: dict[int, CoefficientTable] = {}
    superdescendant_components: dict[int, CoefficientTable] = {}
    external_weight = ns_liouville_weight(external_momentum, b)
    for form_parity in (0, 1):
        tensor_zero[form_parity] = tensor_table(
            cut_branches=(Fraction(0),),
            super_form_parity=form_parity,
        )
        tensor_half[form_parity] = tensor_table(
            cut_branches=(Fraction(-1, 2), Fraction(1, 2)),
            super_form_parity=form_parity,
        )
        primary = _divide_series(
            tensor_zero[form_parity],
            auxiliary_zero,
            maximum_level_2,
            maximum_level_3,
        )
        mixed_auxiliary = _convolve_tables(
            primary,
            auxiliary_half,
            maximum_level_2,
            maximum_level_3,
        )
        cut_half_numerator = _subtract_tables(
            tensor_half[form_parity],
            mixed_auxiliary,
            maximum_level_2,
            maximum_level_3,
        )
        cut_half = _divide_series(
            cut_half_numerator,
            auxiliary_zero,
            maximum_level_2,
            maximum_level_3,
        )
        primary_components[form_parity] = primary
        superdescendant_components[form_parity] = {
            key: 2.0 * external_weight * value
            for key, value in cut_half.items()
        }

    return GenericRamondCoefficientResult(
        primary_primary=_sum_component_tables(
            primary_components,
            maximum_level_2,
            maximum_level_3,
        ),
        superdescendant_superdescendant=_sum_component_tables(
            superdescendant_components,
            maximum_level_2,
            maximum_level_3,
        ),
        primary_components=primary_components,
        superdescendant_components=superdescendant_components,
        tensor_cut_level_zero=tensor_zero,
        tensor_cut_level_half=tensor_half,
        auxiliary_cut_level_zero=auxiliary_zero,
        auxiliary_cut_level_half=auxiliary_half,
        b=b,
        maximum_level_2=maximum_level_2,
        maximum_level_3=maximum_level_3,
        virasoro_recursion_order=recursion_order,
        auxiliary_form_parity=auxiliary_parity,
        branch_term_count=term_count,
        max_branch_eigen_residual=max_residual,
        min_branch_spectral_gap=min_gap,
    )


@dataclass(frozen=True)
class SelfDualRamondNecklaceCoefficientResult:
    r"""Coefficientwise :math:`b=1` Ramond necklace block.

    The finite part is taken separately for every bivariate plumbing
    coefficient before the series is evaluated.  This is stronger than
    projecting a value at fixed ``q``: the latter can hide cancellations
    between different plumbing orders and is not a valid definition of the
    self-dual conformal block.
    """

    primary_primary: CoefficientTable
    superdescendant_superdescendant: CoefficientTable
    primary_diagnostics: dict[LevelPair, FinitePartDiagnostics]
    superdescendant_diagnostics: dict[LevelPair, FinitePartDiagnostics]
    maximum_previous_level: int
    maximum_current_level: int
    virasoro_recursion_order: int
    radius: float
    check_radius: float
    samples: int
    generic_evaluations: int
    max_branch_eigen_residual: float
    min_branch_spectral_gap: float

    @staticmethod
    def _evaluate_table(
        table: CoefficientTable,
        q_previous: complex,
        q_current: complex,
    ) -> complex:
        return complex(
            sum(
                coefficient
                * complex(q_previous) ** previous_level
                * complex(q_current) ** current_level
                for (previous_level, current_level), coefficient in table.items()
            )
        )

    def primary_value(self, q_previous: complex, q_current: complex) -> complex:
        """Evaluate the finite primary-primary descendant series."""

        return self._evaluate_table(
            self.primary_primary,
            q_previous,
            q_current,
        )

    def superdescendant_value(
        self,
        q_previous: complex,
        q_current: complex,
    ) -> complex:
        """Evaluate the finite superdescendant descendant series."""

        return self._evaluate_table(
            self.superdescendant_superdescendant,
            q_previous,
            q_current,
        )


def self_dual_ramond_torus_two_point_necklace_coefficients(
    *,
    previous_internal_momentum: complex,
    current_internal_momentum: complex,
    external_ns_momentum: complex,
    left_structure_sign: int,
    right_structure_sign: int,
    previous_lift_sign: int = 1,
    current_lift_sign: int = 1,
    maximum_previous_level: int,
    maximum_current_level: int,
    virasoro_recursion_order: int | None = None,
    auxiliary_form_parity: int | None = None,
    radius: float = 0.04,
    check_radius: float = 0.05,
    samples: int = 24,
) -> SelfDualRamondNecklaceCoefficientResult:
    r"""Project the complete generic-``b`` coefficient table to ``b=1``.

    All two-Virasoro branches and the auxiliary-fermion quotient are
    assembled at generic ``b`` first.  The constant Laurent coefficient in
    ``t=log(b)`` is then projected independently at each pair of Ramond edge
    levels.  The two contour radii provide a numerical continuation
    diagnostic rather than a rigorous error bound.
    """

    for value, name in (
        (maximum_previous_level, "maximum_previous_level"),
        (maximum_current_level, "maximum_current_level"),
    ):
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    complete_order = maximum_previous_level + maximum_current_level
    recursion_order = (
        complete_order
        if virasoro_recursion_order is None
        else int(virasoro_recursion_order)
    )
    if recursion_order < complete_order:
        raise ValueError(
            "virasoro_recursion_order must be at least the sum of the two "
            "Ramond-edge cutoffs"
        )

    coefficient_levels = tuple(
        (previous_level, current_level)
        for previous_level in range(maximum_previous_level + 1)
        for current_level in range(maximum_current_level + 1)
    )
    keys = tuple(
        (component, previous_level, current_level)
        for component in ("PP", "GG")
        for previous_level, current_level in coefficient_levels
    )
    max_residual = 0.0
    min_gap = math.inf
    generic_evaluations = 0

    @lru_cache(maxsize=None)
    def evaluate_at_b(b: complex) -> tuple[tuple[tuple[str, int, int], complex], ...]:
        nonlocal max_residual, min_gap, generic_evaluations
        result = generic_ramond_torus_two_point_necklace_coefficients(
            b=b,
            previous_internal_momentum=previous_internal_momentum,
            current_internal_momentum=current_internal_momentum,
            external_ns_momentum=external_ns_momentum,
            left_structure_sign=left_structure_sign,
            right_structure_sign=right_structure_sign,
            previous_lift_sign=previous_lift_sign,
            current_lift_sign=current_lift_sign,
            maximum_previous_level=maximum_previous_level,
            maximum_current_level=maximum_current_level,
            virasoro_recursion_order=recursion_order,
            auxiliary_form_parity=auxiliary_form_parity,
        )
        generic_evaluations += 1
        max_residual = max(max_residual, result.max_branch_eigen_residual)
        min_gap = min(min_gap, result.min_branch_spectral_gap)
        values = {
            **{
                ("PP", *level): value
                for level, value in result.primary_primary.items()
            },
            **{
                ("GG", *level): value
                for level, value in result.superdescendant_superdescendant.items()
            },
        }
        return tuple(sorted(values.items()))

    def evaluator(b: complex) -> dict[tuple[str, int, int], complex]:
        return dict(evaluate_at_b(complex(b)))

    values, diagnostics = dictionary_finite_part(
        evaluator,
        keys=keys,
        radius=radius,
        check_radius=check_radius,
        samples=samples,
        # The complete two-Virasoro branch sum is invariant under the
        # Liouville self-duality b <-> 1/b.  Antipodal t=log(b) contour
        # samples are therefore identical only after this assembled sum.
        inversion_symmetric=True,
    )
    primary = {
        level: values[("PP", *level)] for level in coefficient_levels
    }
    superdescendant = {
        level: values[("GG", *level)] for level in coefficient_levels
    }
    return SelfDualRamondNecklaceCoefficientResult(
        primary_primary=primary,
        superdescendant_superdescendant=superdescendant,
        primary_diagnostics={
            level: diagnostics[("PP", *level)]
            for level in coefficient_levels
        },
        superdescendant_diagnostics={
            level: diagnostics[("GG", *level)]
            for level in coefficient_levels
        },
        maximum_previous_level=maximum_previous_level,
        maximum_current_level=maximum_current_level,
        virasoro_recursion_order=recursion_order,
        radius=float(radius),
        check_radius=float(check_radius),
        samples=int(samples),
        generic_evaluations=generic_evaluations,
        max_branch_eigen_residual=max_residual,
        min_branch_spectral_gap=min_gap,
    )


__all__ = [
    "GenericRamondCoefficientResult",
    "GenericRamondNecklaceCoefficientResult",
    "SelfDualRamondNecklaceCoefficientResult",
    "generic_ramond_torus_two_point_coefficients",
    "generic_ramond_torus_two_point_necklace_coefficients",
    "ramond_branch_numbers",
    "self_dual_ramond_torus_two_point_necklace_coefficients",
]
