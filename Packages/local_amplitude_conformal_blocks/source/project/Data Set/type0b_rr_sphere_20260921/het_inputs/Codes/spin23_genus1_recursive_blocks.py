#!/usr/bin/env python3
r"""Certified recursive two-point blocks at the self-dual Liouville point.

The direct inverse-Gram blocks in :mod:`spin23_genus1_blocks` remain the
definition and low-level oracle.  This module only adapts recursive results
that have an independent coefficient-level comparison with that oracle:

* bottom-component NS--NS external fields, using the two-edge NS
  internal-weight recursion;
* equal-momentum NS PP or GG external components, using the independently
  validated two-Virasoro decomposition; and
* equal-momentum Ramond-handle PP or GG external components, using the
  two-Virasoro decomposition and coefficientwise ``b=1`` finite part.

Mixed external components and necklaces with more than two vertices are not
silently approximated here.  Their all-level matrix recursion has not yet
been derived and certified, so callers must retain the direct finite-level
definition for those cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Literal, Mapping, Sequence

from ns_algebra.ns_sca import G, Word
from spin23_genus1_blocks import TorusNecklaceBlockSeries
from spin23_genus1_recursion import (
    FinitePartDiagnostics,
    SelfDualNSTorusTwoPointHRecursion,
)
from spin23_ramond_torus_recursion import (
    SelfDualRamondNecklaceCoefficientResult,
    self_dual_ramond_torus_two_point_necklace_coefficients,
)
from spin23_ns_torus_two_virasoro import (
    SelfDualNSNecklaceCoefficientResult,
    self_dual_ns_torus_two_point_necklace_coefficients,
)
from spin23_super_liouville_data import (
    ns_weight,
    ramond_liouville_weight,
)


RecursiveSector = Literal["NS", "R"]
EMPTY_WORD: Word = ()
G_MINUS_HALF: Word = (G(Fraction(-1, 2)),)


@dataclass(frozen=True)
class RecursiveTorusBlockResult:
    """One recursive series and its coefficientwise continuation diagnostic."""

    series: TorusNecklaceBlockSeries
    sector: RecursiveSector
    component: Literal["PP", "GG"]
    finite_part_diagnostics: Mapping[
        tuple[int, ...], FinitePartDiagnostics
    ]
    max_branch_eigen_residual: float | None = None
    min_branch_spectral_gap: float | None = None

    @property
    def maximum_finite_part_absolute_error(self) -> float:
        """Return the largest change under the second contour radius."""

        return max(
            (
                diagnostic.absolute_error
                for diagnostic in self.finite_part_diagnostics.values()
            ),
            default=0.0,
        )

    @property
    def maximum_finite_part_relative_error(self) -> float:
        """Return the largest coefficientwise relative contour change."""

        return max(
            (
                diagnostic.relative_error
                for diagnostic in self.finite_part_diagnostics.values()
            ),
            default=0.0,
        )

    @property
    def scaled_finite_part_error(self) -> float:
        """Return the largest contour change relative to the table scale."""

        scale = max(
            1.0,
            *(
                max(abs(diagnostic.value), abs(diagnostic.check_value))
                for diagnostic in self.finite_part_diagnostics.values()
            ),
        )
        return self.maximum_finite_part_absolute_error / scale


def _two_entries(
    values: Sequence[complex],
    name: str,
) -> tuple[complex, complex]:
    result = tuple(complex(value) for value in values)
    if len(result) != 2:
        raise ValueError(f"{name} must contain exactly two entries")
    return result  # type: ignore[return-value]


def _two_cutoffs(
    maximum_twice_levels: int | Sequence[int],
) -> tuple[int, int]:
    if isinstance(maximum_twice_levels, int):
        result = (maximum_twice_levels, maximum_twice_levels)
    else:
        result = tuple(maximum_twice_levels)
    if len(result) != 2:
        raise ValueError("maximum_twice_levels must contain two entries")
    if any(not isinstance(value, int) for value in result):
        raise TypeError("both descendant cutoffs must be integers")
    if any(value < 0 for value in result):
        raise ValueError("both descendant cutoffs must be nonnegative")
    return result  # type: ignore[return-value]


def _two_lifts(edge_lift_signs: Sequence[int] | None) -> tuple[int, int]:
    result = (1, 1) if edge_lift_signs is None else tuple(edge_lift_signs)
    if len(result) != 2:
        raise ValueError("edge_lift_signs must contain two entries")
    if any(value not in (-1, 1) for value in result):
        raise ValueError("edge lift signs must be +1 or -1")
    return result  # type: ignore[return-value]


def _two_words(external_words: Sequence[Word] | None) -> tuple[Word, Word]:
    result = (
        (EMPTY_WORD, EMPTY_WORD)
        if external_words is None
        else tuple(tuple(word) for word in external_words)
    )
    if len(result) != 2:
        raise ValueError("external_words must contain two entries")
    return result  # type: ignore[return-value]


@lru_cache(maxsize=4096)
def _recursive_ns_cached(
    internal_momenta: tuple[complex, complex],
    external_momenta: tuple[complex, complex],
    cutoffs: tuple[int, int],
    lifts: tuple[int, int],
    form_parity: int,
    radius: float,
    check_radius: float,
    samples: int,
) -> RecursiveTorusBlockResult:
    recursion = SelfDualNSTorusTwoPointHRecursion(
        internal_momentum_1=internal_momenta[0],
        internal_momentum_2=internal_momenta[1],
        external_momentum_1=external_momenta[0],
        external_momentum_2=external_momenta[1],
        form_parity=form_parity,
        radius=radius,
        check_radius=check_radius,
        samples=samples,
    )
    raw = recursion.coefficients(*cutoffs)
    coefficients = {
        levels: value
        * lifts[0] ** levels[0]
        * lifts[1] ** levels[1]
        for levels, value in raw.items()
    }
    series = TorusNecklaceBlockSeries(
        coefficients=coefficients,
        sector="NS",
        c=13.5,
        internal_weights=tuple(
            ns_weight(momentum) for momentum in internal_momenta
        ),
        external_weights=tuple(
            ns_weight(momentum) for momentum in external_momenta
        ),
        external_words=(EMPTY_WORD, EMPTY_WORD),
        edge_lift_signs=lifts,
        maximum_twice_levels=cutoffs,
        gram_condition_numbers={},
    )
    return RecursiveTorusBlockResult(
        series=series,
        sector="NS",
        component="PP",
        finite_part_diagnostics=recursion.diagnostics(*cutoffs),
    )


@lru_cache(maxsize=4096)
def _self_dual_ns_coefficients_cached(
    internal_momenta: tuple[complex, complex],
    external_momentum: complex,
    cutoffs: tuple[int, int],
    lifts: tuple[int, int],
    radius: float,
    check_radius: float,
    samples: int,
) -> SelfDualNSNecklaceCoefficientResult:
    # At vertex zero, direct necklace order is (edge 1, external, edge 0).
    return self_dual_ns_torus_two_point_necklace_coefficients(
        previous_internal_momentum=internal_momenta[1],
        current_internal_momentum=internal_momenta[0],
        external_ns_momentum=external_momentum,
        previous_lift_sign=lifts[1],
        current_lift_sign=lifts[0],
        maximum_previous_twice_level=cutoffs[1],
        maximum_current_twice_level=cutoffs[0],
        radius=radius,
        check_radius=check_radius,
        samples=samples,
    )


@lru_cache(maxsize=8192)
def _recursive_ns_two_virasoro_cached(
    internal_momenta: tuple[complex, complex],
    external_momentum: complex,
    cutoffs: tuple[int, int],
    lifts: tuple[int, int],
    form_parity: int,
    component: Literal["PP", "GG"],
    radius: float,
    check_radius: float,
    samples: int,
) -> RecursiveTorusBlockResult:
    recursive = _self_dual_ns_coefficients_cached(
        internal_momenta,
        external_momentum,
        cutoffs,
        lifts,
        radius,
        check_radius,
        samples,
    )
    selected = (
        recursive.primary_components[form_parity]
        if component == "PP"
        else recursive.superdescendant_components[form_parity]
    )
    selected_diagnostics = (
        recursive.primary_diagnostics[form_parity]
        if component == "PP"
        else recursive.superdescendant_diagnostics[form_parity]
    )
    coefficients = {
        (current_level, previous_level): value
        for (previous_level, current_level), value in selected.items()
    }
    diagnostics = {
        (current_level, previous_level): value
        for (previous_level, current_level), value in (
            selected_diagnostics.items()
        )
    }
    word = EMPTY_WORD if component == "PP" else G_MINUS_HALF
    series = TorusNecklaceBlockSeries(
        coefficients=coefficients,
        sector="NS",
        c=13.5,
        internal_weights=tuple(
            ns_weight(momentum) for momentum in internal_momenta
        ),
        external_weights=(ns_weight(external_momentum),) * 2,
        external_words=(word, word),
        edge_lift_signs=lifts,
        maximum_twice_levels=cutoffs,
        gram_condition_numbers={},
    )
    return RecursiveTorusBlockResult(
        series=series,
        sector="NS",
        component=component,
        finite_part_diagnostics=diagnostics,
        max_branch_eigen_residual=recursive.max_branch_eigen_residual,
        min_branch_spectral_gap=recursive.min_branch_spectral_gap,
    )


def recursive_ns_torus_two_point_series(
    *,
    internal_momenta: Sequence[complex],
    external_ns_momenta: Sequence[complex],
    maximum_twice_levels: int | Sequence[int],
    form_parity: int,
    edge_lift_signs: Sequence[int] | None = None,
    external_words: Sequence[Word] | None = None,
    radius: float = 0.04,
    check_radius: float = 0.05,
    samples: int = 24,
    equality_tolerance: float = 1.0e-12,
) -> RecursiveTorusBlockResult:
    r"""Return a certified self-dual NS PP or GG two-point series.

    ``form_parity`` selects the same homogeneous NS three-form at both
    vertices.  Equal-momentum PP and GG components use the two-Virasoro
    construction.  Unequal-momentum PP components use the scalar
    internal-weight recursion.  Mixed components and unequal-momentum GG
    components have not been independently certified and are rejected.
    """

    momenta = _two_entries(internal_momenta, "internal_momenta")
    external = _two_entries(external_ns_momenta, "external_ns_momenta")
    cutoffs = _two_cutoffs(maximum_twice_levels)
    lifts = _two_lifts(edge_lift_signs)
    words = _two_words(external_words)
    if words == (EMPTY_WORD, EMPTY_WORD):
        component: Literal["PP", "GG"] = "PP"
    elif words == (G_MINUS_HALF, G_MINUS_HALF):
        component = "GG"
    else:
        raise NotImplementedError(
            "the certified NS recursion supports only PP or GG external "
            "components"
        )
    if form_parity not in (0, 1):
        raise ValueError("form_parity must be zero or one")
    scale = max(1.0, abs(external[0]), abs(external[1]))
    equal_external_momenta = (
        abs(external[0] - external[1]) <= equality_tolerance * scale
    )
    if equal_external_momenta:
        return _recursive_ns_two_virasoro_cached(
            momenta,
            external[0],
            cutoffs,
            lifts,
            int(form_parity),
            component,
            float(radius),
            float(check_radius),
            int(samples),
        )
    if component == "GG":
        raise NotImplementedError(
            "the certified NS GG recursion currently requires equal "
            "external Liouville momenta"
        )
    return _recursive_ns_cached(
        momenta,
        external,
        cutoffs,
        lifts,
        int(form_parity),
        float(radius),
        float(check_radius),
        int(samples),
    )


@lru_cache(maxsize=4096)
def _self_dual_ramond_coefficients_cached(
    internal_momenta: tuple[complex, complex],
    external_momentum: complex,
    structure_signs: tuple[int, int],
    cutoffs: tuple[int, int],
    lifts: tuple[int, int],
    radius: float,
    check_radius: float,
    samples: int,
) -> SelfDualRamondNecklaceCoefficientResult:
    # At vertex zero, direct necklace order is (edge 1, external, edge 0).
    return self_dual_ramond_torus_two_point_necklace_coefficients(
        previous_internal_momentum=internal_momenta[1],
        current_internal_momentum=internal_momenta[0],
        external_ns_momentum=external_momentum,
        left_structure_sign=structure_signs[0],
        right_structure_sign=structure_signs[1],
        previous_lift_sign=lifts[1],
        current_lift_sign=lifts[0],
        maximum_previous_level=cutoffs[1] // 2,
        maximum_current_level=cutoffs[0] // 2,
        radius=radius,
        check_radius=check_radius,
        samples=samples,
    )


@lru_cache(maxsize=8192)
def _recursive_ramond_cached(
    internal_momenta: tuple[complex, complex],
    external_momentum: complex,
    structure_signs: tuple[int, int],
    cutoffs: tuple[int, int],
    lifts: tuple[int, int],
    component: Literal["PP", "GG"],
    radius: float,
    check_radius: float,
    samples: int,
) -> RecursiveTorusBlockResult:
    recursive = _self_dual_ramond_coefficients_cached(
        internal_momenta,
        external_momentum,
        structure_signs,
        cutoffs,
        lifts,
        radius,
        check_radius,
        samples,
    )
    selected = (
        recursive.primary_primary
        if component == "PP"
        else recursive.superdescendant_superdescendant
    )
    selected_diagnostics = (
        recursive.primary_diagnostics
        if component == "PP"
        else recursive.superdescendant_diagnostics
    )
    coefficients = {
        (2 * current_level, 2 * previous_level): value
        for (previous_level, current_level), value in selected.items()
    }
    diagnostics = {
        (2 * current_level, 2 * previous_level): value
        for (previous_level, current_level), value in selected_diagnostics.items()
    }
    word = EMPTY_WORD if component == "PP" else G_MINUS_HALF
    series = TorusNecklaceBlockSeries(
        coefficients=coefficients,
        sector="R",
        c=13.5,
        internal_weights=tuple(
            ramond_liouville_weight(momentum)
            for momentum in internal_momenta
        ),
        external_weights=(ns_weight(external_momentum),) * 2,
        external_words=(word, word),
        edge_lift_signs=lifts,
        maximum_twice_levels=(
            cutoffs[0] - cutoffs[0] % 2,
            cutoffs[1] - cutoffs[1] % 2,
        ),
        gram_condition_numbers={},
    )
    return RecursiveTorusBlockResult(
        series=series,
        sector="R",
        component=component,
        finite_part_diagnostics=diagnostics,
        max_branch_eigen_residual=recursive.max_branch_eigen_residual,
        min_branch_spectral_gap=recursive.min_branch_spectral_gap,
    )


def recursive_ramond_torus_two_point_series(
    *,
    internal_momenta: Sequence[complex],
    external_ns_momenta: Sequence[complex],
    structure_signs: Sequence[int],
    maximum_twice_levels: int | Sequence[int],
    edge_lift_signs: Sequence[int] | None = None,
    external_words: Sequence[Word] | None = None,
    radius: float = 0.04,
    check_radius: float = 0.05,
    samples: int = 24,
    equality_tolerance: float = 1.0e-12,
) -> RecursiveTorusBlockResult:
    r"""Return the certified self-dual long-R PP or GG two-point series."""

    momenta = _two_entries(internal_momenta, "internal_momenta")
    external = _two_entries(external_ns_momenta, "external_ns_momenta")
    signs = tuple(int(value) for value in structure_signs)
    if len(signs) != 2 or any(value not in (-1, 1) for value in signs):
        raise ValueError("structure_signs must contain two signs")
    cutoffs = _two_cutoffs(maximum_twice_levels)
    lifts = _two_lifts(edge_lift_signs)
    words = _two_words(external_words)
    if lifts[0] != lifts[1]:
        raise NotImplementedError(
            "opposite Ramond edge lifts require a matrix-valued auxiliary "
            "ground-fiber deconvolution"
        )
    scale = max(1.0, abs(external[0]), abs(external[1]))
    if abs(external[0] - external[1]) > equality_tolerance * scale:
        raise NotImplementedError(
            "the certified two-Virasoro adapter currently requires equal "
            "external Liouville momenta"
        )
    if words == (EMPTY_WORD, EMPTY_WORD):
        component: Literal["PP", "GG"] = "PP"
    elif words == (G_MINUS_HALF, G_MINUS_HALF):
        component = "GG"
    else:
        raise NotImplementedError(
            "the certified Ramond recursion supports only PP or GG external "
            "components"
        )
    return _recursive_ramond_cached(
        momenta,
        external[0],
        signs,  # type: ignore[arg-type]
        cutoffs,
        lifts,
        component,
        float(radius),
        float(check_radius),
        int(samples),
    )


__all__ = [
    "RecursiveTorusBlockResult",
    "recursive_ns_torus_two_point_series",
    "recursive_ramond_torus_two_point_series",
]
