#!/usr/bin/env python3
"""Coefficient-level checks for the certified genus-one recursion adapters."""

from __future__ import annotations

import math
import unittest

from ramond_algebra.nrr_three_point_tensor import (
    hjs_to_polynomial_ground_tensor,
)
from spin23_genus1_blocks import (
    direct_ns_torus_necklace_series,
    direct_ramond_torus_necklace_series,
)
from spin23_genus1_recursion import (
    NSTorusTwoPointHRecursion,
    ns_liouville_weight,
    super_liouville_central_charge,
)
from spin23_genus1_recursive_blocks import (
    G_MINUS_HALF,
    recursive_ns_torus_two_point_series,
    recursive_ramond_torus_two_point_series,
)
from spin23_ns_torus_two_virasoro import (
    generic_ns_torus_two_point_necklace_coefficients,
)
from spin23_super_liouville_data import (
    ns_weight,
    ramond_liouville_weight,
)


class GenusOneRecursionTests(unittest.TestCase):
    """Compare every enabled recursive component with the direct oracle."""

    def test_generic_ns_primary_recursion_matches_direct_through_level_two(
        self,
    ) -> None:
        b = 1.17
        internal_momenta = (0.37, 0.52)
        external_momenta = (0.41, 0.63)
        central_charge = super_liouville_central_charge(b)
        internal_weights = tuple(
            ns_liouville_weight(momentum, b)
            for momentum in internal_momenta
        )
        external_weights = tuple(
            ns_liouville_weight(momentum, b)
            for momentum in external_momenta
        )
        for form_parity in (0, 1):
            recursive = NSTorusTwoPointHRecursion(
                b=b,
                internal_weight_1=internal_weights[0],
                internal_weight_2=internal_weights[1],
                external_weight_1=external_weights[0],
                external_weight_2=external_weights[1],
                form_parity=form_parity,
            ).coefficients(4, 4)
            form = (1.0, 0.0) if form_parity == 0 else (0.0, 1.0)
            direct = direct_ns_torus_necklace_series(
                c=central_charge,
                internal_weights=internal_weights,
                external_weights=external_weights,
                maximum_twice_levels=4,
                form_weights=(form, form),
            )
            for levels, expected in direct.coefficients.items():
                with self.subTest(form_parity=form_parity, levels=levels):
                    self.assertAlmostEqual(recursive[levels], expected)

    def test_self_dual_ns_adapter_preserves_spin_lifts(self) -> None:
        internal_momenta = (0.37, 0.52)
        external_momenta = (0.41, 0.63)
        lifts = (-1, 1)
        recursive = recursive_ns_torus_two_point_series(
            internal_momenta=internal_momenta,
            external_ns_momenta=external_momenta,
            maximum_twice_levels=4,
            edge_lift_signs=lifts,
            form_parity=0,
            radius=0.035,
            check_radius=0.05,
            samples=8,
        )
        direct = direct_ns_torus_necklace_series(
            c=13.5,
            internal_weights=tuple(
                ns_weight(momentum) for momentum in internal_momenta
            ),
            external_weights=tuple(
                ns_weight(momentum) for momentum in external_momenta
            ),
            maximum_twice_levels=4,
            edge_lift_signs=lifts,
            form_weights=((1.0, 0.0), (1.0, 0.0)),
        )
        for levels, expected in direct.coefficients.items():
            with self.subTest(levels=levels):
                self.assertAlmostEqual(
                    recursive.series.coefficients[levels],
                    expected,
                    places=8,
                )
        self.assertLess(recursive.maximum_finite_part_absolute_error, 1.0e-8)

    def test_generic_ns_two_virasoro_pp_and_gg_match_direct(self) -> None:
        b = 1.17
        internal_momenta = (0.37, 0.52)
        external_momentum = 0.41
        lifts = (-1, 1)
        recursive = generic_ns_torus_two_point_necklace_coefficients(
            b=b,
            previous_internal_momentum=internal_momenta[1],
            current_internal_momentum=internal_momenta[0],
            external_ns_momentum=external_momentum,
            previous_lift_sign=lifts[1],
            current_lift_sign=lifts[0],
            maximum_previous_twice_level=2,
            maximum_current_twice_level=2,
        )
        for form_parity in (0, 1):
            form = (1.0, 0.0) if form_parity == 0 else (0.0, 1.0)
            components = (
                ((), recursive.primary_components[form_parity]),
                (
                    G_MINUS_HALF,
                    recursive.superdescendant_components[form_parity],
                ),
            )
            for word, coefficients in components:
                direct = direct_ns_torus_necklace_series(
                    c=super_liouville_central_charge(b),
                    internal_weights=tuple(
                        ns_liouville_weight(momentum, b)
                        for momentum in internal_momenta
                    ),
                    external_weights=(
                        ns_liouville_weight(external_momentum, b),
                    )
                    * 2,
                    maximum_twice_levels=2,
                    edge_lift_signs=lifts,
                    external_words=(word, word),
                    form_weights=(form, form),
                    vertex_backend="template",
                )
                for (previous, current), expected in coefficients.items():
                    levels = (current, previous)
                    with self.subTest(
                        form_parity=form_parity,
                        word=word,
                        levels=levels,
                    ):
                        self.assertAlmostEqual(
                            direct.coefficients[levels],
                            expected,
                            places=10,
                        )

    def test_self_dual_ns_gg_adapter_matches_direct(self) -> None:
        internal_momenta = (0.37, 0.52)
        external_momentum = 0.41
        recursive = recursive_ns_torus_two_point_series(
            internal_momenta=internal_momenta,
            external_ns_momenta=(external_momentum,) * 2,
            maximum_twice_levels=2,
            form_parity=1,
            external_words=(G_MINUS_HALF, G_MINUS_HALF),
            radius=0.035,
            check_radius=0.05,
            samples=8,
        )
        direct = direct_ns_torus_necklace_series(
            c=13.5,
            internal_weights=tuple(
                ns_weight(momentum) for momentum in internal_momenta
            ),
            external_weights=(ns_weight(external_momentum),) * 2,
            maximum_twice_levels=2,
            external_words=(G_MINUS_HALF, G_MINUS_HALF),
            form_weights=((0.0, 1.0),) * 2,
            vertex_backend="template",
        )
        for levels, expected in direct.coefficients.items():
            with self.subTest(levels=levels):
                self.assertAlmostEqual(
                    recursive.series.coefficients[levels],
                    expected,
                    places=10,
                )
        self.assertLess(recursive.scaled_finite_part_error, 1.0e-10)

    def test_ns_adapter_rejects_unsupported_descendant_components(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "PP or GG"):
            recursive_ns_torus_two_point_series(
                internal_momenta=(0.37, 0.52),
                external_ns_momenta=(0.41, 0.41),
                maximum_twice_levels=2,
                form_parity=0,
                external_words=(G_MINUS_HALF, ()),
                samples=8,
            )
        with self.assertRaisesRegex(NotImplementedError, "equal"):
            recursive_ns_torus_two_point_series(
                internal_momenta=(0.37, 0.52),
                external_ns_momenta=(0.41, 0.63),
                maximum_twice_levels=2,
                form_parity=0,
                external_words=(G_MINUS_HALF, G_MINUS_HALF),
                samples=8,
            )

    def test_self_dual_ramond_adapter_uses_direct_edge_order(self) -> None:
        internal_momenta = (0.52, 0.37)
        external_momentum = 0.41
        signs = (-1, -1)
        beta = lambda momentum: 1j * momentum / math.sqrt(2.0)
        tensors = (
            hjs_to_polynomial_ground_tensor(
                beta(internal_momenta[1]),
                beta(internal_momenta[0]),
                structure_sign=signs[0],
            ),
            hjs_to_polynomial_ground_tensor(
                beta(internal_momenta[0]),
                beta(internal_momenta[1]),
                structure_sign=signs[1],
            ),
        )
        recursive = recursive_ramond_torus_two_point_series(
            internal_momenta=internal_momenta,
            external_ns_momenta=(external_momentum, external_momentum),
            structure_signs=signs,
            maximum_twice_levels=0,
            radius=0.035,
            check_radius=0.05,
            samples=8,
        )
        direct = direct_ramond_torus_necklace_series(
            c=13.5,
            internal_weights=tuple(
                ramond_liouville_weight(momentum)
                for momentum in internal_momenta
            ),
            external_weights=(ns_weight(external_momentum),) * 2,
            ground_tensors=tensors,
            maximum_twice_levels=0,
        )
        self.assertAlmostEqual(
            recursive.series.leading_coefficient,
            direct.leading_coefficient,
        )
        self.assertLess(recursive.maximum_finite_part_absolute_error, 1.0e-9)


if __name__ == "__main__":
    unittest.main()
