"""Regression tests for the multipoint NS central-charge recursion."""

import unittest

import mpmath
import numpy as np

from ns_genus12_finite_c_check import (
    NSDescendantThreeForm,
    NumericNSVermaModule,
)
from ns_multipoint_c_recursion import (
    NSSphereLinearCRecursion,
    NSTorusNecklaceCRecursion,
    ns_non_global_vacuum_coefficients,
)
from superconformal_blocks import HighPrecisionNSSphereFourPointBlock


class MultipointNSCRecursionTests(unittest.TestCase):
    def test_sphere_driver_reduces_to_tested_four_point_recursion(self):
        central_charge = 14.19870372000744
        external_weights = (0.37, 0.61, 0.48, 0.29)
        internal_weight = 0.73
        reference = HighPrecisionNSSphereFourPointBlock(
            c=central_charge,
            h1=external_weights[0],
            h2=external_weights[1],
            h3=external_weights[2],
            h4=external_weights[3],
            internal_weight=internal_weight,
            working_precision=70,
        )
        even = NSSphereLinearCRecursion(
            central_charge=central_charge,
            external_weights=external_weights,
            internal_weights=(internal_weight,),
            vertex_sectors=(0, 0),
            working_precision=70,
        )
        odd = NSSphereLinearCRecursion(
            central_charge=central_charge,
            external_weights=external_weights,
            internal_weights=(internal_weight,),
            vertex_sectors=(1, 1),
            working_precision=70,
        )

        for twice_level in range(9):
            block = even if twice_level % 2 == 0 else odd
            with self.subTest(twice_level=twice_level):
                self.assertLess(
                    abs(
                        block.coefficient((twice_level,))
                        - reference.coefficient(twice_level)
                    ),
                    5.0e-14,
                )

    def test_sphere_external_components_reduce_to_all_four_starred_blocks(self):
        central_charge = 14.19870372000744
        external_weights = (0.37, 0.61, 0.48, 0.29)
        internal_weight = 0.73
        for star2 in (False, True):
            for star3 in (False, True):
                reference = HighPrecisionNSSphereFourPointBlock(
                    c=central_charge,
                    h1=external_weights[0],
                    h2=external_weights[1],
                    h3=external_weights[2],
                    h4=external_weights[3],
                    internal_weight=internal_weight,
                    star2=star2,
                    star3=star3,
                    working_precision=70,
                )
                descendants = (0, int(star2), int(star3), 0)
                for twice_level in range(9):
                    edge_parity = twice_level % 2
                    sectors = (
                        int(star2) ^ edge_parity,
                        int(star3) ^ edge_parity,
                    )
                    block = NSSphereLinearCRecursion(
                        central_charge=central_charge,
                        external_weights=external_weights,
                        external_descendants=descendants,
                        internal_weights=(internal_weight,),
                        vertex_sectors=sectors,
                        working_precision=70,
                    )
                    with self.subTest(
                        star2=star2,
                        star3=star3,
                        twice_level=twice_level,
                    ):
                        self.assertLess(
                            abs(
                                block.coefficient((twice_level,))
                                - reference.coefficient(twice_level)
                            ),
                            5.0e-14,
                        )

    def test_explicit_plumbing_log_selects_conjugate_ns_lift(self):
        """A negative-real tube has opposite holomorphic square roots."""

        block = NSSphereLinearCRecursion(
            central_charge=14.19870372000744,
            external_weights=(0.37, 0.61, 0.48, 0.29),
            internal_weights=(0.73,),
            vertex_sectors=(1, 1),
            working_precision=70,
        )
        q_value = mpmath.mpc("-0.2")
        holomorphic_log = mpmath.log(q_value)
        holomorphic = block.series_value(
            (q_value,),
            (1,),
            q_log_values=(holomorphic_log,),
        )
        antiholomorphic = block.series_value(
            (q_value,),
            (1,),
            q_log_values=(mpmath.conj(holomorphic_log),),
        )
        self.assertLess(abs(antiholomorphic - mpmath.conj(holomorphic)), 1.0e-15)
        self.assertGreater(abs(antiholomorphic - holomorphic), 1.0e-3)

    def test_minimum_edge_level_evaluates_descendant_remainder_directly(self):
        block = NSSphereLinearCRecursion(
            central_charge=14.19870372000744,
            external_weights=(0.31, 0.42, 0.53, 0.47, 0.28),
            internal_weights=(0.73, 0.81),
            vertex_sectors=(0, 0, 0),
            working_precision=70,
        )
        q_values = (0.07 + 0.02j, 0.19 - 0.03j)
        full = block.series_value(q_values, (4, 4), max_total_twice_level=6)
        leading = block.series_value(q_values, (0, 4), max_total_twice_level=6)
        remainder = block.series_value(
            q_values,
            (4, 4),
            max_total_twice_level=6,
            minimum_twice_levels=(2, 0),
        )
        self.assertLess(abs(full - leading - remainder), 1.0e-15)

        def recursive_leading(q1):
            total = block.recursive_series_value(
                (q1, q_values[1]), 2, (4, 4),
                global_max_total_twice_level=6,
            )
            descendant = block.recursive_series_value(
                (q1, q_values[1]), 2, (4, 4),
                global_max_total_twice_level=6,
                minimum_twice_levels=(2, 0),
            )
            return total - descendant

        self.assertLess(
            abs(recursive_leading(q_values[0]) - recursive_leading(0.11 - 0.01j)),
            1.0e-15,
        )
        selected_leading = block.recursive_series_value(
            q_values,
            2,
            (4, 4),
            global_max_total_twice_level=6,
            maximum_accumulated_twice_levels=(0, None),
        )
        self.assertLess(
            abs(selected_leading - recursive_leading(q_values[0])), 1.0e-15
        )

    def test_global_seed_preserves_multiprecision_weights(self):
        with mpmath.workdps(90):
            perturbation = mpmath.mpf("1e-50")
            common = dict(
                central_charge=mpmath.mpf("14.2"),
                external_weights=(
                    mpmath.mpf("0.31"),
                    mpmath.mpf("0.42"),
                    mpmath.mpf("0.53"),
                    mpmath.mpf("0.47"),
                    mpmath.mpf("0.28"),
                ),
                vertex_sectors=(0, 0, 0),
                working_precision=90,
            )
            unperturbed = NSSphereLinearCRecursion(
                internal_weights=(mpmath.mpf("0.73"), mpmath.mpf("0.81")),
                **common,
            ).global_coefficient((10, 8))
            perturbed = NSSphereLinearCRecursion(
                internal_weights=(
                    mpmath.mpf("0.73") + perturbation,
                    mpmath.mpf("0.81"),
                ),
                **common,
            ).global_coefficient((10, 8))
            self.assertNotEqual(unperturbed, perturbed)

    def test_sphere_five_point_matches_independent_finite_c_sewing(self):
        central_charge = 14.19870372000744
        external_weights = (0.31, 0.42, 0.53, 0.47, 0.28)
        internal_weights = (0.73, 0.81)
        modules = tuple(
            NumericNSVermaModule(c=central_charge, weight=weight)
            for weight in internal_weights
        )
        vertices = (
            NSDescendantThreeForm(
                c=central_charge,
                bra_weight=internal_weights[0],
                middle_weight=external_weights[1],
                ket_weight=external_weights[0],
            ),
            NSDescendantThreeForm(
                c=central_charge,
                bra_weight=internal_weights[1],
                middle_weight=external_weights[2],
                ket_weight=internal_weights[0],
            ),
            NSDescendantThreeForm(
                c=central_charge,
                bra_weight=external_weights[4],
                middle_weight=external_weights[3],
                ket_weight=internal_weights[1],
            ),
        )

        def direct_coefficient(twice_levels):
            bases = tuple(
                modules[edge].basis(level)
                for edge, level in enumerate(twice_levels)
            )
            inverses = tuple(
                modules[edge].numeric_inverse_gram(level)
                for edge, level in enumerate(twice_levels)
            )
            left = np.asarray(
                [vertices[0].value(state, (), ()) for state in bases[0]],
                dtype=np.complex128,
            )
            middle = np.asarray(
                [
                    [vertices[1].value(upper, (), lower) for lower in bases[0]]
                    for upper in bases[1]
                ],
                dtype=np.complex128,
            )
            right = np.asarray(
                [vertices[2].value((), (), state) for state in bases[1]],
                dtype=np.complex128,
            )
            return np.einsum(
                "a,ab,cb,cd,d->",
                left,
                inverses[0],
                middle,
                inverses[1],
                right,
                optimize=True,
            )

        # These levels cross the odd (3,1) and even (2,2) c-poles on one or
        # both edges.  Their vertex sectors are fixed by adjacent edge parity.
        for levels in ((3, 0), (4, 0), (3, 1), (3, 3), (4, 2), (4, 4)):
            epsilon_1, epsilon_2 = (level % 2 for level in levels)
            sectors = (
                epsilon_1,
                epsilon_1 ^ epsilon_2,
                epsilon_2,
            )
            recursive = NSSphereLinearCRecursion(
                central_charge=central_charge,
                external_weights=external_weights,
                internal_weights=internal_weights,
                vertex_sectors=sectors,
                working_precision=70,
            )
            with self.subTest(levels=levels, sectors=sectors):
                self.assertLess(
                    abs(direct_coefficient(levels) - recursive.coefficient(levels)),
                    2.0e-12,
                )

    def test_necklace_global_seed_matches_known_leading_coefficients(self):
        h1, h2 = 0.73, 0.91
        d1, d2 = 0.37, 0.52
        block = NSTorusNecklaceCRecursion(
            central_charge=14.2,
            external_weights=(d1, d2),
            internal_weights=(h1, h2),
        )
        expected_10 = (h1 + d1 - h2) * (h1 + d2 - h2) / (2 * h1)
        expected_01 = (h2 + d1 - h1) * (h2 + d2 - h1) / (2 * h2)
        expected_half_half = (
            (h1 + h2 - d1) * (h1 + h2 - d2) / (4 * h1 * h2)
        )
        self.assertLess(abs(block.global_coefficient((2, 0)) - expected_10), 1e-14)
        self.assertLess(abs(block.global_coefficient((0, 2)) - expected_01), 1e-14)
        self.assertLess(
            abs(block.global_coefficient((1, 1)) - expected_half_half),
            1e-14,
        )
        self.assertEqual(block.global_coefficient((1, 0)), 0)
        self.assertEqual(block.global_coefficient((0, 1)), 0)

    def test_necklace_regular_seed_has_diagonal_vacuum_factor(self):
        self.assertEqual(
            ns_non_global_vacuum_coefficients(6, 1),
            (1, 0, 0, 1, 1, 1, 1),
        )
        self.assertEqual(
            ns_non_global_vacuum_coefficients(6, -1),
            (1, 0, 0, -1, 1, -1, 1),
        )
        plus = NSTorusNecklaceCRecursion(
            central_charge=14.2,
            external_weights=(0.37, 0.52),
            internal_weights=(0.73, 0.91),
            spin_lift=1,
        )
        minus = NSTorusNecklaceCRecursion(
            central_charge=14.2,
            external_weights=(0.37, 0.52),
            internal_weights=(0.73, 0.91),
            spin_lift=-1,
        )
        self.assertLess(
            abs(
                plus.regular_coefficient((3, 3))
                - plus.global_coefficient((3, 3))
                - 1
            ),
            1e-14,
        )
        self.assertLess(
            abs(
                minus.regular_coefficient((3, 3))
                - minus.global_coefficient((3, 3))
                + 1
            ),
            1e-14,
        )

    def test_two_point_necklace_matches_independent_finite_c_sewing(self):
        """Check the first odd/even poles without residue ingredients."""

        central_charge = 14.19870372000744
        internal_weights = (0.73, 0.91)
        external_weights = (0.37, 0.52)
        modules = tuple(
            NumericNSVermaModule(c=central_charge, weight=weight)
            for weight in internal_weights
        )
        forms = (
            NSDescendantThreeForm(
                c=central_charge,
                bra_weight=internal_weights[0],
                middle_weight=external_weights[0],
                ket_weight=internal_weights[1],
            ),
            NSDescendantThreeForm(
                c=central_charge,
                bra_weight=internal_weights[1],
                middle_weight=external_weights[1],
                ket_weight=internal_weights[0],
            ),
        )
        recursive = NSTorusNecklaceCRecursion(
            central_charge=central_charge,
            external_weights=external_weights,
            internal_weights=internal_weights,
            working_precision=70,
        )

        def direct_coefficient(twice_levels):
            level_1, level_2 = twice_levels
            bases = (
                modules[0].basis(level_1),
                modules[1].basis(level_2),
            )
            inverses = (
                modules[0].numeric_inverse_gram(level_1),
                modules[1].numeric_inverse_gram(level_2),
            )
            vertex_1 = np.asarray(
                [
                    [forms[0].value(state_1, (), state_2) for state_2 in bases[1]]
                    for state_1 in bases[0]
                ],
                dtype=np.complex128,
            )
            vertex_2 = np.asarray(
                [
                    [forms[1].value(state_2, (), state_1) for state_1 in bases[0]]
                    for state_2 in bases[1]
                ],
                dtype=np.complex128,
            )
            return np.einsum(
                "ac,ab,db,dc->",
                vertex_1,
                inverses[0],
                vertex_2,
                inverses[1],
                optimize=True,
            )

        # (3,1) and (3,3) see the odd (3,1) c-pole; (4,0) and
        # (4,4) additionally see the first even (2,2) c-pole.
        for levels in ((3, 1), (1, 3), (3, 3), (4, 0), (0, 4), (4, 4)):
            with self.subTest(levels=levels):
                self.assertLess(
                    abs(direct_coefficient(levels) - recursive.coefficient(levels)),
                    2.0e-12,
                )

    def test_three_point_necklace_is_cyclic(self):
        central_charge = 14.31
        external_weights = (0.27, 0.36, 0.44)
        internal_weights = (0.71, 0.83, 0.92)
        levels = (3, 1, 1)
        block = NSTorusNecklaceCRecursion(
            central_charge=central_charge,
            external_weights=external_weights,
            internal_weights=internal_weights,
        )
        rotated = NSTorusNecklaceCRecursion(
            central_charge=central_charge,
            external_weights=external_weights[1:] + external_weights[:1],
            internal_weights=internal_weights[1:] + internal_weights[:1],
        )
        self.assertLess(
            abs(block.coefficient(levels) - rotated.coefficient(levels[1:] + levels[:1])),
            5.0e-13,
        )

    def test_three_point_necklace_matches_finite_c_sewing(self):
        central_charge = 14.19870372000744
        internal_weights = (0.73, 0.91, 0.84)
        external_weights = (0.37, 0.52, 0.41)
        modules = tuple(
            NumericNSVermaModule(c=central_charge, weight=weight)
            for weight in internal_weights
        )
        vertices = tuple(
            NSDescendantThreeForm(
                c=central_charge,
                bra_weight=internal_weights[index],
                middle_weight=external_weights[index],
                ket_weight=internal_weights[(index + 1) % 3],
            )
            for index in range(3)
        )

        def direct_coefficient(levels):
            bases = tuple(
                modules[edge].basis(level) for edge, level in enumerate(levels)
            )
            inverses = tuple(
                modules[edge].numeric_inverse_gram(level)
                for edge, level in enumerate(levels)
            )
            tensors = tuple(
                np.asarray(
                    [
                        [
                            vertices[index].value(bra, (), ket)
                            for ket in bases[(index + 1) % 3]
                        ]
                        for bra in bases[index]
                    ],
                    dtype=np.complex128,
                )
                for index in range(3)
            )
            return np.einsum(
                "ab,cd,ef,af,cb,ed->",
                tensors[0],
                tensors[1],
                tensors[2],
                inverses[0],
                inverses[1],
                inverses[2],
                optimize=True,
            )

        samples = (
            ((3, 1, 1), (0, 0, 0)),
            ((4, 0, 0), (0, 0, 0)),
            ((3, 3, 3), (0, 0, 0)),
            ((3, 0, 1), (1, 1, 0)),
        )
        for levels, sectors in samples:
            recursive = NSTorusNecklaceCRecursion(
                central_charge=central_charge,
                external_weights=external_weights,
                internal_weights=internal_weights,
                vertex_sectors=sectors,
                working_precision=70,
            )
            with self.subTest(levels=levels, sectors=sectors):
                self.assertLess(
                    abs(direct_coefficient(levels) - recursive.coefficient(levels)),
                    2.0e-12,
                )

    def test_bottom_external_sector_parity_is_validated(self):
        with self.assertRaises(ValueError):
            NSSphereLinearCRecursion(
                central_charge=14.2,
                external_weights=(0.1, 0.2, 0.3, 0.4, 0.5),
                internal_weights=(0.7, 0.8),
                vertex_sectors=(1, 0, 0),
            )
        with self.assertRaises(ValueError):
            NSTorusNecklaceCRecursion(
                central_charge=14.2,
                external_weights=(0.2, 0.3),
                internal_weights=(0.7, 0.8),
                vertex_sectors=(0.5, 0),
            )

    def test_functional_recursion_starts_from_the_global_sphere_block(self):
        block = NSSphereLinearCRecursion(
            central_charge=14.2,
            external_weights=(0.31, 0.42, 0.53, 0.47, 0.28),
            internal_weights=(0.73, 0.81),
            vertex_sectors=(0, 0, 0),
            working_precision=70,
        )
        q_values = (0.17, -0.08)
        maxima = (6, 4)
        expected = 0.0j
        for levels, coefficient in block.coefficient_table(maxima).items():
            # No NS c-pole occurs below twice-level three, so the functional
            # recursion at budget two is exactly the global seed.  Use the
            # public global coefficient rather than the finite-c table.
            expected += complex(block.global_coefficient(levels)) * (
                q_values[0] ** (levels[0] / 2)
                * q_values[1] ** (levels[1] / 2)
            )
        actual = block.recursive_series_value(q_values, 2, maxima)
        self.assertLess(abs(expected - actual), 2.0e-14)


if __name__ == "__main__":
    unittest.main()
