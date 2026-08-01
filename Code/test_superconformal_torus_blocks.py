"""Low-level NS/R sewing tests for the genus-one block layer."""

import unittest

from superconformal_blocks import central_charge
from superconformal_torus_blocks import (
    NSPlumbingParameter,
    NSTorusOnePointBlock,
    RamondGroundFiber,
    RamondPlumbingParameter,
    RamondTorusOnePointBlock,
    SelfDualNSTorusOnePointBlock,
    SelfDualRamondTorusOnePointBlock,
    TorusTwoPointSpinStructure,
    ns_verma_character_coefficients,
    ramond_positive_character_coefficients,
)
from superconformal_torus_descendants import (
    BruteForceNSTorusOnePointBlock,
    BruteForceRamondTorusOnePointBlock,
)


class SuperconformalTorusBlockTests(unittest.TestCase):
    def test_two_point_spin_ledger_classifies_equal_edge_sectors(self):
        ns = TorusTwoPointSpinStructure("NS", "NS")
        ns_tilde = TorusTwoPointSpinStructure(
            "NS", "NS", ns_lift_sign_2=-1
        )
        ramond = TorusTwoPointSpinStructure("R", "R")
        ramond_tilde = TorusTwoPointSpinStructure(
            "R", "R", r_cycle_insertion_1="parity"
        )
        self.assertEqual(ns.external_sectors, ("NS", "NS"))
        self.assertEqual(ns.spin_label, "NS")
        self.assertEqual(ns_tilde.spin_label, "NS_tilde")
        self.assertEqual(ramond.external_sectors, ("NS", "NS"))
        self.assertEqual(ramond.spin_label, "R")
        self.assertEqual(ramond_tilde.spin_label, "R_tilde")

    def test_two_point_spin_ledger_classifies_mixed_edges(self):
        assignments = {
            TorusTwoPointSpinStructure(
                "NS",
                "R",
                ns_lift_sign_1=lift,
                r_cycle_insertion_2=insertion,
            ).mixed_spin_bits
            for lift in (-1, 1)
            for insertion in ("identity", "parity")
        }
        self.assertEqual(
            assignments,
            {(-1, -1), (-1, 1), (1, -1), (1, 1)},
        )
        block = TorusTwoPointSpinStructure("R", "NS")
        self.assertEqual(block.external_sectors, ("R", "R"))
        self.assertEqual(block.spin_label, "mixed")

    def test_two_point_spin_ledger_builds_typed_plumbing(self):
        assignment = TorusTwoPointSpinStructure(
            "NS",
            "R",
            ns_lift_sign_1=-1,
            r_cycle_insertion_2="parity",
        )
        plumbing_1, plumbing_2 = assignment.plumbing_parameters(
            0.07, 0.05
        )
        self.assertIsInstance(plumbing_1, NSPlumbingParameter)
        self.assertIsInstance(plumbing_2, RamondPlumbingParameter)
        self.assertEqual(plumbing_1.lift_sign, -1)
        self.assertEqual(plumbing_2.cycle_insertion, "parity")

    def test_two_point_spin_ledger_rejects_edge_mismatches(self):
        with self.assertRaisesRegex(ValueError, "edge 1 is NS"):
            TorusTwoPointSpinStructure(
                "NS", "NS", r_cycle_insertion_1="parity"
            )
        with self.assertRaisesRegex(ValueError, "edge 2 is R"):
            TorusTwoPointSpinStructure(
                "R", "R", ns_lift_sign_2=-1
            )

    def test_ns_character_and_spin_lift(self):
        ordinary = ns_verma_character_coefficients(6, lift_sign=1)
        twisted = ns_verma_character_coefficients(6, lift_sign=-1)
        self.assertEqual(
            tuple(int(ordinary[level].real) for level in range(7)),
            (1, 1, 1, 2, 3, 4, 5),
        )
        for twice_level in range(7):
            self.assertEqual(
                twisted[twice_level],
                (-1) ** twice_level * ordinary[twice_level],
            )

    def test_ns_first_elliptic_coefficients(self):
        internal_weight = 0.73
        external_weight = 0.29
        block = NSTorusOnePointBlock(
            b=1.27,
            internal_weight=internal_weight,
            external_weight=external_weight,
        )
        coefficients = block.elliptic_coefficients(2)
        self.assertAlmostEqual(coefficients[0], 1.0, places=14)
        self.assertAlmostEqual(
            coefficients[1],
            -external_weight / (2.0 * internal_weight),
            places=13,
        )
        self.assertAlmostEqual(
            coefficients[2],
            external_weight**2 / (2.0 * internal_weight),
            places=13,
        )
        q = 0.04
        plus = block.evaluate(NSPlumbingParameter(q, 1), 2)
        minus = block.evaluate(NSPlumbingParameter(q, -1), 2)
        expected_difference = (
            2.0 * coefficients[1] * q**0.5
        )
        self.assertAlmostEqual(plus - minus, expected_difference, places=13)
        raw = block.raw_coefficients(2)
        self.assertAlmostEqual(
            raw[1],
            1.0 - external_weight / (2.0 * internal_weight),
            places=13,
        )
        plumbing = NSPlumbingParameter(q, 1)
        self.assertAlmostEqual(
            block.chiral_block(plumbing, 2),
            q ** (internal_weight - block.c / 24.0)
            * sum(raw[level] * q ** (level / 2.0) for level in raw),
            places=13,
        )

    def test_ramond_ground_fiber_keeps_g0_matrix_explicit(self):
        fiber = RamondGroundFiber(c=13.1, weight=1.2)
        kappa_squared = fiber.kappa_squared
        g0_squared = (
            fiber.g0[0][1] * fiber.g0[1][0]
        )
        self.assertAlmostEqual(g0_squared, kappa_squared, places=14)

        plus_vertex = fiber.even_vertex(1)
        minus_vertex = fiber.even_vertex(-1)
        self.assertAlmostEqual(
            fiber.contract(plus_vertex, "identity"), 2.0, places=14
        )
        self.assertAlmostEqual(
            fiber.contract(plus_vertex, "parity"), 0.0, places=14
        )
        self.assertAlmostEqual(
            fiber.contract(minus_vertex, "identity"), 0.0, places=14
        )
        self.assertAlmostEqual(
            fiber.contract(minus_vertex, "parity"), 2.0, places=14
        )
        self.assertAlmostEqual(
            fiber.contract(plus_vertex, "g0"), 0.0, places=14
        )

    def test_ramond_positive_character(self):
        self.assertEqual(
            ramond_positive_character_coefficients(6),
            (1, 2, 4, 8, 14, 24, 40),
        )

    def test_ramond_level_one_matches_hjs_appendix(self):
        b = 1.31
        beta = 0.37j
        external_weight = 0.28
        delta = central_charge(b) / 24.0 - beta**2
        denominator = (
            (3.0 + 6.0 * b**2 + 16.0 * delta)
            * (6.0 + b**2 * (3.0 + 16.0 * delta))
        )

        expected_plus = (
            6.0
            * (2.0 + 5.0 * b**2 + 2.0 * b**4)
            * (
                3.0
                + 4.0
                * (external_weight - 1.0)
                * external_weight
            )
            + 64.0
            * (
                3.0
                + 3.0 * b**4
                + b**2
                * (
                    3.0
                    - 6.0 * external_weight
                    + 2.0 * external_weight**2
                )
            )
            * delta
            + 512.0 * b**2 * delta**2
        ) / denominator
        expected_minus = (
            32.0
            * (
                4.0
                * (external_weight - 2.0)
                * external_weight
                + 3.0
            )
            * delta
            * b**2
            + 6.0
            * (2.0 * b**4 + 5.0 * b**2 + 2.0)
            * (4.0 * external_weight**2 - 1.0)
        ) / denominator

        plus = RamondTorusOnePointBlock(
            b=b,
            internal_beta=beta,
            external_weight=external_weight,
            sign=1,
        )
        minus = RamondTorusOnePointBlock(
            b=b,
            internal_beta=beta,
            external_weight=external_weight,
            sign=-1,
        )
        self.assertAlmostEqual(
            plus.raw_even_coefficients(1)[1],
            expected_plus,
            places=12,
        )
        self.assertAlmostEqual(
            minus.raw_even_coefficients(1)[1],
            expected_minus,
            places=12,
        )

    def test_ns_leading_recursion_matches_direct_descendant_sewing(self):
        internal_weight = 0.73
        external_weight = 0.29
        direct = BruteForceNSTorusOnePointBlock(
            internal_weight=internal_weight,
            external_weight=external_weight,
        )
        recursive = NSTorusOnePointBlock(
            b=1.27,
            internal_weight=internal_weight,
            external_weight=external_weight,
        )
        direct_coefficients = direct.elliptic_coefficients()
        recursive_coefficients = recursive.elliptic_coefficients(2)
        for twice_level in (0, 1, 2):
            self.assertAlmostEqual(
                direct_coefficients[twice_level],
                recursive_coefficients[twice_level],
                places=13,
            )

        self.assertEqual(
            direct.gram_matrices()[1],
            ((2.0 * internal_weight + 0.0j,),),
        )
        self.assertEqual(
            direct.vertex_matrices()[1],
            ((2.0 * internal_weight - external_weight + 0.0j,),),
        )

    def test_ramond_level_one_recursion_matches_direct_matrix_sewing(self):
        b = 1.31
        beta = 0.37j
        external_weight = 0.28
        c = central_charge(b)
        internal_weight = c / 24.0 - beta**2
        for sign in (1, -1):
            direct = BruteForceRamondTorusOnePointBlock(
                central_charge=c,
                internal_weight=internal_weight,
                external_weight=external_weight,
                sign=sign,
            )
            recursive = RamondTorusOnePointBlock(
                b=b,
                internal_beta=beta,
                external_weight=external_weight,
                sign=sign,
            )
            for direct_value, recursive_value in zip(
                direct.raw_even_coefficients(),
                recursive.raw_even_coefficients(1),
            ):
                self.assertAlmostEqual(
                    direct_value, recursive_value, places=12
                )

    def test_exact_type0b_ns_and_r_coefficients(self):
        internal_ns_momentum = 0.61
        external_momentum = 0.33
        ns_block = SelfDualNSTorusOnePointBlock(
            internal_momentum=internal_ns_momentum,
            external_momentum=external_momentum,
            samples=16,
        )
        ns_coefficients = ns_block.elliptic_coefficients(2)
        h_internal_ns = 0.5 + internal_ns_momentum**2 / 2.0
        h_external = 0.5 + external_momentum**2 / 2.0
        self.assertAlmostEqual(
            ns_coefficients[1],
            -h_external / (2.0 * h_internal_ns),
            places=12,
        )
        self.assertAlmostEqual(
            ns_coefficients[2],
            h_external**2 / (2.0 * h_internal_ns),
            places=12,
        )
        direct_ns = BruteForceNSTorusOnePointBlock(
            internal_weight=h_internal_ns,
            external_weight=h_external,
        ).elliptic_coefficients()
        for twice_level in (0, 1, 2):
            self.assertAlmostEqual(
                ns_coefficients[twice_level],
                direct_ns[twice_level],
                places=12,
            )

        internal_r_momentum = 0.60
        beta = internal_r_momentum * 1j / 2.0**0.5
        b = 1.0
        delta = central_charge(b) / 24.0 - beta**2
        denominator = (
            (3.0 + 6.0 * b**2 + 16.0 * delta)
            * (6.0 + b**2 * (3.0 + 16.0 * delta))
        )
        expected_plus = (
            6.0
            * (2.0 + 5.0 * b**2 + 2.0 * b**4)
            * (
                3.0
                + 4.0 * (h_external - 1.0) * h_external
            )
            + 64.0
            * (
                3.0
                + 3.0 * b**4
                + b**2
                * (
                    3.0
                    - 6.0 * h_external
                    + 2.0 * h_external**2
                )
            )
            * delta
            + 512.0 * b**2 * delta**2
        ) / denominator
        expected_minus = (
            32.0
            * (4.0 * (h_external - 2.0) * h_external + 3.0)
            * delta
            * b**2
            + 6.0
            * (2.0 * b**4 + 5.0 * b**2 + 2.0)
            * (4.0 * h_external**2 - 1.0)
        ) / denominator

        plus = SelfDualRamondTorusOnePointBlock(
            internal_momentum=internal_r_momentum,
            external_momentum=external_momentum,
            sign=1,
            samples=16,
        )
        minus = SelfDualRamondTorusOnePointBlock(
            internal_momentum=internal_r_momentum,
            external_momentum=external_momentum,
            sign=-1,
            samples=16,
        )
        self.assertAlmostEqual(
            plus.raw_even_coefficients(1)[1],
            expected_plus,
            places=11,
        )
        self.assertAlmostEqual(
            minus.raw_even_coefficients(1)[1],
            expected_minus,
            places=11,
        )
        for recursive in (plus, minus):
            direct_r = BruteForceRamondTorusOnePointBlock(
                central_charge=central_charge(1.0),
                internal_weight=delta,
                external_weight=h_external,
                sign=recursive.sign,
            )
            for direct_value, recursive_value in zip(
                direct_r.raw_even_coefficients(),
                recursive.raw_even_coefficients(1),
            ):
                self.assertAlmostEqual(
                    direct_value, recursive_value, places=11
                )
        self.assertLess(
            plus.coefficient_diagnostics(1)[1].relative_error,
            1.0e-11,
        )
        self.assertLess(
            minus.coefficient_diagnostics(1)[1].relative_error,
            1.0e-11,
        )
        plus.elliptic_coefficients(3)
        self.assertLess(
            max(
                diagnostic.relative_error
                for diagnostic in plus.coefficient_diagnostics(3).values()
            ),
            2.0e-12,
        )
        self.assertAlmostEqual(
            plus.cycle_projected_raw_coefficients(
                RamondPlumbingParameter(0.03, "identity"), 0
            )[0],
            2.0,
            places=13,
        )
        self.assertAlmostEqual(
            plus.cycle_projected_raw_coefficients(
                RamondPlumbingParameter(0.03, "parity"), 0
            )[0],
            0.0,
            places=13,
        )
        self.assertAlmostEqual(
            minus.cycle_projected_raw_coefficients(
                RamondPlumbingParameter(0.03, "identity"), 0
            )[0],
            0.0,
            places=13,
        )
        self.assertAlmostEqual(
            minus.cycle_projected_raw_coefficients(
                RamondPlumbingParameter(0.03, "parity"), 0
            )[0],
            2.0,
            places=13,
        )


if __name__ == "__main__":
    unittest.main()
