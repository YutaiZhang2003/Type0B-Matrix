"""Regression tests for NS fusion factorization against global osp(1|2)."""

from __future__ import annotations

from itertools import product
import unittest

from check_ns_fusion_global_osp import (
    human_note_endpoint_sign,
    relative_structure_label,
    run_checks,
)


class NSFusionGlobalOSPTests(unittest.TestCase):
    def test_human_note_endpoint_signs(self) -> None:
        for null_parity in (0, 1):
            for bits in product((0, 1), repeat=3):
                for primaries in product((0, 1), repeat=3):
                    expected_exponents = (
                        null_parity * (primaries[0] + bits[0]),
                        0,
                        null_parity * (1 + primaries[1]),
                    )
                    for slot, exponent in enumerate(expected_exponents):
                        self.assertEqual(
                            human_note_endpoint_sign(
                                slot=slot,
                                null_parity=null_parity,
                                descendant_parities=bits,
                                primary_parities=primaries,
                            ),
                            (-1) ** exponent,
                            (null_parity, bits, primaries, slot),
                        )

    def test_absolute_to_relative_label_conversion(self) -> None:
        for absolute in (0, 1):
            for primaries in (
                (0, 0, 0),
                (1, 0, 0),
                (1, 1, 0),
                (1, 1, 1),
            ):
                self.assertEqual(
                    relative_structure_label(absolute, primaries),
                    absolute ^ (sum(primaries) % 2),
                )

    def test_primary_global_osp_factorization_in_all_slots(self) -> None:
        summary = run_checks(maximum_total_occupation=0)
        self.assertEqual(summary.null_labels, ("(1,1)", "(3,1)", "(2,2)", "(5,1)"))
        self.assertEqual(summary.slots_checked, ("infinity", "one", "zero"))
        self.assertEqual(summary.exact_factorization_count, 768)
        self.assertEqual(summary.intrinsic_parity_label_count, 768)
        self.assertEqual(summary.primary_parity_covariance_count, 1536)


if __name__ == "__main__":
    unittest.main()
