#!/usr/bin/env python3
"""Regression tests for off-axis fixed-spin analytic continuation."""

from __future__ import annotations

import unittest

from run_nsrr_nsnsns_offaxis_constant_scan import continued_target_lifts


class ContinuedTargetLiftsTest(unittest.TestCase):
    def test_reference_branch_keeps_reference_lifts(self) -> None:
        reference = ((-1, -1), (-1, 0))
        self.assertEqual(
            continued_target_lifts((0.1, 0.2, 0.3), reference, reference),
            (1, -1, 1),
        )

    def test_two_observed_branch_changes(self) -> None:
        reference = ((-1, -1), (-1, 0))
        self.assertEqual(
            continued_target_lifts(
                (0.1, 0.2, 0.3), reference, ((0, 0), (0, 1))
            ),
            (1, -1, -1),
        )
        self.assertEqual(
            continued_target_lifts(
                (0.1, 0.2, 0.3), reference, ((-1, -1), (-1, 1))
            ),
            (1, 1, 1),
        )


if __name__ == "__main__":
    unittest.main()
