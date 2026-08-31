#!/usr/bin/env python3
"""Regression tests for generic-b super-Liouville structure constants."""

from __future__ import annotations

import unittest

from generic_super_liouville_structure_constants import (
    GenericSuperLiouvilleConstants,
)
from super_liouville_structure_constants import (
    ns_structure_constant,
    ns_tilde_structure_constant,
    rr_ns_structure_constants,
)


class GenericStructureConstantTests(unittest.TestCase):
    def test_generic_b_symmetric_leg_metrics(self) -> None:
        generic = GenericSuperLiouvilleConstants(1.4, dps=40)
        for momentum in (0.13, 0.47, 0.91):
            self.assertAlmostEqual(
                float(generic._symmetric_leg_metric(momentum, "NS")),
                1 / 1.4,
                places=12,
            )
            self.assertAlmostEqual(
                float(generic._symmetric_leg_metric(momentum, "R")),
                1.0,
                places=12,
            )

    def test_b_one_reduces_to_existing_bry_constants(self) -> None:
        momenta = (0.23, 0.41, 0.67)
        generic = GenericSuperLiouvilleConstants(1.0, dps=40)
        c, c_tilde = generic.ns_constants(*momenta)
        old_c = ns_structure_constant(*momenta, precision=40)
        old_tilde = ns_tilde_structure_constant(*momenta, precision=40)
        self.assertLess(abs(c - old_c), 2.0e-10)
        self.assertLess(abs(c_tilde - old_tilde), 2.0e-10)

        generic_rr = generic.rr_ns_constants(
            momenta[0], momenta[1], momenta[2]
        )
        old_rr = rr_ns_structure_constants(*momenta, precision=40)
        self.assertLess(max(abs(a - b) for a, b in zip(generic_rr, old_rr)), 2.0e-10)

    def test_cosmological_factor_is_common(self) -> None:
        plain = GenericSuperLiouvilleConstants(1.4, dps=30)
        dressed = GenericSuperLiouvilleConstants(
            1.4, dps=30, mu=0.8, include_cosmological_prefactor=True
        )
        ratio = dressed.cosmological_three_point_factor()
        p = (0.17, 0.36, 0.72)
        for dressed_value, plain_value in zip(
            dressed.ns_constants(*p), plain.ns_constants(*p)
        ):
            self.assertLess(abs(dressed_value / plain_value - ratio), 2.0e-9)


if __name__ == "__main__":
    unittest.main()
