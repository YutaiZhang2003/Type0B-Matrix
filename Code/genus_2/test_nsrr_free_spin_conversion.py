"""The new free-spin adapter must not silently assume a frame identity."""
from pathlib import Path
import unittest
from unittest.mock import patch

import audit_nsrr_free_spin_conversion as conversion
import compare_nsrr_nsnsns_theta as comparison


class FreeSpinConversionTests(unittest.TestCase):
    def test_actual_old_source_fails_the_necessary_spin_ratio_test(self):
        result=conversion.audit(comparison.SOURCE_Q,comparison.SOURCE_OMEGA_CHART,max_mode=24)
        self.assertFalse(result["compatible"])
        self.assertGreater(result["maximum_relative_incompatibility"], .4)
        self.assertEqual(len(result["rows"]),4)

    def test_the_new_source_denominator_refuses_uncertified_conversion(self):
        with self.assertRaisesRegex(ArithmeticError,"not certified"):
            comparison.same_frame_free_factors(24)

    def test_compatible_fixture_passes_without_fitting_a_normalization(self):
        with patch.object(conversion,"audit",return_value={"compatible":True,"maximum_relative_incompatibility":0.}) as check:
            result=conversion.require_compatible_theta_ratio((.01,.02,.03),None)
            self.assertTrue(result["compatible"])
            check.assert_called_once()


if __name__ == "__main__":
    unittest.main()
