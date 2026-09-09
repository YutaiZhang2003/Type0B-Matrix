from __future__ import annotations

import unittest

from genus_2.glasses_c_recursion_pbw import run_level_four_audit


class GlassesCRecursionPBWTests(unittest.TestCase):
    def test_every_coefficient_through_total_level_four(self) -> None:
        result = run_level_four_audit()
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["max_total_physical_level"], 4.0)
        self.assertEqual(result["coefficient_comparisons"], 1320)
        self.assertLess(result["maximum_absolute_error"], 1.0e-10)


if __name__ == "__main__":
    unittest.main()
