"""Checks for the two-quantity moduli plot; no new Liouville nodes."""
import unittest
import xml.etree.ElementTree as ET

import plot_nsrr_moduli_differences as plot


class ModuliDifferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = plot.assemble()

    def test_uniform_cutoffs_and_original_modulus(self):
        self.assertEqual([r["t"] for r in self.data["rows"]], [.52, .56, .60, .64, .68])
        self.assertEqual(self.data["common_cutoffs"], {"source_L": 3, "source_N": 5, "target_R": 16, "target_N": 5})
        self.assertEqual(self.data["new_Liouville_integrals"], 0)
        self.assertIsNone(self.data["physical_Q_NSrr"])

    def test_two_distinct_ratios_and_kappa_power(self):
        for r in self.data["rows"]:
            free_ratio = r["Z_free_target_legacy"]/r["Z_free_target_fixed"]
            qr = r["Q_NSrr_trial_N5_L3"]/r["Q_NSnsns_N5_R16"]
            self.assertAlmostEqual(r["target_free_legacy_over_fixed_minus_one_percent"], 100*(free_ratio-1), places=12)
            self.assertAlmostEqual(r["Q_ratio_minus_one_percent"], 100*(qr-1), places=12)
            self.assertAlmostEqual(r["target_only_legacy_denominator_Q_ratio_minus_one_percent"],
                                   100*(qr*free_ratio**self.data["kappa"]-1), places=12)

    def test_free_factors_recomputed_and_marked(self):
        for r in self.data["rows"]:
            for c in r["free_checks"].values():
                self.assertLess(c["mode_32_to_40_change"], 1e-12)
                self.assertLess(c["saved_free_reproduction_error"], 1e-12)
                self.assertLess(c["fundamental_domain_invariant_error"], 1e-8)
            self.assertLess(r["legacy_mode_32_to_40_change"], 1e-12)
            self.assertLess(r["saved_legacy_free_reproduction_error"], 1e-12)

    def test_reference_point_reproduction(self):
        r = self.data["rows"][2]
        self.assertAlmostEqual(r["Q_ratio_minus_one_percent"], -3.690691044623906, places=10)
        self.assertAlmostEqual(r["target_free_legacy_over_fixed_minus_one_percent"], -4.72446707494335, places=10)

    def test_refinement_not_spliced_into_common_curve(self):
        refined = self.data["separate_refined_point"]
        self.assertEqual((refined["source_N"], refined["target_N"]), (6, 7))
        self.assertAlmostEqual(refined["Q_ratio_minus_one_percent"], -3.948798790269625, places=12)
        self.assertNotEqual(refined["Q_ratio_minus_one_percent"], self.data["rows"][2]["Q_ratio_minus_one_percent"])

    def test_render_has_two_panels_and_trial_warning(self):
        svg = plot.svg_plot(self.data)
        root = ET.fromstring(svg)
        self.assertEqual(root.attrib["width"], "1120")
        self.assertEqual(len(root.findall('.//{http://www.w3.org/2000/svg}polyline')), 2)
        self.assertIn("Physical NSRR nonchiral assembly remains a trial", svg)
        self.assertIn("same target plumbing frame", svg)


if __name__ == "__main__":
    unittest.main()
