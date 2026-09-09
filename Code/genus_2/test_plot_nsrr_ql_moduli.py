"""Presentation checks for the Q_L-only plot; no new integrations."""
import unittest
import xml.etree.ElementTree as ET

import plot_nsrr_ql_moduli as plot


class QLModuliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = plot.load_verified()

    def test_common_curve(self):
        expected = [-2.9404987964404605, -3.2267741930153027,
                    -3.690691044623906, -4.301000008590794, -5.012963327220188]
        for row, value in zip(self.data["rows"], expected):
            self.assertAlmostEqual(row[plot.GAP], value, places=11)

    def test_refinement_is_separate(self):
        fine = self.data["separate_refined_point"]
        self.assertEqual((fine["t"], fine["source_N"], fine["target_N"]), (.6, 6, 7))
        self.assertAlmostEqual(fine[plot.GAP], -3.9487987902696253, places=11)
        self.assertNotEqual(fine[plot.GAP], self.data["rows"][2][plot.GAP])

    def test_only_ql_is_compared(self):
        svg = plot.svg_plot(self.data)
        root = ET.fromstring(svg)
        ns = '{http://www.w3.org/2000/svg}'
        self.assertEqual(len(root.findall('.//' + ns + 'polyline')), 1)
        self.assertEqual(len(root.findall('.//' + ns + 'circle')), 5)
        self.assertEqual(len(root.findall('.//' + ns + 'path')), 1)
        self.assertIn("Q_L,NSRR trial / Q_L,NSNSNS", svg)
        self.assertNotIn("Z_free", svg)
        self.assertNotIn("legacy", svg)
        self.assertIn("remains a trial", svg)


if __name__ == "__main__":
    unittest.main()
