import unittest

from certify_human_note_fivepoint import DEFAULT_CONFIG, build


class HumanNoteFivePointCertificateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = build(DEFAULT_CONFIG, None)

    def test_all_five_points_are_present_and_close(self):
        rows = self.result["fivepoint_R24_N10"]
        self.assertEqual(len(rows), 5)
        self.assertLess(
            max(abs(row["relative_difference"]) for row in rows),
            5.0e-4,
        )

    def test_human_note_odd_phase_recombination(self):
        for row in self.result["fivepoint_R24_N10"]:
            for channel in ("theta", "glasses"):
                data = row["channels"][channel]
                self.assertAlmostEqual(
                    data["human_note_numerator"],
                    data["even_sector_sum"]
                    - data["odd_sector_sum_before_coefficient_phase"],
                )
                self.assertGreater(data["odd_fraction_after_phase"], 0.0)

    def test_direct_physical_free_cutoff(self):
        free = self.result["physical_free"]
        self.assertFalse(free["period_matrix_used"])
        self.assertFalse(free["auxiliary_double_virasoro_fermion_used"])
        self.assertLess(free["max_relative_change_M24_to_M28"], 1.0e-10)

    def test_spin_and_period_matching_are_explicit(self):
        self.assertEqual(len(self.result["spin_ledger"]), 5)
        self.assertTrue(
            all(
                not row["matching_assumed"]
                for row in self.result["geometry"]["rows"]
            )
        )
        self.assertLess(
            self.result["geometry"]["max_symplectic_transport_residual"],
            5.0e-10,
        )


if __name__ == "__main__":
    unittest.main()
