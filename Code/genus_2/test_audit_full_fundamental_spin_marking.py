"""Complete marking checks, including deliberately omitted branch controls."""
import unittest

import numpy as np

import audit_full_fundamental_spin_marking as audit


class FullMarkingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = audit.run()

    def test_historical_branch_changes_spin_and_is_not_omitted(self):
        for row in self.report["historical"]:
            atlas, branch, fd = row["marking_steps"]
            self.assertEqual(atlas["spin_before"], [[0, 0], [0, 0]])
            self.assertEqual(atlas["spin_after"], [[0, 0], [1, 1]])
            self.assertEqual(branch["spin_after"], [[0, 0], [0, 0]])
            self.assertEqual(fd["spin_after"], row["glasses_direct_FD_step"]["spin_after"])
            self.assertEqual(row["spin_FD"], [[0, 0], [0, 0]])
            composed = np.asarray(branch["matrix"])@np.asarray(atlas["matrix"])
            np.testing.assert_array_equal(composed, row["atlas_branch_composition"])

    def test_original_overlap_record_reconstructed(self):
        for row in self.report["historical"]:
            self.assertLess(row["original_atlas_record_residual"], 1e-12)
            self.assertGreater(row["FD_domain_margin"], 0)

    def test_collocation_paths_include_an_additional_branch(self):
        for row in self.report["historical"]:
            f = row["channels"]["theta"]["forward"]
            expected = audit.OLD_COLLOCATION_MINUS_SAVED[row["point_id"]]
            np.testing.assert_array_equal(f["raw_collocation_minus_saved_branch"], expected)
            self.assertLess(f["native_period_residual"], 4e-10)
            self.assertLess(f["Schottky_word9_native_residual"], 1e-8)
            self.assertEqual(f["raw_collocation_to_saved_spin_step"]["spin_after"], [[0, 0], [0, 0]])
        changed = self.report["historical"][2]["channels"]["theta"]["forward"]
        self.assertEqual(changed["raw_collocation_spin"], [[0, 0], [0, 1]])
        raw = audit.previous.omega_array(changed["omega_forward_raw"])
        corrected = audit.previous.omega_array(changed["omega_forward"])
        self.assertGreater(np.max(abs(raw-corrected)), .99)

    def test_q_periods_reach_same_fundamental_domain(self):
        for section in ("historical", "current"):
            for row in self.report[section]:
                self.assertGreaterEqual(row["FD_domain_margin"], -1e-10)
                for channel in row["channels"].values():
                    self.assertLess(channel["forward"]["native_period_residual"], 2e-9)
                    self.assertLess(channel["all_ten_even_spin_covariance_error"], 1e-13)

    def test_current_spin_changes_in_the_chosen_FD_marking(self):
        for row in self.report["current"]:
            self.assertEqual(row["original_spin"], [[0, 1], [1, 0]])
            expected = [[0, 1], [0, 0]] if row["t"] < .60 else [[0, 0], [0, 1]]
            self.assertEqual(row["spin_FD"], expected)
            for channel in row["channels"].values():
                self.assertEqual(channel["marking_steps"][-1]["spin_after"], expected)

    def test_composed_current_sp4_matrix_not_just_matching_periods(self):
        geometry = audit.previous.load(audit.GEOMETRY)
        source_to_target = np.asarray(geometry["source_to_target"])
        for row in self.report["current"]:
            source, target = (row["channels"][ch] for ch in ("source", "target"))
            np.testing.assert_array_equal(np.asarray(target["native_to_FD"])@source_to_target,
                                          source["native_to_FD"])

    def test_charge_branch_and_FD_are_both_included(self):
        for row in self.report["current"]:
            for channel in row["channels"].values():
                branch_step, fd_step = channel["marking_steps"]
                self.assertEqual(branch_step["spin_after"], channel["native_spin"])
                np.testing.assert_array_equal(np.asarray(fd_step["matrix"])@np.asarray(branch_step["matrix"]),
                                              channel["charge_to_FD"])
                self.assertLess(channel["charge_to_FD_period_residual"], 2e-9)

    def test_forgetting_spin_transport_would_be_a_large_error(self):
        row = next(r for r in self.report["current"] if r["t"] == .60)
        target = row["channels"]["target"]
        self.assertAlmostEqual(target["wrong_untransported_native_spin_in_FD_relative_error"], .1237752510075727, places=12)
        self.assertLess(abs(target["fixed_free_invariant_over_FD_minus_one"]), 2e-11)

    def test_legacy_amplitude_difference_survives_correct_marking(self):
        row = next(r for r in self.report["current"] if r["t"] == .60)
        target = row["channels"]["target"]
        self.assertAlmostEqual(target["legacy_free_invariant_over_correct_FD_minus_one"], -.0472446707332006, places=12)
        for section in ("historical", "current"):
            for point in self.report[section]:
                for channel in point["channels"].values():
                    self.assertLess(abs(channel["fixed_free_invariant_over_FD_minus_one"]), 5e-10)

    def test_word_and_symplectic_inverse(self):
        word = "T12^-1 T22 gl-shear-12 T11 full-s I"
        matrix = audit.word_matrix(word)
        np.testing.assert_array_equal(matrix, [[2, -1, -1, -1], [-2, 1, 0, -1],
                                              [1, 0, 0, 0], [-1, 1, 0, 0]])
        np.testing.assert_array_equal(matrix@audit.symplectic_inverse(matrix), np.eye(4, dtype=int))


if __name__ == "__main__":
    unittest.main()
