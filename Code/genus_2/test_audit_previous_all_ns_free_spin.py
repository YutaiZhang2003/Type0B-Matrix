"""Fast historical/free-spin controls; no Liouville integrations."""
import unittest

import numpy as np

import audit_previous_all_ns_free_spin as audit


class PreviousAllNSFreeSpinTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = audit.run()

    def test_parity_fourier_inverse(self):
        components = {"000": .9+.1j, "110": .08-.02j,
                      "101": .007+.004j, "011": -.03+.01j}
        raw = [components["000"]+a*b*components["110"]
               +a*components["101"]+b*components["011"] for a, b, _ in audit.LIFTS]
        actual = audit.parity_components(raw)
        for key in components:
            self.assertLess(abs(actual[key]-components[key]), 1e-15)

    def test_five_historical_free_values_reproduced(self):
        self.assertEqual(len(self.result["historical_rows"]), 5)
        for point in self.result["historical_rows"]:
            for channel in point["channels"].values():
                self.assertLess(channel["saved_free_reproduction_error"], 1e-12)
                self.assertLess(channel["mode_24_to_32_fixed_relative_change"], 1e-11)
                self.assertLess(channel["mode_24_to_32_legacy_relative_change"], 1e-11)

    def test_both_channels_four_spin_bosonization(self):
        for point in self.result["historical_rows"]:
            for channel in point["channels"].values():
                self.assertEqual(len(channel["four_NS_controls"]), 4)
                for row in channel["four_NS_controls"]:
                    self.assertLess(row["complex_bosonization_relative_error"], 1e-12)

    def test_charge_marking_is_not_the_recent_NSrr_branch(self):
        for point in self.result["historical_rows"]:
            theta, glasses = (point["channels"][c] for c in ("theta", "glasses"))
            self.assertEqual(theta["integer_period_branch"], [[0, 0], [0, 0]])
            self.assertLess(theta["period_residual"], 4e-10)
            self.assertLess(glasses["period_residual"], 1e-13)
            marked = audit.omega_array(glasses["omega_marked"])
            charge = audit.omega_array(glasses["omega_charge"])
            self.assertGreater(np.max(abs(charge-marked)), .05)

    def test_old_label_does_not_identify_filtered_free_spin(self):
        for point in self.result["historical_rows"]:
            theta = point["channels"]["theta"]
            selected = theta["four_NS_controls"][1]
            self.assertEqual(selected["lifts"], [1, -1, 1])
            self.assertEqual(selected["old_helper_label"], [[0, 0], [0, 0]])
            self.assertEqual(selected["fixed_spin_marked"], [[0, 0], [0, 1]])
            self.assertLess(theta["F_selected_equals_D0000_minus_2_S101_error"], 1e-12)
            self.assertLess(theta["free_basis_identity_error"], 1e-12)
            self.assertLess(theta["legacy_over_fixed_minus_one"], -9e-7)
            self.assertGreater(theta["legacy_over_fixed_minus_one"], -6.5e-6)

    def test_glasses_passes_but_filtered_theta_fails_frame_identity(self):
        for point in self.result["historical_rows"]:
            self.assertLess(abs(point["channels"]["glasses"]["legacy_over_fixed_minus_one"]), 1e-13)
            self.assertLess(abs(point["fixed_free_modular_frame_residual"]), 5e-11)
            self.assertGreater(abs(point["legacy_free_modular_frame_residual"]), 9e-7)

    def test_denominator_only_comparison_is_not_new_numerator(self):
        first = self.result["historical_rows"][0]
        self.assertAlmostEqual(first["historical_R24_N10_Q_ratio_minus_one"], -.0002983919014382108, places=14)
        self.assertAlmostEqual(first["counterfactual_fixed_denominator_Q_ratio_minus_one"], -.0003475858501, places=12)
        self.assertIsNone(self.result["physical_Q_NSrr"])

    def test_thin_tube_stress_exposes_hidden_sector(self):
        rows = self.result["thin_tube_stress_test"]
        errors = [abs(r["legacy_over_fixed_minus_one"]) for r in rows]
        self.assertEqual(errors, sorted(errors))
        self.assertLess(errors[0], 6e-6)
        self.assertGreater(errors[-1], .047)
        self.assertLess(errors[-1], .049)
        for row in rows:
            self.assertLess(max(abs(complex(q)) for q in row["q"]), .25)

    def test_current_target_discrepancy_is_different_parity_sector(self):
        target = self.result["current_spin_pair_control"]["target_filtered_control"]
        self.assertEqual(target["fixed_spin_unfiltered_lifts"], [-1, 1, 1])
        self.assertLess(target["F_selected_equals_D_fixed_plus_2_S110_error"], 1e-12)
        self.assertAlmostEqual(target["legacy_over_fixed_minus_one"], -.0472446707494335, places=13)

    def test_consistent_free_spin_change_cannot_fit_Q_ratio(self):
        control = self.result["current_spin_pair_control"]
        first, second = control["matched_even_pairs"]
        self.assertNotEqual(first["Z_free_source"], second["Z_free_source"])
        self.assertEqual(first["target_spin"], [[0, 0], [0, 0]])
        self.assertEqual(second["target_spin"], [[0, 0], [1, 0]])
        self.assertLess(abs(first["source_over_target_free"]/second["source_over_target_free"]-1), 1e-10)
        self.assertLess(abs(control["paired_denominator_change_Q_ratio_multiplier"]-1), 1e-9)


if __name__ == "__main__":
    unittest.main()
