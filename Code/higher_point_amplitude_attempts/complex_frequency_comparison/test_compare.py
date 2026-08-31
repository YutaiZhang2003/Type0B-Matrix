"""Normalization and input-provenance checks; no worldsheet convergence claims."""
import copy
import math
import unittest

from compare import (SUMMARY_SCHEMA, PRELIMINARY_SUMMARY_SCHEMA, compare_summary, configured_energies,
                     decoded, encoded, matrix_coefficients)


class ComparisonTests(unittest.TestCase):
    def test_bry_low_point_coefficients_at_unequal_complex_energies(self):
        outgoing = (0.11 + 0.03j, 0.23 + 0.04j, 0.19 + 0.01j)
        for n in (2, 3):
            w = outgoing[:n]
            omega = sum(w)
            expected = 2**n * 1j * omega * math.prod(w)
            if n == 3:
                expected *= 1 + 2j * omega
            prediction = matrix_coefficients(w)
            self.assertAlmostEqual(decoded(prediction["all_right_mode"]), expected)
            self.assertAlmostEqual(decoded(prediction["all_tachyon"]), expected / 2**n)

    def test_frozen_c1_five_and_six_point_polynomials_with_energy_rescaling(self):
        # Frozen c=1 reference: A5=-4 t^5(2-12t+16t^2),
        # A6=-5i t^6(1-5t)(2-5t)(3-5t). Here t=2*Im(omega_WS).
        for t in (0.04, 0.13, 0.27):
            cases = (
                (4, -4 * t**5 * (2 - 12*t + 16*t*t)),
                (5, -5j * t**6 * (1-5*t) * (2-5*t) * (3-5*t)),
            )
            for n, c1 in cases:
                actual = decoded(matrix_coefficients((0.5j*t,) * n)["all_right_mode"])
                self.assertAlmostEqual(actual, c1 / 2)

    def fixture(self):
        config = {"physics": {"real_outgoing_energies": [.25]*4,
                              "epsilon": .02, "epsilon_weights": [1, 2, 3, 4]},
                  "array": {"task_count": 4},
                  "subtraction": {"collar_radii": [.01, .005]}}
        row = {"collar_radius": .01, "block_backend": "c", "integral_mean": encoded(2+3j),
               "standard_error_real": .4, "standard_error_imag": .7,
               "replicate_count": 64, "face_collar_certificates_passed": False}
        summary = {"schema": SUMMARY_SCHEMA, "config_sha256": "verified-hash",
                   "task_count": 4, "matrix_model_used": False,
                   "status": "worldsheet_cluster_preflight_not_frozen",
                   "radius_summaries": [row, dict(row, collar_radius=.005)]}
        return config, summary

    def test_preliminary_data_keeps_complex_energies_all_radii_and_error_rotation(self):
        config, summary = self.fixture()
        before = copy.deepcopy(summary)
        result = compare_summary(summary, config, "verified-hash")
        self.assertEqual(summary, before)
        self.assertEqual(configured_energies(config), (.25+.02j, .25+.04j, .25+.06j, .25+.08j))
        self.assertEqual(decoded(result["prediction"]["incoming_energy"]), 1+.2j)
        self.assertFalse(result["epsilon_extrapolation_required"])
        self.assertFalse(result["convergence_certified"])
        self.assertEqual(len(result["comparisons"]), 2)
        row = result["comparisons"][0]
        self.assertAlmostEqual(decoded(row["worldsheet_all_tachyon"]), (-3+2j)/math.pi**2)
        self.assertAlmostEqual(row["worldsheet_qmc_standard_error_real"], .7/math.pi**2)
        self.assertAlmostEqual(row["worldsheet_qmc_standard_error_imag"], .4/math.pi**2)
        self.assertFalse(row["face_collar_certificates_passed"])

    def test_rejects_mismatched_config_or_missing_radius(self):
        config, summary = self.fixture()
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            compare_summary(summary, config, "wrong-hash")
        summary["radius_summaries"].pop()
        with self.assertRaisesRegex(ValueError, "every configured collar radius"):
            compare_summary(summary, config, "verified-hash")

    def test_one_radius_preliminary_summary_needs_no_convergence_certificate(self):
        config, summary = self.fixture()
        config["subtraction"]["collar_radii"] = [.01]
        summary["schema"] = PRELIMINARY_SUMMARY_SCHEMA
        summary["radius_summaries"] = summary["radius_summaries"][:1]
        summary["radius_summaries"][0]["face_collar_certificates_passed"] = None
        result = compare_summary(summary, config, "verified-hash")
        self.assertEqual(len(result["comparisons"]), 1)
        self.assertIsNone(result["comparisons"][0]["face_collar_certificates_passed"])
        self.assertFalse(result["convergence_certified"])

    def test_cannot_label_fivepoint_data_as_sixpoint(self):
        config, summary = self.fixture()
        config["physics"]["real_outgoing_energies"].append(.2)
        config["physics"]["epsilon_weights"].append(1)
        with self.assertRaisesRegex(ValueError, "not 1->5"):
            compare_summary(summary, config, "verified-hash")

    def test_matrix_polynomial_has_expected_complex_energy_zero(self):
        # A zero prediction is legitimate on an analytically continued ray;
        # the matrix function itself supports it independently of this run's chamber.
        self.assertEqual(decoded(matrix_coefficients((.125j,)*4)["all_tachyon"]), 0)


if __name__ == "__main__":
    unittest.main()
