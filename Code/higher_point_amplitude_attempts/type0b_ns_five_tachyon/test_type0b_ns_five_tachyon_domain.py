"""Regressions for the exploratory five-point boundary-domain audit."""

import unittest

from type0b_ns_five_tachyon import balanced_equal_energy
from type0b_ns_five_tachyon_domain import (
    CERTIFIED_OUTGOING_FREQUENCIES,
    all_c_atlas_orderings,
    general_complex_energy_convergence_audit,
    is_unavoidable_three_fixed_pco_record,
    is_one_divisor_subtraction_record,
    minimal_subtraction_ray_certificate,
    minimal_subtraction_ray_frequencies,
    one_divisor_ray_certificate,
    one_divisor_ray_frequencies,
    physical_i_epsilon_frequencies,
    physical_i_epsilon_subtraction_audit,
    three_fixed_pco_subtraction_free_no_go,
)


class Type0BNSFiveTachyonDomainTests(unittest.TestCase):
    def test_physical_i_epsilon_audit_enumerates_complete_degree_zero_forest(self):
        outgoing = physical_i_epsilon_frequencies((0.25,) * 4, 0.01)
        self.assertEqual(outgoing, (0.25 + 0.01j,) * 4)
        audit = physical_i_epsilon_subtraction_audit(
            (0.25,) * 4,
            0.01,
            central_charge_shift=1.0e-5,
        )
        self.assertTrue(audit["exact_energy_conservation"])
        self.assertTrue(audit["undeformed_positive_real_liouville_contours"])
        self.assertEqual(audit["crossed_structure_poles"], [])
        self.assertAlmostEqual(audit["first_C_wall_parameter"], 0.05)
        self.assertAlmostEqual(audit["first_C_wall_clearance"], 0.95)
        self.assertTrue(
            audit["all_counterterm_denominator_imaginary_parts_negative"]
        )
        self.assertEqual(len(audit["ten_boundary_divisors"]), 10)
        self.assertEqual(audit["required_polynomial_mode_count"], 10)
        self.assertEqual(audit["maximum_diagonal_degree"], 0)
        self.assertTrue(audit["all_required_modes_degree_zero"])
        self.assertEqual(audit["compatible_corner_count"], 15)
        self.assertTrue(
            all(
                corner["required_degree_pairs"] == [[0, 0]]
                for corner in audit["fifteen_compatible_corners"]
            )
        )

        records = {
            tuple(record["pair"]): record
            for record in audit["ten_boundary_divisors"]
        }
        self.assertEqual(records[(0, 1)]["threshold"], 1)
        self.assertEqual(records[(1, 3)]["threshold"], 0)
        self.assertEqual(records[(3, 4)]["threshold"], 1)
        self.assertAlmostEqual(
            records[(0, 1)]["divergent_momentum_intervals"][0][1] ** 2,
            1.0 + 1.0e-5 / 12.0 + ((0.75 + 0.03j) ** 2).real,
        )

    def test_physical_i_epsilon_audit_lists_higher_diagonal_modes_when_needed(self):
        audit = physical_i_epsilon_subtraction_audit((1.0,) * 4, 0.01)
        raised_pair = next(
            record
            for record in audit["ten_boundary_divisors"]
            if record["pair"] == [1, 2]
        )
        self.assertEqual(raised_pair["required_diagonal_degrees"], [0, 1, 2])
        self.assertFalse(audit["all_required_modes_degree_zero"])

    def test_old_separated_point_is_rejected_by_superghost_face(self):
        audit = general_complex_energy_convergence_audit(
            CERTIFIED_OUTGOING_FREQUENCIES
        )
        self.assertFalse(audit["strictly_subtraction_free"])
        face = next(
            record
            for record in audit["records"]
            if record["kind"] == "face-continuous"
            and set(record["pair"]) == {3, 4}
        )
        self.assertEqual(face["picture_zero_count"], 0)
        self.assertEqual(face["threshold"], 1)
        self.assertAlmostEqual(face["margin"], -0.751744, places=12)

    def test_no_go_when_both_endpoint_faces_converge(self):
        outgoing = (
            -0.10 + 0.61j,
            -0.11 + 0.64j,
            -0.12 + 0.63j,
            -0.13 + 0.66j,
        )
        certificate = three_fixed_pco_subtraction_free_no_go(outgoing)
        self.assertFalse(certificate["subtraction_free_domain_exists"])
        self.assertTrue(certificate["both_endpoint_faces_converge"])
        self.assertLess(certificate["middle_margin_to_raised_pair"], -1.0)
        self.assertLess(
            certificate["middle_margin_to_minus_one_pair"], -1.0
        )

    def test_no_go_also_covers_a_nonconvergent_endpoint_face(self):
        certificate = three_fixed_pco_subtraction_free_no_go(
            CERTIFIED_OUTGOING_FREQUENCIES
        )
        self.assertFalse(certificate["subtraction_free_domain_exists"])
        self.assertFalse(certificate["both_endpoint_faces_converge"])
        self.assertLess(certificate["minus_one_face_margin"], 0.0)

    def test_equal_frequency_endpoint_curve_is_not_global(self):
        omega = balanced_equal_energy(0.602)
        audit = general_complex_energy_convergence_audit((omega,) * 4)
        self.assertFalse(audit["strictly_subtraction_free"])
        self.assertLessEqual(audit["minimum_integrability_margin"], 0.0)

    def test_all_c_atlas_has_all_stable_topologies(self):
        orderings = all_c_atlas_orderings(CERTIFIED_OUTGOING_FREQUENCIES)
        self.assertEqual(len(orderings), 120)
        self.assertEqual(len(set(orderings)), 120)
        topologies = {
            frozenset((frozenset(ordering[:2]), frozenset(ordering[3:])))
            for ordering in orderings
        }
        self.assertEqual(len(topologies), 15)
        self.assertTrue(
            all(
                sum(
                    frozenset((frozenset(ordering[:2]), frozenset(ordering[3:])))
                    == topology
                    for ordering in orderings
                )
                == 8
                for topology in topologies
            )
        )

    def test_mixed_real_signs_do_not_evade_the_middle_line(self):
        outgoing = (
            -0.41 + 0.71j,
            0.09 + 0.62j,
            -0.39 + 0.73j,
            0.08 + 0.64j,
        )
        certificate = three_fixed_pco_subtraction_free_no_go(outgoing)
        self.assertTrue(certificate["both_endpoint_faces_converge"])
        self.assertFalse(certificate["subtraction_free_domain_exists"])

    def test_historical_one_corner_ray_is_rejected_by_complete_ledger(self):
        certificate = minimal_subtraction_ray_certificate()
        self.assertFalse(certificate["minimal_subtraction_interval_certified"])
        self.assertEqual(certificate["subtraction_stratum_count"], 31)
        self.assertEqual(certificate["theorem_mandated_stratum_count"], 1)
        self.assertLess(certificate["minimum_remainder_margin"], -5.6)
        self.assertLess(certificate["minimum_pole_chamber_clearance"], 0.001)
        self.assertEqual(len(certificate["ten_sampling_parameters"]), 10)
        for parameter in certificate["ten_sampling_parameters"]:
            audit = general_complex_energy_convergence_audit(
                minimal_subtraction_ray_frequencies(parameter)
            )
            negative = [
                record
                for record in audit["records"]
                if record["margin"] <= 0.0
            ]
            self.assertEqual(len(negative), 31)
            self.assertEqual(
                sum(is_unavoidable_three_fixed_pco_record(item) for item in negative),
                1,
            )

    def test_corrected_ray_has_exactly_one_divisor_subtraction(self):
        certificate = one_divisor_ray_certificate()
        self.assertTrue(certificate["one_divisor_interval_certified"])
        self.assertEqual(certificate["subtraction_stratum_count"], 1)
        self.assertGreater(
            certificate["minimum_positive_remainder_margin"], 0.044
        )
        self.assertGreater(
            certificate["minimum_first_descendant_margin"], 0.014
        )
        self.assertGreater(
            certificate["minimum_pole_chamber_clearance"], 0.021
        )
        self.assertGreater(certificate["minimum_frequency_separation"], 0.298)
        self.assertEqual(len(certificate["ten_sampling_parameters"]), 10)
        for parameter in certificate["ten_sampling_parameters"]:
            audit = general_complex_energy_convergence_audit(
                one_divisor_ray_frequencies(parameter)
            )
            negative = [
                record
                for record in audit["records"]
                if record["margin"] <= 0.0
            ]
            self.assertEqual(len(negative), 1)
            self.assertTrue(is_one_divisor_subtraction_record(negative[0]))


if __name__ == "__main__":
    unittest.main()
