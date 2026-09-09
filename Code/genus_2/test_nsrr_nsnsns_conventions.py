"""Verify the actual integrand's coefficient phase and independent sewing sign.

The net positive odd all-NS contribution is NOT sufficient as a test: omitting
both minus signs would accidentally give the same answer. Check each boundary.
"""
from itertools import product
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import compare_nsrr_nsnsns_theta as comparison
from theta_partition import theta_sector_pair, theta_diagonal_sector_contribution
from generic_super_liouville_structure_constants import hjs_rr_ns_constant
from run_nsrr_nsnsns_toy import refined_fivepoint_config, toy_config
from nsrr_nsnsns_theta_omega_scan import _load, tasks
from direct_state_check import PhysicalThreePoint
import nsrr_double_virasoro_block as double_virasoro


class CoefficientConventionTests(unittest.TestCase):
    def test_actual_all_ns_caller_inserts_i_once_before_squaring(self):
        constants = SimpleNamespace(ns_constants=lambda *args: (2., 3.))
        recursion = SimpleNamespace(collision_aware_block_mp=lambda **kwargs: 1.)
        with patch.object(comparison, "_primary", return_value=1.), patch.object(
            comparison, "theta_diagonal_sector_contribution",
            wraps=theta_diagonal_sector_contribution,
        ) as assembly:
            values = comparison.all_ns_node(
                b=1.4, q_values=(.02, .03, .04), lifts=(1, -1, 1),
                recursion_order=8, momenta=(.2, .3, .4), measure=1.,
                constants=constants, recursion=recursion, block_method="collision_aware_mp")
        self.assertEqual([c.kwargs["structure_weight"] for c in assembly.call_args_list], [4., -9.])
        self.assertEqual(values, (4., 9.))
        self.assertEqual([c.kwargs["primary_parities"] for c in assembly.call_args_list], [(0, 0, 0)] * 2)

    def test_decomposition_sign_depends_on_absolute_not_relative_parity(self):
        for parities in product((0, 1), repeat=3):
            for sector in (0, 1):
                absolute = (sector + sum(parities)) % 2
                pair = theta_sector_pair(sector, holomorphic_primary_parities=parities,
                                         antiholomorphic_primary_parities=parities)
                self.assertEqual(pair.sign, (-1) ** absolute)
                self.assertEqual(theta_diagonal_sector_contribution(
                    sector=sector, measure=1., structure_weight=1., primary_times_block=1.,
                    primary_parities=parities), (-1) ** absolute)

    def test_hjs_eta_is_not_the_all_ns_top_coefficient_label(self):
        self.assertEqual(hjs_rr_ns_constant((2., 3.), 1), 2.)
        self.assertEqual(hjs_rr_ns_constant((2., 3.), -1), 3.)
        modules = (None, SimpleNamespace(ground=lambda state: state),
                   SimpleNamespace(ground=lambda state: state))
        for eta in (1, -1):
            even = PhysicalThreePoint(modules, form_parity=0, eta=eta)
            odd = PhysicalThreePoint(modules, form_parity=1, eta=eta)
            phase = odd.ramond_odd_phase
            self.assertAlmostEqual(even.base_value((None, 0, 0)), 1.)
            self.assertAlmostEqual(even.base_value((None, 1, 1)), -1j * eta)
            self.assertAlmostEqual(odd.base_value((None, 0, 1)), phase)
            self.assertAlmostEqual(odd.base_value((None, 1, 0)), 1j * eta * phase)

    def test_double_virasoro_sewing_sign_is_the_literal_human_note_sign(self):
        # Isolate the contraction from the expensive branching solve. In
        # particular the 2*n1 term and raw left*right (not conjugated) product
        # must survive. This tests the actual enlarged_series implementation.
        labels = (Fraction(1, 2), Fraction(1, 4), Fraction(1, 4))
        for primary_parity in (0, 1):
            evaluator = object.__new__(double_virasoro.NSRRDoubleVirasoroTheta)
            evaluator.b = 1.4
            evaluator.note_momenta = (.2j, .3j, .4j)
            evaluator.primary_parity = primary_parity
            evaluator.triples = (labels,)
            evaluator.cutoff_twice = 4
            evaluator.reduced_products = {labels: {(0, 0, 0): 1.}}
            evaluator.vacuum_squared = {(0, 0, 0): 1.}
            evaluator.raw_grids = {
                (f, eta, a2, a3): {labels: 2+1j if eta == 1 else 3-2j}
                for f, eta, a2, a3 in product((0, 1), (1, -1), (0, 1), (0, 1))}
            for f in (0,):
                with patch.object(double_virasoro, "norm_product", return_value=2.):
                    series = evaluator.enlarged_series(f, 1, -1)
                self.assertEqual(len(series), 2)
                for a2, a3 in product((0, 1), repeat=2):
                    if (1 + a2 + a3) % 2 != f:
                        continue
                    power = 1 + (1 + primary_parity) * (a2 + a3) + a2 * a3
                    key = (1, 0, 0, (1 + primary_parity) % 2, a2, a3)
                    expected = (-1) ** power * (2+1j) * (3-2j) / 4
                    self.assertEqual(series[key], expected)
            even = evaluator.enlarged_series(0, 1, -1)
            odd = evaluator.enlarged_series(1, 1, -1)
            self.assertEqual(odd, {key[:5]+(1-key[5],): -1j*(-1)**(key[3]+key[4])*v
                                   for key, v in even.items()})

    def test_refinement_is_explicit_and_does_not_change_toy_design(self):
        base = _load(Path(__file__).parents[1] / "config/nsrr_nsnsns_theta_omega_scan_20260830.json")
        toy, refined = toy_config(base), refined_fivepoint_config(base)
        self.assertEqual(len(tasks(toy)), 182)
        self.assertEqual(len(tasks(refined)), 378)
        self.assertEqual(refined["source_physical_levels"], [3, 4])
        self.assertEqual(refined["target_recursion_order_twice_level"], 8)
        self.assertEqual(refined["quadrature_orders"], [4, 5])
        self.assertEqual(refined["convention_ledger"], toy["convention_ledger"])


if __name__ == "__main__":
    unittest.main()
