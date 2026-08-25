#!/usr/bin/env python3
"""Focused exact tests for the reconstructed NS branching-state layer."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import sympy as sp


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import check_ns_branch_norms as ns


class NSBranchingConverterTests(unittest.TestCase):
    def test_chi_path_signs(self):
        modes = (3, 1)
        self.assertEqual(ns.coefficient_in_chi_product(modes, ()), 1)
        self.assertEqual(ns.coefficient_in_chi_product(modes, (3,)), sp.I)
        self.assertEqual(ns.coefficient_in_chi_product(modes, (1,)), -sp.I)
        self.assertEqual(ns.coefficient_in_chi_product(modes, (3, 1)), -1)

    def test_documented_first_branch(self):
        self.assertEqual(ns.audit_first_branch(), 2 / (2 * ns.P + ns.Q))
        self.assertEqual(
            ns.branch_norm(sp.Rational(1, 2))[3],
            -4 * ns.P / (2 * ns.P + ns.Q),
        )

    def test_symbolic_and_specialized_transports_agree(self):
        q_value = sp.Rational(13, 6)
        momentum = sp.Rational(1, 5)
        for modes in ((), (1,), (3,), (3, 1)):
            symbolic_basis, symbolic = ns.abstract_eta_coefficients(modes)
            exact_basis, exact = ns.abstract_eta_coefficients_at(
                modes, q_value, momentum
            )
            self.assertEqual(symbolic_basis, exact_basis)
            residual = symbolic.subs(
                {ns.Q: q_value, ns.P: momentum}, simultaneous=True
            ) - exact
            self.assertTrue(all(sp.cancel(value) == 0 for value in residual))

    def test_endpoint_round_trips_through_n_one(self):
        self.assertEqual(ns.audit_endpoints(sp.Integer(1)), 7)


if __name__ == "__main__":
    unittest.main()
