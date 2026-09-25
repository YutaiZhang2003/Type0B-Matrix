#!/usr/bin/env python3
"""Algebraic checks for the genuine Spin(23) closed-form candidates."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
import unittest

import numpy as np

from heterotic_so23_1to3_vvvv_fit_bundle.heterotic_so23_1to3 import (
    ssvv_resonance_coefficient,
)

from spin23_genuine_formulas import (
    s_to_sss_raw_candidate,
    s_to_sss_unit_descendants_candidate,
    s_to_svv_raw_candidate,
    s_to_svv_unit_descendants_candidate,
    singlet_descendant_norm,
    v_to_vss_raw_candidate,
    v_to_vss_unit_descendants_candidate,
)


class GenuineFormulaTests(unittest.TestCase):
    """Check exact symmetries, normalizations, and factorization residues."""

    def setUp(self) -> None:
        self.energies = (
            0.13 + 0.16j,
            0.21 + 0.18j,
            0.27 + 0.14j,
        )

    def test_ssvv_is_symmetric_in_vector_legs(self) -> None:
        w1, w2, w3 = self.energies
        self.assertAlmostEqual(
            s_to_svv_raw_candidate(w1, w2, w3),
            s_to_svv_raw_candidate(w1, w3, w2),
        )

    def test_ssss_is_symmetric_in_all_outgoing_legs(self) -> None:
        reference = s_to_sss_raw_candidate(*self.energies)
        for permutation in itertools.permutations(self.energies):
            self.assertAlmostEqual(reference, s_to_sss_raw_candidate(*permutation))

    def test_v_to_vss_is_symmetric_in_singlet_legs(self) -> None:
        w1, w2, w3 = self.energies
        self.assertAlmostEqual(
            v_to_vss_raw_candidate(w1, w2, w3),
            v_to_vss_raw_candidate(w1, w3, w2),
        )

    def test_v_to_vss_audit_points_match_locked_formula(self) -> None:
        audit_path = (
            Path(__file__).resolve().parent
            / "data_exports"
            / "spin23_v_to_vss_formula_audit.json"
        )
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        residuals = []
        for row in audit["q4_points"]:
            energies = tuple(
                complex(value["real"], value["imag"])
                for value in row["outgoing_energies"]
            )
            recorded = complex(
                row["locked_formula"]["real"],
                row["locked_formula"]["imag"],
            )
            direct = complex(
                row["direct_amplitude"]["real"],
                row["direct_amplitude"]["imag"],
            )
            self.assertAlmostEqual(v_to_vss_raw_candidate(*energies), recorded)
            residual = abs(direct - recorded) / abs(recorded)
            self.assertAlmostEqual(residual, row["relative_residual"])
            residuals.append(residual)
        fit = audit["coefficient_fit"]
        self.assertAlmostEqual(max(residuals), fit["maximum_relative_residual"])
        self.assertAlmostEqual(
            math.sqrt(sum(value**2 for value in residuals) / len(residuals)),
            fit["rms_relative_residual"],
        )

        for row in audit["q5_checks"]:
            energies = tuple(
                complex(value["real"], value["imag"])
                for value in row["outgoing_energies"]
            )
            recorded = complex(
                row["locked_formula"]["real"],
                row["locked_formula"]["imag"],
            )
            self.assertAlmostEqual(v_to_vss_raw_candidate(*energies), recorded)

    def test_unit_descendant_normalizations(self) -> None:
        w1, w2, w3 = self.energies
        w0 = w1 + w2 + w3
        self.assertAlmostEqual(
            s_to_svv_unit_descendants_candidate(w1, w2, w3),
            s_to_svv_raw_candidate(w1, w2, w3)
            / (singlet_descendant_norm(w0) * singlet_descendant_norm(w1)),
        )
        self.assertAlmostEqual(
            v_to_vss_unit_descendants_candidate(w1, w2, w3),
            v_to_vss_raw_candidate(w1, w2, w3)
            / (singlet_descendant_norm(w2) * singlet_descendant_norm(w3)),
        )

    def test_v_to_vss_pole_plus_contact_energy_form(self) -> None:
        w1, w2, w3 = self.energies
        w0 = w1 + w2 + w3
        product = w0 * w1 * w2 * w3
        expected = math.pi * product * (
            1
            + 2j * w0
            + w2 * w3 / (1 + 1j * (w2 + w3))
        )
        self.assertAlmostEqual(v_to_vss_raw_candidate(w1, w2, w3), expected)

    def test_v_to_vss_locked_numerator_coefficients(self) -> None:
        w1, w2, w3 = self.energies
        x1, x2, x3 = (1j * w for w in (w1, w2, w3))
        v = x2 + x3
        p = x2 * x3
        x0 = x1 + v
        product = x0 * x1 * p
        denominator = 1 + v
        locked_numerator = (
            1 + 2 * x1 + 3 * v + 2 * x1 * v + 2 * v**2 - p
        )
        normalized = v_to_vss_raw_candidate(w1, w2, w3) / (
            math.pi * product
        )
        self.assertAlmostEqual(denominator * normalized, locked_numerator)
        self.assertAlmostEqual(
            locked_numerator,
            denominator * (1 + 2 * x0) - p,
        )

    def test_svv_pole_plus_contact_energy_form(self) -> None:
        w1, w2, w3 = self.energies
        w0 = w1 + w2 + w3
        product = w0 * w1 * w2 * w3
        expected = math.pi * product * (
            1
            + 2j * w0
            - w0 * w1 / (1 + 1j * (w2 + w3))
        )
        self.assertAlmostEqual(s_to_svv_raw_candidate(w1, w2, w3), expected)

    def test_svv_locked_numerator_coefficients(self) -> None:
        w1, w2, w3 = self.energies
        x1, x2, x3 = (1j * w for w in (w1, w2, w3))
        u = x1
        v = x2 + x3
        p = x2 * x3
        x0 = u + v
        product = x0 * x1 * x2 * x3
        denominator = 1 + v
        locked_numerator = (
            1 + 0 * p + 3 * v + 2 * v**2 + 2 * u + 3 * u * v + u**2
        )
        normalized = s_to_svv_raw_candidate(w1, w2, w3) / (math.pi * product)
        self.assertAlmostEqual(denominator * normalized, locked_numerator)
        self.assertAlmostEqual(
            locked_numerator,
            (1 + x0) * (1 + x0 + v),
        )

    def test_svv_fully_factorized_energy_form(self) -> None:
        w1, w2, w3 = self.energies
        w0 = w1 + w2 + w3
        product = w0 * w1 * w2 * w3
        expected = (
            math.pi
            * product
            * (1 + 1j * w0)
            * (1 + 1j * (2 * w0 - w1))
            / (1 + 1j * (w0 - w1))
        )
        self.assertAlmostEqual(s_to_svv_raw_candidate(w1, w2, w3), expected)

    def test_sss_pole_plus_contact_energy_form(self) -> None:
        w1, w2, w3 = self.energies
        w0 = w1 + w2 + w3
        product = w0 * w1 * w2 * w3
        pole_sum = sum(
            1 / (1 + 1j * (first + second))
            for first, second in ((w1, w2), (w1, w3), (w2, w3))
        )
        ward_polynomial = 1 + 0.5 * sum(w**2 for w in (w0, w1, w2, w3))
        expected = math.pi * product * (
            product * pole_sum + (1 + 2j * w0) * ward_polynomial
        )
        self.assertAlmostEqual(s_to_sss_raw_candidate(w1, w2, w3), expected)
        self.assertAlmostEqual(
            s_to_sss_unit_descendants_candidate(w1, w2, w3),
            s_to_sss_raw_candidate(w1, w2, w3)
            / np.prod([singlet_descendant_norm(w) for w in (w0, w1, w2, w3)]),
        )

    def test_sss_locked_symmetric_coefficients(self) -> None:
        w1, w2, w3 = self.energies
        x1, x2, x3 = (1j * w for w in (w1, w2, w3))
        x0 = x1 + x2 + x3
        s2 = x1 * x2 + x1 * x3 + x2 * x3
        product = x0 * x1 * x2 * x3
        pole_basis = product * sum(
            1 / (1 + first + second)
            for first, second in ((x1, x2), (x1, x3), (x2, x3))
        )
        locked = (
            pole_basis
            + 1
            + s2
            + 2 * x0
            + 2 * x0 * s2
            - x0**2
            - 2 * x0**3
        )
        normalized = s_to_sss_raw_candidate(w1, w2, w3) / (math.pi * product)
        self.assertAlmostEqual(normalized, locked)

    def test_ssvv_pair_pole_residue(self) -> None:
        # In x=i*omega variables, d23=epsilon and the normalized residue
        # tends to x0*x1, the product of the resonant three-point Ward factor.
        x1 = 0.17 + 0.09j
        x2 = -0.31 + 0.12j
        for epsilon in (1.0e-4, 1.0e-6, 1.0e-8):
            x3 = -1 - x2 + epsilon
            x0 = x1 + x2 + x3
            omegas = tuple(-1j * x for x in (x1, x2, x3))
            product = x0 * x1 * x2 * x3
            normalized = s_to_svv_raw_candidate(*omegas) / (np.pi * product)
            self.assertLess(abs(epsilon * normalized - x0 * x1), 20 * epsilon)

    def test_ssss_pair_pole_residue(self) -> None:
        # For d23 -> 0, M/(pi*Pi) has residue Pi.  This is the product of
        # the two resonant SSS three-point Ward factors.
        x1 = 0.17 + 0.09j
        x2 = -0.31 + 0.12j
        for epsilon in (1.0e-4, 1.0e-6, 1.0e-8):
            x3 = -1 - x2 + epsilon
            x0 = x1 + x2 + x3
            omegas = tuple(-1j * x for x in (x1, x2, x3))
            product = x0 * x1 * x2 * x3
            normalized = s_to_sss_raw_candidate(*omegas) / (np.pi * product)
            self.assertLess(abs(epsilon * normalized - product), 20 * epsilon)

    def test_v_to_vss_pair_pole_residue(self) -> None:
        # The fixed-incoming-vector residue differs from the S -> SVV one:
        # d23*M/(pi*Pi) tends to -x2*x3.
        x1 = 0.17 + 0.09j
        x2 = -0.31 + 0.12j
        for epsilon in (1.0e-4, 1.0e-6, 1.0e-8):
            x3 = -1 - x2 + epsilon
            x0 = x1 + x2 + x3
            omegas = tuple(-1j * x for x in (x1, x2, x3))
            product = x0 * x1 * x2 * x3
            normalized = v_to_vss_raw_candidate(*omegas) / (np.pi * product)
            self.assertLess(abs(epsilon * normalized + x2 * x3), 20 * epsilon)

    def test_v_to_vss_first_resonance_in_raw_block_convention(self) -> None:
        # The older normalized resonance helper chose the opposite overall
        # phase for the two external singlets.  In the raw block convention
        # used by spin23_singlet_amplitudes.py the sign is positive here.
        w2 = 0.23 + 0.11j
        w3 = -0.07 + 0.19j
        w1 = 1j - w2 - w3
        expected_raw = (
            math.pi
            * w2
            * w3
            * (1 + 1j * w2)
            * (1 + 1j * w3)
        )
        self.assertAlmostEqual(
            v_to_vss_raw_candidate(w1, w2, w3),
            expected_raw,
        )

    def test_v_to_vss_normalized_resonance_matches_legacy_up_to_phase(
        self,
    ) -> None:
        # Use imaginary energies for which sqrt(a*b)=sqrt(a)*sqrt(b) on the
        # principal branch.  The legacy helper explicitly omits the overall
        # common sign, and chose the negative of the raw-block convention.
        w1, w2, w3 = 0.2j, 0.3j, 0.5j
        actual = v_to_vss_unit_descendants_candidate(w1, w2, w3)
        legacy = complex(ssvv_resonance_coefficient(w1, w2, w3))
        self.assertAlmostEqual(actual, -legacy)


if __name__ == "__main__":
    unittest.main()
