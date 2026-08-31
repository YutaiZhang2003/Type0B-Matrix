"""Regression tests for the Human-convention NS--R--R PBW ingredients."""

from __future__ import annotations

import unittest

from nsrr_genus2_block import HumanAuxiliaryThreePoint


class NSRRGenusTwoTests(unittest.TestCase):
    def test_current_human_note_first_tube_signs_termwise(self) -> None:
        for primary_parity in (0, 1):
            for physical_parity in (0, 1):
                for auxiliary_parity in (0, 1):
                    enlarged_sign = (-1) ** (
                        physical_parity + auxiliary_parity
                    )
                    auxiliary_sign = (-1) ** auxiliary_parity
                    self.assertEqual(
                        enlarged_sign * auxiliary_sign,
                        (-1) ** physical_parity,
                    )

        # A Vir x Vir primary v_n^(p_1) has total first-leg parity
        # p_1+2n.  Every free-field component therefore obeys
        # A+mathsf A=2n, independently of p_1.
        for primary_parity in (0, 1):
            for twice_n in (0, 1):
                for physical_parity in (0, 1):
                    auxiliary_parity = (twice_n + physical_parity) % 2
                    total_parity = (
                        primary_parity
                        + physical_parity
                        + auxiliary_parity
                    ) % 2
                    self.assertEqual(
                        total_parity,
                        (primary_parity + twice_n) % 2,
                    )
                    self.assertEqual(
                        (-1) ** (physical_parity + auxiliary_parity),
                        (-1) ** twice_n,
                    )

    def test_auxiliary_u1_ground_value_follows_from_zero_mode_ward(self) -> None:
        # The ground-value rule does not act on descendant modules.
        form = HumanAuxiliaryThreePoint((None, None, None))
        rho_00 = form.base_value(((), ((), 0), ((), 0)))
        rho_11 = form.base_value(((), ((), 1), ((), 1)))
        zero_mode_coefficient = 1 / (2 ** 0.5)
        ward_residual = (
            zero_mode_coefficient * rho_11
            - 1j * zero_mode_coefficient * rho_00
        )
        self.assertEqual(rho_00, 1.0 + 0.0j)
        self.assertLess(abs(ward_residual), 1.0e-15)



if __name__ == "__main__":
    unittest.main()
