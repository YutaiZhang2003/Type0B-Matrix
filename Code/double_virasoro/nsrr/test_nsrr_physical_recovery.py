"""Check the physical restriction that makes the singular convolution invertible."""

from fractions import Fraction
from itertools import product
import unittest

import sympy as sp

from nsrr_genus2_block import (
    auxiliary_majorana_nsrr_series,
    direct_pbw_nsrr_series,
    ramond_sector_residual,
    recover_same_structure_nsrr_series,
    star_convolve_series,
)
from ramond_pbw_generalized_ward import GeneralizedNRRWard, word_parity


class PhysicalRecoveryTests(unittest.TestCase):
    def test_exact_vertex_parity_exchange(self):
        # S w^+ = exp(-i pi/4) w^-, S w^- = exp(i pi/4) w^+.
        # K=(S P)_2 tensor S_3 has the eigenvalue used in the proof.
        phases = ((1 - sp.I) / sp.sqrt(2), (1 + sp.I) / sp.sqrt(2))
        c, h, beta2, beta3 = map(sp.Rational, ("27/2", "5/7", "2/7", "3/11"))
        for p, f, eta in product((0, 1), (0, 1), (-1, 1)):
            form = GeneralizedNRRWard(
                p_phi=p, form_parity=(p + f) % 2, eta=eta,
                h_ns=h, h_second=c / 24 - beta2**2,
                h_third=c / 24 - beta3**2,
                beta_second=beta2, beta_third=beta3, central_charge=c,
            )
            words = product(
                ((), (("G", -sp.Rational(1, 2)),)),
                ((), (("G", -sp.Integer(1)),)),
                ((), (("L", -sp.Integer(1)),)), (0, 1), (0, 1),
            )
            for w1, w2, w3, g2, g3 in words:
                transformed = (
                    (-1) ** (word_parity(w2) + g2) * phases[g2] * phases[g3]
                    * form.value(w1, w2, 1 - g2, w3, 1 - g3)
                )
                expected = (
                    -sp.I * (-1) ** (f + word_parity(w1)) * eta
                    * form.value(w1, w2, g2, w3, g3)
                )
                self.assertEqual(sp.simplify(transformed - expected), 0)

    def test_physical_sectors_and_recovery(self):
        auxiliary = auxiliary_majorana_nsrr_series(maximum_total_twice_level=2)
        for p, f, eta, eta_prime in product((0, 1), (0, 1), (-1, 1), (-1, 1)):
            physical = direct_pbw_nsrr_series(
                b=Fraction(7, 5),
                momenta=(Fraction(11, 23), Fraction(13, 29), Fraction(17, 31)),
                primary_parity=p, form_parity=f, etas=(eta, eta_prime),
                maximum_total_twice_level=2,
            )
            self.assertLess(ramond_sector_residual(physical, eta * eta_prime), 1e-12)
            enlarged = star_convolve_series(auxiliary, physical, maximum_total_twice_level=2)
            if eta == eta_prime:
                recovered = recover_same_structure_nsrr_series(
                    enlarged, auxiliary, maximum_total_twice_level=2, etas=(eta, eta_prime)
                )
                for levels in physical:
                    for a, b in zip(recovered[levels], physical[levels]):
                        self.assertLess(abs(a - b), 1e-12)
            else:
                self.assertLess(max(abs(x) for v in enlarged.values() for x in v), 1e-12)
                with self.assertRaisesRegex(ValueError, "opposite vertex signs"):
                    recover_same_structure_nsrr_series(
                        enlarged, auxiliary, maximum_total_twice_level=2, etas=(eta, eta_prime)
                    )

    def test_rejects_uncontrolled_sector_projection(self):
        auxiliary = {(0, 0, 0): (1, 0, 0, 0, 0, 0, 1, 0)}
        enlarged = {(0, 0, 0): (2.1, 0, 0, 0, 0, 0, 1.9, 0)}
        with self.assertRaisesRegex(ValueError, "violates the recoverable Ramond sector"):
            recover_same_structure_nsrr_series(enlarged, auxiliary, maximum_total_twice_level=0)


if __name__ == "__main__":
    unittest.main()
