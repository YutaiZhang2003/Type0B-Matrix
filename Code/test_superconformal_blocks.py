"""Regression tests for the NS sphere four-point block implementation."""

import unittest

from superconformal_blocks import (
    NSSphereFourPointBlock,
    central_charge,
    elliptic_nome,
    ns_liouville_weight,
)


class NSSphereFourPointBlockTests(unittest.TestCase):
    def test_type_0b_liouville_parameters(self):
        self.assertAlmostEqual(central_charge(1.0).real, 13.5)
        self.assertAlmostEqual(central_charge(1.0).imag, 0.0)
        self.assertAlmostEqual(ns_liouville_weight(0.6, 1.0).real, 0.68)

    def test_endpoint_poles_are_included(self):
        block = NSSphereFourPointBlock.from_liouville_momenta(
            p1=0.4,
            p2=0.7,
            p3=0.3,
            p4=0.55,
            internal_momentum=0.8,
        )
        # The first allowed odd pole is (r,s)=(3,1) at 2m=3, and the
        # first allowed even pole is (2,2) at 2m=4.  Both sit precisely at
        # rs=2m and would be lost if the printed strict inequality were used.
        self.assertNotAlmostEqual(block.coefficient(3), block.seed_coefficient(3), places=10)
        self.assertNotAlmostEqual(block.coefficient(4), block.seed_coefficient(4), places=10)

    def test_all_eight_blocks_agree_in_z_and_q_representations(self):
        z = 0.02 + 0.01j
        for star2 in (False, True):
            for star3 in (False, True):
                block = NSSphereFourPointBlock.from_liouville_momenta(
                    p1=0.4,
                    p2=0.7,
                    p3=0.3,
                    p4=0.55,
                    internal_momentum=0.8,
                    star2=star2,
                    star3=star3,
                )
                for parity in ("even", "odd"):
                    with self.subTest(star2=star2, star3=star3, parity=parity):
                        local = block.z_block(z, order=8, parity=parity)
                        elliptic = block.elliptic_block(z, order=8, parity=parity)
                        self.assertLess(abs(local - elliptic), 2.0e-11)

    def test_bry_ancillary_notebook_benchmark(self):
        # BRY ancillary notebook: c=27/2+10^-5, momenta
        # (P4,P3,P2,P1)=(3/5,1/4,1/3,1/2), h=11/10, and W insertions
        # at punctures 3 and 2.  The small central-charge displacement avoids
        # taking an exact b->1 limit in the independent crossing check.
        c = 13.5 + 1.0e-5
        q_background = (c / 3.0 - 0.5) ** 0.5

        def weight(momentum):
            return 0.5 * (q_background * q_background / 4.0 + momentum * momentum)

        block = NSSphereFourPointBlock(
            c=c,
            h1=weight(1.0 / 2.0),
            h2=weight(1.0 / 3.0),
            h3=weight(1.0 / 4.0),
            h4=weight(3.0 / 5.0),
            internal_weight=1.1,
            star2=True,
            star3=True,
        )
        even_expected = [
            1.0,
            1.0998737373737373,
            15.824538587885476,
            21.09367085588721,
        ]
        odd_expected = [
            -1.7823926767676768,
            -6.738809221289842,
            -13.848138977969848,
            -20.488098285468837,
        ]
        even = block.elliptic_coefficients(4, "even")
        odd = block.elliptic_coefficients(4, "odd")
        for power, expected in enumerate(even_expected):
            self.assertAlmostEqual(even[2 * power].real, expected, places=10)
            self.assertAlmostEqual(even[2 * power].imag, 0.0, places=12)
        for power, expected in enumerate(odd_expected):
            self.assertAlmostEqual(odd[2 * power + 1].real, expected, places=10)
            self.assertAlmostEqual(odd[2 * power + 1].imag, 0.0, places=12)

        z = 1.0 / 3.0 + 3.0j / 5.0
        self.assertAlmostEqual(
            block.elliptic_block(z, 8, "even"),
            1.3109708952736965 + 0.18058411587390863j,
            places=11,
        )
        self.assertAlmostEqual(
            block.elliptic_block(z, 8, "odd"),
            -0.3051714889777302 - 0.4379494400515853j,
            places=11,
        )

    def test_nome_is_small_in_the_direct_channel(self):
        q = elliptic_nome(0.2)
        self.assertGreater(q.real, 0.0)
        self.assertLess(abs(q), 0.02)
        self.assertAlmostEqual(q.imag, 0.0, places=14)

    def test_bry_q_power_cutoff_is_parity_aware(self):
        block = NSSphereFourPointBlock.from_liouville_momenta(
            p1=0.4,
            p2=0.7,
            p3=0.3,
            p4=0.55,
            internal_momentum=0.8,
            star2=True,
            star3=True,
        )
        z = 0.2 + 0.1j
        self.assertAlmostEqual(
            block.bry_elliptic_block(z, 4, "even"),
            block.elliptic_block(z, 5, "even"),
            places=14,
        )
        self.assertAlmostEqual(
            block.bry_elliptic_block(z, 4, "odd"),
            block.elliptic_block(z, 4, "odd"),
            places=14,
        )


if __name__ == "__main__":
    unittest.main()
