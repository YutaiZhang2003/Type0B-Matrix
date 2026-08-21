"""Regression tests for the all-NS two-Virasoro fusion coefficient."""

import itertools
import unittest

import mpmath

from two_virasoro_fusion import (
    blow_up_factor,
    branch_norm,
    ns_fusion_coefficient_squared,
    ns_fusion_data,
    s_even,
    s_odd,
)
from ns_human_convention import (
    enlarged_ns_three_form_crossing_sign,
    primary_parity_ward_sign,
)


class TwoVirasoroFusionTests(unittest.TestCase):
    precision = 70

    def assert_mp_close(self, actual, expected, tolerance="1e-60"):
        self.assertLess(
            abs(actual - expected),
            mpmath.mpf(tolerance),
            msg=f"{actual!r} != {expected!r}",
        )

    def test_elementary_s_products_and_negative_continuation(self):
        with mpmath.workdps(self.precision):
            b = mpmath.mpf("1.3")
            q = b + 1 / b
            x = mpmath.mpf("0.47")
            self.assert_mp_close(
                s_even(x, 0, b, precision=self.precision),
                1,
            )
            self.assert_mp_close(
                s_odd(x, 0, b, precision=self.precision),
                1,
            )
            self.assert_mp_close(
                s_even(x, 1, b, precision=self.precision),
                x / mpmath.sqrt(2),
            )
            self.assert_mp_close(
                s_even(x, -2, b, precision=self.precision),
                s_even(q - x, 2, b, precision=self.precision),
            )
            self.assert_mp_close(
                s_even(x, -1, b, precision=self.precision),
                -s_even(q - x, 1, b, precision=self.precision),
            )
            self.assert_mp_close(
                s_odd(x, -2, b, precision=self.precision),
                s_odd(q - x, 2, b, precision=self.precision),
            )

    def test_low_level_blow_up_products(self):
        with mpmath.workdps(self.precision):
            b = mpmath.mpf("1.2")
            q = b + 1 / b
            alpha = mpmath.mpf("0.83")
            p_prime = mpmath.mpf("0.31")
            p = mpmath.mpf("-0.27")
            common = dict(
                alpha=alpha,
                p_prime=p_prime,
                p=p,
                b=b,
                precision=self.precision,
            )

            self.assert_mp_close(
                blow_up_factor(m=0, k_prime=0, k=0, **common),
                1,
            )
            self.assert_mp_close(
                blow_up_factor(m=1, k_prime=0, k=0, **common),
                1,
            )
            self.assert_mp_close(
                blow_up_factor(m=0, k_prime=0, k=1, **common),
                1,
            )
            self.assert_mp_close(
                blow_up_factor(m=0, k_prime=1, k=0, **common),
                1,
            )
            self.assert_mp_close(
                blow_up_factor(m=0, k_prime=1, k=1, **common),
                (
                    (alpha + p_prime + p)
                    * (alpha - p_prime - p - q)
                    / 2
                ),
            )
            self.assert_mp_close(
                blow_up_factor(m=1, k_prime=1, k=0, **common),
                (
                    (alpha + p_prime + p)
                    * (alpha + p_prime - p)
                    / 2
                ),
            )
            self.assert_mp_close(
                blow_up_factor(m=1, k_prime=0, k=1, **common),
                (
                    (alpha + p_prime + p)
                    * (alpha - p_prime + p)
                    / 2
                ),
            )

    def test_norm_is_identity_specialization_and_level_half_anchor(self):
        with mpmath.workdps(self.precision):
            b = mpmath.mpf("0.91")
            q = b + 1 / b
            p = mpmath.mpf("0.37")
            for k in range(-4, 5):
                direct = branch_norm(p, k, b, precision=self.precision)
                identity = blow_up_factor(
                    0,
                    0,
                    p,
                    k,
                    p,
                    k,
                    b,
                    precision=self.precision,
                )
                self.assert_mp_close(direct, identity)

            self.assert_mp_close(
                branch_norm(p, 1, b, precision=self.precision),
                -p * (q + 2 * p),
            )
            self.assert_mp_close(
                branch_norm(p, -1, b, precision=self.precision),
                p * (q - 2 * p),
            )

    def test_human_note_trinion_order_and_normalization(self):
        with mpmath.workdps(self.precision):
            b = mpmath.mpf("1.17")
            q = b + 1 / b
            p1 = mpmath.mpf("0.21")
            p2 = mpmath.mpf("-0.34")
            p3 = mpmath.mpf("0.46")
            h1 = (q**2 / 4 - p1**2) / 2
            h2 = (q**2 / 4 - p2**2) / 2
            h3 = (q**2 / 4 - p3**2) / 2

            vacuum = ns_fusion_data(
                b=b,
                p1=p1,
                p2=p2,
                p3=p3,
                k1=0,
                k2=0,
                k3=0,
                precision=self.precision,
            )
            self.assertEqual(vacuum.parity, 0)
            self.assert_mp_close(vacuum.numerator, 1)
            self.assert_mp_close(vacuum.coefficient_squared, 1)

            one_slot3_branch = ns_fusion_data(
                b=b,
                p1=p1,
                p2=p2,
                p3=p3,
                k1=0,
                k2=0,
                k3=1,
                precision=self.precision,
            )
            self.assertEqual(one_slot3_branch.parity, 1)
            self.assert_mp_close(one_slot3_branch.numerator, -1)
            self.assert_mp_close(
                one_slot3_branch.coefficient_squared,
                1 / branch_norm(p3, 1, b, precision=self.precision),
            )

            gamma1 = q / 2 + p1
            gamma2 = q / 2 - p2
            gamma3 = q / 2 + p3

            two_slot12 = ns_fusion_data(
                b=b,
                p1=p1,
                p2=p2,
                p3=p3,
                k1=1,
                k2=-1,
                k3=0,
                precision=self.precision,
            )
            self.assert_mp_close(
                two_slot12.numerator,
                h1 + h2 - h3 - gamma1 * gamma2,
            )

            two_slot13 = ns_fusion_data(
                b=b,
                p1=p1,
                p2=p2,
                p3=p3,
                k1=1,
                k2=0,
                k3=1,
                precision=self.precision,
            )
            self.assert_mp_close(
                two_slot13.numerator,
                h1 - h2 + h3 - gamma1 * gamma3,
            )

            two_slot23 = ns_fusion_data(
                b=b,
                p1=p1,
                p2=p2,
                p3=p3,
                k1=0,
                k2=-1,
                k3=1,
                precision=self.precision,
            )
            self.assert_mp_close(
                two_slot23.numerator,
                h1 - h2 - h3 + gamma2 * gamma3,
            )

            three_slots = ns_fusion_data(
                b=b,
                p1=p1,
                p2=p2,
                p3=p3,
                k1=1,
                k2=-1,
                k3=1,
                precision=self.precision,
            )
            expected_human = (
                -(h1 + h2 + h3 - mpmath.mpf("0.5"))
                + gamma1 * gamma2
                + gamma1 * gamma3
                + gamma2 * gamma3
            )
            self.assert_mp_close(three_slots.numerator, expected_human)
            self.assert_mp_close(
                three_slots.coefficient_squared,
                ns_fusion_coefficient_squared(
                    b=b,
                    p1=p1,
                    p2=p2,
                    p3=p3,
                    k1=1,
                    k2=-1,
                    k3=1,
                    precision=self.precision,
                ),
            )
            self.assert_mp_close(
                three_slots.principal_coefficient**2,
                three_slots.coefficient_squared,
                tolerance="1e-58",
            )

            paper_ratio = blow_up_factor(
                q / 2 + p2,
                -1,
                p1,
                1,
                p3,
                1,
                b,
                precision=self.precision,
            )
            self.assert_mp_close(three_slots.numerator, paper_ratio)

    def test_all_first_primary_numerators_from_ungraded_components(self):
        """Check all 27 labels and all primary parities directly."""

        with mpmath.workdps(self.precision):
            b = mpmath.mpf("1.23")
            q = b + 1 / b
            momenta = (
                mpmath.mpf("0.19"),
                mpmath.mpf("-0.31"),
                mpmath.mpf("0.43"),
            )
            weights = tuple(
                (q**2 / 4 - momentum**2) / 2
                for momentum in momenta
            )

            fermion_rho = {
                (0, 0, 0): 1,
                (1, 1, 0): -1,
                (1, 0, 1): -1,
                (0, 1, 1): 1,
            }
            h1, h2, h3 = weights
            sca_rho = {
                (0, 0, 0): 1,
                (1, 1, 0): h1 + h2 - h3,
                (1, 0, 1): h1 - h2 + h3,
                (0, 1, 1): h1 - h2 - h3,
                (1, 0, 0): 1,
                (0, 1, 0): 1,
                (0, 0, 1): -1,
                (1, 1, 1): -(h1 + h2 + h3 - mpmath.mpf("0.5")),
            }

            changed_squares = 0
            for primary_parities in itertools.product((0, 1), repeat=3):
                for labels in itertools.product((-1, 0, 1), repeat=3):
                    gammas = tuple(
                        q / 2 + label * momentum if label else 0
                        for label, momentum in zip(labels, momenta)
                    )
                    components = tuple(
                        (((0, 0, 1),) if label == 0 else
                         ((0, 1, 1), (1, 0, gamma)))
                        for label, gamma in zip(labels, gammas)
                    )
                    direct = mpmath.mpf("0")
                    for terms in itertools.product(*components):
                        fermion_parities = tuple(term[0] for term in terms)
                        sca_parities = tuple(term[1] for term in terms)
                        direct += (
                            terms[0][2]
                            * terms[1][2]
                            * terms[2][2]
                            * enlarged_ns_three_form_crossing_sign(
                                sca_parities,
                                fermion_parities,
                                primary_parities,
                            )
                            * fermion_rho.get(fermion_parities, 0)
                            * primary_parity_ward_sign(
                                sca_parities, primary_parities
                            )
                            * sca_rho[sca_parities]
                        )

                    data = ns_fusion_data(
                        b=b,
                        p1=momenta[0],
                        p2=momenta[1],
                        p3=momenta[2],
                        k1=labels[0],
                        k2=labels[1],
                        k3=labels[2],
                        primary_parities=primary_parities,
                        precision=self.precision,
                    )
                    even_data = ns_fusion_data(
                        b=b,
                        p1=momenta[0],
                        p2=momenta[1],
                        p3=momenta[2],
                        k1=labels[0],
                        k2=labels[1],
                        k3=labels[2],
                        precision=self.precision,
                    )
                    self.assert_mp_close(data.numerator, direct)
                    square_difference = abs(
                        data.coefficient_squared
                        - even_data.coefficient_squared
                    )
                    if primary_parities[0] == 0:
                        self.assertLess(square_difference, mpmath.mpf("1e-60"))
                    elif square_difference > mpmath.mpf("1e-60"):
                        changed_squares += 1
            self.assertEqual(changed_squares, 64)

    def test_higher_human_branch_labels_are_not_silently_imported(self):
        with self.assertRaises(NotImplementedError):
            ns_fusion_data(
                b=1.2,
                p1=0.1,
                p2=0.2,
                p3=0.3,
                k1=2,
                k2=0,
                k3=0,
            )

    def test_branch_labels_must_be_exact_integers(self):
        with self.assertRaises(TypeError):
            branch_norm(0.3, 1.0, 1.2)
        with self.assertRaises(TypeError):
            blow_up_factor(0.4, 0, 0.2, 0, 0.1, True, 1.2)
        with self.assertRaises(TypeError):
            ns_fusion_data(
                b=1.2,
                p1=0.1,
                p2=0.2,
                p3=0.3,
                k1=0,
                k2=0.5,
                k3=0,
            )

    def test_zero_norm_is_reported_as_a_singularity(self):
        with self.assertRaises(ZeroDivisionError):
            ns_fusion_data(
                b=1.2,
                p1=0,
                p2=0.2,
                p3=0.3,
                k1=1,
                k2=0,
                k3=0,
            )

    def test_self_dual_complex_momenta_are_supported(self):
        with mpmath.workdps(self.precision):
            data = ns_fusion_data(
                b=1,
                p1=mpmath.mpc(0, "0.23"),
                p2=mpmath.mpc(0, "0.37"),
                p3=mpmath.mpc(0, "0.41"),
                k1=1,
                k2=-1,
                k3=1,
                precision=self.precision,
            )
            self.assertEqual(data.parity, 1)
            for value in (
                data.numerator,
                data.slot1_norm,
                data.slot2_norm,
                data.slot3_norm,
                data.coefficient_squared,
                data.principal_coefficient,
            ):
                self.assertTrue(mpmath.isfinite(value))

            from_strings = ns_fusion_data(
                b="1",
                p1="0.23j",
                p2="0.37i",
                p3="0+0.41j",
                k1=1,
                k2=-1,
                k3=1,
                precision=self.precision,
            )
            self.assert_mp_close(
                from_strings.coefficient_squared,
                data.coefficient_squared,
            )


if __name__ == "__main__":
    unittest.main()
