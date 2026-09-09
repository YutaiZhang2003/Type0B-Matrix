"""Checks for the fixed-beta Ramond-channel sphere c-recursion."""

from __future__ import annotations

import unittest

from mixed_ramond_sphere_blocks import MixedRExchangeSphereFourPointBlock
from ramond_fixed_beta_c_recursion import (
    FixedBetaRExchangeSphereFourPointBlock,
    normalize_hjs_series,
    ramond_c_poles,
    ramond_large_c_seed_candidate_series,
    ramond_large_c_seed_series,
)
from superconformal_blocks import central_charge


class FixedBetaRamondCRecursionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.b = 1.37
        cls.c = central_charge(cls.b)
        cls.h1 = 0.37
        cls.h4 = 0.43
        cls.beta2 = 0.19j
        cls.beta3 = 0.27j
        cls.beta = 0.41j

    def block(
        self, sign3: int, sign2: int
    ) -> FixedBetaRExchangeSphereFourPointBlock:
        return FixedBetaRExchangeSphereFourPointBlock(
            c=self.c,
            h1_ns=self.h1,
            beta2_r=self.beta2,
            beta3_r=self.beta3,
            h4_ns=self.h4,
            internal_beta=self.beta,
            sign3=sign3,
            sign2=sign2,
        )

    def reference(
        self, b: complex, sign3: int, sign2: int
    ) -> MixedRExchangeSphereFourPointBlock:
        return MixedRExchangeSphereFourPointBlock.from_fixed_data(
            b=b,
            h1_ns=self.h1,
            beta2_r=self.beta2,
            beta3_r=self.beta3,
            h4_ns=self.h4,
            internal_beta=self.beta,
            sign3=sign3,
            sign2=sign2,
        )

    def test_two_c_poles_are_level_one_gram_zeros(self) -> None:
        for pole in ramond_c_poles(self.beta, 2, 1):
            h = pole.c / 24.0 - self.beta * self.beta
            kappa_squared = -self.beta * self.beta
            gram00 = 2.0 * h
            gram01 = 1.5 * kappa_squared
            gram11 = kappa_squared * (2.0 * h + pole.c / 4.0)
            determinant = gram00 * gram11 - gram01 * gram01
            self.assertLess(abs(determinant), 1.0e-12)

    def test_level_one_reconstructs_hjs_for_all_component_signs(self) -> None:
        for sign3 in (1, -1):
            for sign2 in (1, -1):
                with self.subTest(sign3=sign3, sign2=sign2):
                    recursive = self.block(sign3, sign2)
                    reference = self.reference(self.b, sign3, sign2)
                    raw = [
                        reference.elliptic_coefficients(2)[power]
                        for power in range(2)
                    ]
                    normalized = normalize_hjs_series(raw, self.c)
                    self.assertAlmostEqual(
                        recursive.coefficient(1), normalized[1]
                    )
                    self.assertAlmostEqual(
                        recursive.raw_coefficients(2)[1], raw[1]
                    )

    def test_level_two_reconstructs_hjs_for_all_component_signs(self) -> None:
        for sign3 in (1, -1):
            for sign2 in (1, -1):
                with self.subTest(sign3=sign3, sign2=sign2):
                    recursive = self.block(sign3, sign2)
                    reference = self.reference(self.b, sign3, sign2)
                    raw = [
                        reference.elliptic_coefficients(3)[power]
                        for power in range(3)
                    ]
                    normalized = normalize_hjs_series(raw, self.c)
                    self.assertAlmostEqual(
                        recursive.coefficient(2), normalized[2]
                    )
                    self.assertAlmostEqual(
                        recursive.raw_coefficients(3)[2], raw[2]
                    )

    def test_closed_seed_reproduces_the_known_first_two_terms(self) -> None:
        seed = ramond_large_c_seed_series(
            max_power=2,
            internal_beta=self.beta,
            beta2_r=self.beta2,
            beta3_r=self.beta3,
            h1_ns=self.h1,
            h4_ns=self.h4,
        )
        difference = (
            self.beta * self.beta
            - self.beta2 * self.beta2
            - self.beta3 * self.beta3
            - self.h1
            - self.h4
        )
        expected_one = 8.0 * difference - 0.5
        expected_two = (
            32.0 * difference * difference
            - 16.0 * self.beta * self.beta
            + 12.0 * self.beta2 * self.beta2
            + 12.0 * self.beta3 * self.beta3
            - 4.0 * self.h1
            - 4.0 * self.h4
            - 3.0 / 8.0
        )
        self.assertAlmostEqual(seed[0], 1.0)
        self.assertAlmostEqual(seed[1], expected_one)
        self.assertAlmostEqual(seed[2], expected_two)

    def test_predicted_level_one_residues(self) -> None:
        displacement = 1.0e-6
        for sign3 in (1, -1):
            for sign2 in (1, -1):
                recursive = self.block(sign3, sign2)
                for pole in ramond_c_poles(self.beta, 2, 1):
                    predicted = recursive._residue_multiplier(
                        pole, sign3, sign2
                    )
                    samples = []
                    for direction in (1.0, -1.0):
                        b = pole.b + direction * displacement
                        reference = self.reference(b, sign3, sign2)
                        raw = [
                            reference.elliptic_coefficients(2)[power]
                            for power in range(2)
                        ]
                        normalized = normalize_hjs_series(raw, reference.c)
                        samples.append(
                            (reference.c - pole.c) * normalized[1]
                        )
                    measured = 0.5 * (samples[0] + samples[1])
                    relative_error = abs(predicted - measured) / max(
                        abs(measured), 1.0e-300
                    )
                    with self.subTest(
                        sign3=sign3,
                        sign2=sign2,
                        branch=pole.branch,
                    ):
                        self.assertLess(relative_error, 2.0e-9)

    def test_level_two_residue_contains_shifted_ramond_tail(self) -> None:
        displacement = 1.0e-6
        for sign3 in (1, -1):
            for sign2 in (1, -1):
                recursive = self.block(sign3, sign2)
                for pole in ramond_c_poles(self.beta, 2, 1):
                    predicted = recursive._residue_multiplier(
                        pole, sign3, sign2
                    ) * recursive._coefficient(
                        1,
                        pole.beta_prime,
                        pole.c,
                        sign3,
                        sign2,
                    )
                    samples = []
                    for direction in (1.0, -1.0):
                        b = pole.b + direction * displacement
                        reference = self.reference(b, sign3, sign2)
                        raw = [
                            reference.elliptic_coefficients(3)[power]
                            for power in range(3)
                        ]
                        normalized = normalize_hjs_series(raw, reference.c)
                        samples.append(
                            (reference.c - pole.c) * normalized[2]
                        )
                    measured = 0.5 * (samples[0] + samples[1])
                    relative_error = abs(predicted - measured) / max(
                        abs(measured), 1.0e-300
                    )
                    with self.subTest(
                        sign3=sign3,
                        sign2=sign2,
                        branch=pole.branch,
                    ):
                        self.assertLess(relative_error, 1.0e-8)

    def test_closed_seed_completes_q3_and_q4_recursion(self) -> None:
        for sign3 in (1, -1):
            for sign2 in (1, -1):
                with self.subTest(sign3=sign3, sign2=sign2):
                    def candidate(
                        power: int,
                        internal_beta: complex,
                        _sign3: int,
                        _sign2: int,
                    ) -> complex:
                        return ramond_large_c_seed_series(
                            max_power=power,
                            internal_beta=internal_beta,
                            beta2_r=self.beta2,
                            beta3_r=self.beta3,
                            h1_ns=self.h1,
                            h4_ns=self.h4,
                        )[power]

                    recursive = FixedBetaRExchangeSphereFourPointBlock(
                        c=self.c,
                        h1_ns=self.h1,
                        beta2_r=self.beta2,
                        beta3_r=self.beta3,
                        h4_ns=self.h4,
                        internal_beta=self.beta,
                        sign3=sign3,
                        sign2=sign2,
                        regular_seed=candidate,
                    )
                    reference = self.reference(self.b, sign3, sign2)
                    raw = [
                        reference.elliptic_coefficients(5)[power]
                        for power in range(5)
                    ]
                    normalized = normalize_hjs_series(raw, self.c)
                    self.assertAlmostEqual(
                        recursive.coefficient(3), normalized[3]
                    )
                    self.assertAlmostEqual(
                        recursive.coefficient(4), normalized[4]
                    )

    def test_closed_seed_is_the_default_at_higher_orders(self) -> None:
        reference = self.reference(self.b, 1, 1)
        raw = [
            reference.elliptic_coefficients(5)[power]
            for power in range(5)
        ]
        normalized = normalize_hjs_series(raw, self.c)
        recursive = self.block(1, 1)
        self.assertAlmostEqual(recursive.coefficient(3), normalized[3])
        self.assertAlmostEqual(recursive.coefficient(4), normalized[4])

    def test_candidate_name_remains_a_compatibility_alias(self) -> None:
        parameters = dict(
            max_power=4,
            internal_beta=self.beta,
            beta2_r=self.beta2,
            beta3_r=self.beta3,
            h1_ns=self.h1,
            h4_ns=self.h4,
        )
        self.assertEqual(
            ramond_large_c_seed_candidate_series(**parameters),
            ramond_large_c_seed_series(**parameters),
        )


if __name__ == "__main__":
    unittest.main()
