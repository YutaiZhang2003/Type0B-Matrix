"""Tests for the constant-size pointwise theta star algebra."""

from __future__ import annotations

import unittest

from theta_star_algebra import (
    direct_star_multiply,
    star_divide,
    star_multiply,
)


class ThetaStarAlgebraTests(unittest.TestCase):
    def test_hadamard_product_matches_direct_cocycle(self) -> None:
        left = [complex(index + 1, 0.2 * index) for index in range(8)]
        right = [complex(2 - index, -0.1 * index) for index in range(8)]
        observed = star_multiply(left, right)
        expected = direct_star_multiply(left, right)
        self.assertLess(max(abs(a - b) for a, b in zip(observed, expected)), 1e-12)

    def test_star_division_round_trip(self) -> None:
        quotient = [complex(0.1 * (index + 1), -0.03 * index) for index in range(8)]
        denominator = [1.0 + 0.0j] + [0.01 * (index + 1) for index in range(7)]
        numerator = star_multiply(denominator, quotient)
        recovered = star_divide(numerator, denominator)
        self.assertLess(
            max(abs(a - b) for a, b in zip(recovered, quotient)),
            1e-12,
        )


if __name__ == "__main__":
    unittest.main()
