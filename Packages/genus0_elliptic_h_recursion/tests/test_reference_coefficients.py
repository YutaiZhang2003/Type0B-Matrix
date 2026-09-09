from __future__ import annotations

import json
import unittest
from pathlib import Path

import mpmath as mp

from genus0_elliptic_h_recursion import compute_h_recursion


HERE = Path(__file__).resolve().parent


class ReferenceCoefficientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reference = json.loads(
            (HERE / "reference_coefficients_order3.json").read_text(encoding="utf-8")
        )

    def check_case(
        self,
        name: str,
        central_charge: str,
        external_weights: tuple[str, ...],
        internal_weights: tuple[str, ...],
    ) -> None:
        table = compute_h_recursion(
            central_charge=central_charge,
            external_weights=external_weights,
            internal_weights=internal_weights,
            order=3,
            dps=70,
            pole_tolerance="1e-12",
        )
        expected = {
            tuple(int(part) for part in key.split(",")): mp.mpf(value)
            for key, value in self.reference[name].items()
        }
        self.assertEqual(set(table.coefficients), set(expected))
        for levels, target in expected.items():
            error = abs(table.coefficients[levels] - target)
            self.assertLess(
                error,
                mp.mpf("1e-52"),
                msg=f"{name} coefficient {levels}: error={mp.nstr(error, 12)}",
            )

    def test_four_point_snapshot(self) -> None:
        self.check_case(
            "four",
            "26.215",
            ("0.13", "0.27", "0.41", "0.56"),
            ("0.91",),
        )

    def test_five_point_snapshot(self) -> None:
        self.check_case(
            "five",
            "26.215",
            ("0.17", "0.29", "0.43", "0.58", "0.71"),
            ("0.93", "1.08"),
        )

    def test_six_point_snapshot(self) -> None:
        self.check_case(
            "six",
            "26.215",
            ("0.17", "0.29", "0.43", "0.58", "0.71", "0.86"),
            ("0.9371", "1.0837", "1.3321"),
        )


if __name__ == "__main__":
    unittest.main()

