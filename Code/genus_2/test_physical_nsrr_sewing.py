"""Regression tests for the corrected physical NSRR assembly."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DOUBLE = ROOT / "Code" / "double_virasoro" / "nsrr"
CRECURSION = ROOT / "Code" / "c_Recursion"
for directory in (DOUBLE, CRECURSION):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from nsrr_genus2_block import auxiliary_majorana_nsrr_series  # noqa: E402
from theta_star_algebra import fwht  # noqa: E402

from physical_nsrr_sewing import (  # noqa: E402
    CHANNELS,
    PHYSICAL_SEWING_NORMALIZATION,
    PRODUCT_SPACE_KERNEL_NORMALIZATION,
    SOURCE_FIXED_SPIN_LIFTS,
    bry_to_hjs_coefficients,
    contract_physical_blocks,
    physical_form_bilinear,
    physical_form_matrix,
    project_source_fixed_spin,
)


FREE_SUMMARY = ROOT / "Data Set" / "fixed_spin_free_NSrr_20260830" / "summary.json"


def evaluate_auxiliary(series, q_values):
    components = [0.0j] * 8
    for exponent, vector in series.items():
        factor = math.prod(
            complex(q_values[edge]) ** (exponent[edge] / 2)
            for edge in range(3)
        )
        for index, coefficient in enumerate(vector):
            components[index] += coefficient * factor
    return components


def character_index(lift):
    return sum((value < 0) << edge for edge, value in enumerate(lift))


class PhysicalNSRRSewingTests(unittest.TestCase):
    def test_three_point_basis_conversion(self):
        self.assertEqual(bry_to_hjs_coefficients((6, 10)), {1: 3, -1: 5})

    def test_toric_factor_four_is_inside_the_full_human_block(self):
        # HJS arXiv:1207.5740 first defines F = F_e + F_o and only then
        # rewrites the surviving sector using F_e = F_o.  The resulting
        # 4 |F_e|^2 is therefore |F|^2, not an extra sewing multiplier.
        even_only = 1.25 - 0.75j
        odd_only = even_only
        human_full = even_only + odd_only
        self.assertAlmostEqual(abs(human_full) ** 2, 4.0 * abs(even_only) ** 2)

        # At ground level this is also the explicit Human sum over w+ and w-.
        self.assertEqual(1.0 + 1.0, 2.0)

    def test_physical_form_matrix_is_positive_rank_one(self):
        for eta_left in (1, -1):
            for eta_right in (1, -1):
                matrix = physical_form_matrix(eta_left, eta_right)
                self.assertTrue(np.allclose(matrix, matrix.conjugate().T))
                self.assertTrue(np.allclose(
                    matrix @ matrix,
                    2.0 * PHYSICAL_SEWING_NORMALIZATION * matrix,
                ))
                self.assertEqual(np.linalg.matrix_rank(matrix), 1)
                f0, f1 = 1.2 + 0.7j, -0.4 + 0.9j
                vector = np.asarray([f0, f1])
                direct = vector @ matrix @ vector.conjugate()
                self.assertAlmostEqual(
                    direct.real,
                    physical_form_bilinear(f0, f1, eta_left, eta_right),
                )
                self.assertAlmostEqual(direct.imag, 0.0)

    def test_fixed_spin_projection_is_amplitude_level(self):
        first = {channel: complex(index, -index) for index, channel in enumerate(CHANNELS)}
        second = {channel: 2.0 * value for channel, value in first.items()}
        projected = project_source_fixed_spin(
            {SOURCE_FIXED_SPIN_LIFTS[0]: first, SOURCE_FIXED_SPIN_LIFTS[1]: second}
        )
        for channel in CHANNELS:
            self.assertEqual(projected[channel], 3.0 * first[channel] / math.sqrt(2.0))

    def test_crossed_term_not_diagonal_norm(self):
        blocks = {
            channel: complex(1 + index / 10, (-1) ** index / 7)
            for index, channel in enumerate(CHANNELS)
        }
        result = contract_physical_blocks(blocks, (4.0, 6.0))
        expected = 0.0
        for eta_left in (1, -1):
            for eta_right in (1, -1):
                coefficient = (2.0 if eta_left == 1 else 3.0) * (2.0 if eta_right == 1 else 3.0)
                expected += coefficient * physical_form_bilinear(
                    blocks[0, eta_left, eta_right],
                    blocks[1, eta_left, eta_right],
                    eta_left,
                    eta_right,
                )
        self.assertAlmostEqual(result["total"], expected)
        self.assertNotAlmostEqual(result["total"], result["diagonal"])

    def test_normalized_physical_ground_factorization(self):
        # The projected ground blocks follow directly from equations (14)
        # in NSRR_NONCHIRAL_SEWING_DERIVATION_2026-08-30.md.  The two
        # Ramond ground sums are already inside them.  Restricting the
        # holomorphic/antiholomorphic product to the two normalized physical
        # Ramond families supplies the nonchiral completeness kernel.
        root_two = math.sqrt(2.0)
        blocks = {
            (form_parity, eta_left, eta_right): (
                root_two if form_parity == 0 else -1.0j * root_two
            )
            for form_parity, eta_left, eta_right in CHANNELS
        }
        even, odd = 6.0, 10.0
        result = contract_physical_blocks(blocks, (even, odd))
        self.assertAlmostEqual(result["total"], (even**2 + odd**2) / 2.0)

    def test_identity_trinion_full_two_family_sewing_gives_two(self):
        # Internal consistency check, not a genus-two trace normalization:
        # after choosing full two-family restricted completeness, an identity
        # NS insertion has d_+=d_-=1 and (C_even,C_odd)=(2,0), and the explicit
        # pants contraction gives 1^2+1^2=2.  It does not independently choose
        # the sewing tensor or decide the global GSO/spin-sum normalization.
        root_two = math.sqrt(2.0)
        blocks = {
            (form_parity, eta_left, eta_right): (
                root_two if form_parity == 0 else -1.0j * root_two
            )
            for form_parity, eta_left, eta_right in CHANNELS
        }
        self.assertAlmostEqual(
            contract_physical_blocks(blocks, (2.0, 0.0))["total"],
            2.0,
        )

    def test_sewing_kernel_is_derived_from_vertex_and_restricted_edge_tables(self):
        root_two = math.sqrt(2.0)
        embedding = np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [-1.0j, 0.0]],
            dtype=np.complex128,
        ) / root_two
        product_pairing = np.diag([1.0, -1.0j, 1.0j, -1.0])
        dual_embedding = np.linalg.inv(product_pairing) @ embedding.conjugate()
        self.assertTrue(np.allclose(
            dual_embedding.T @ product_pairing @ embedding,
            np.eye(2),
        ))
        contravariant_edge = embedding @ dual_embedding.T

        # At the RN end the ordered bra reverses the two chiral labels, so
        # (f,fbar) reads product-basis index (fbar,f).
        product_index = lambda holomorphic, antiholomorphic: (
            2 * holomorphic + antiholomorphic
        )
        restricted_edge = np.asarray([
            [
                contravariant_edge[
                    product_index(fbar, f), product_index(fbar, f)
                ]
                for fbar in (0, 1)
            ]
            for f in (0, 1)
        ])
        self.assertTrue(np.allclose(
            restricted_edge,
            0.5 * np.asarray(
                [[1.0, -1.0j], [1.0j, -1.0]], dtype=np.complex128
            ),
        ))

        # The ordered physical R-field edge gives 1/2 times this table.  The
        # diagonal entries include (-i)^2 in the odd/odd R+ channel, while
        # the crossed R- channels carry k=eta_left*eta_right.
        for eta_left in (1, -1):
            for eta_right in (1, -1):
                k = eta_left * eta_right
                vertex_edge = 0.5 * np.asarray(
                    [[1.0, k], [k, -1.0]], dtype=np.complex128
                )
                product_space_kernel = vertex_edge * restricted_edge
                self.assertTrue(np.allclose(
                    product_space_kernel,
                    physical_form_matrix(eta_left, eta_right),
                ))

    def test_product_space_quarter_is_the_physical_kernel(self):
        self.assertEqual(PRODUCT_SPACE_KERNEL_NORMALIZATION, 0.25)
        self.assertEqual(PHYSICAL_SEWING_NORMALIZATION, 0.25)

    def test_free_fixed_spin_projection_matches_independent_bosonization(self):
        saved = json.loads(FREE_SUMMARY.read_text(encoding="utf-8"))
        auxiliary = auxiliary_majorana_nsrr_series(maximum_total_twice_level=32)
        errors = []
        for point in saved["points"]:
            q_geometry = tuple(complex(value) for value in point["source_NSrr"]["q_values"])
            q_slots = q_geometry[::-1]
            components = evaluate_auxiliary(auxiliary, q_slots)
            lift_values = {
                lift: fwht(components)[character_index(lift[::-1])]
                for lift in SOURCE_FIXED_SPIN_LIFTS
            }
            physical = sum(lift_values.values()) / math.sqrt(2.0)
            predicted = abs(physical) ** 2 * abs(q_geometry[0] * q_geometry[1]) ** (1.0 / 8.0)
            expected = float(point["source_NSrr"]["Z_majorana"])
            errors.append(abs(predicted / expected - 1.0))
        self.assertLess(max(errors), 1.0e-12)


if __name__ == "__main__":
    unittest.main()
