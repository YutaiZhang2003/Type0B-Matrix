"""Tests for the parity-correct Type-0B theta partition assembly."""

from __future__ import annotations

from itertools import product
import unittest

from genus_2.theta_partition import (
    theta_diagonal_sector_contribution,
    theta_null_transport,
    theta_partition_term,
    theta_sector_pair,
)


class ThetaPartitionParityTests(unittest.TestCase):
    def test_sector_pair_satisfies_locality_for_every_primary_parity(self) -> None:
        for parities in product((0, 1), repeat=3):
            for anti_parities in product((0, 1), repeat=3):
                for sector in (0, 1):
                    pair = theta_sector_pair(
                        sector,
                        holomorphic_primary_parities=parities,
                        antiholomorphic_primary_parities=anti_parities,
                    )
                    self.assertEqual(
                        (sector + sum(parities)) % 2,
                        (
                            pair.antiholomorphic_sector
                            + sum(anti_parities)
                        )
                        % 2,
                    )
                    self.assertEqual(
                        pair.sign, -1 if pair.absolute_parity else 1
                    )

    def test_type0b_even_primaries_subtract_the_odd_sector(self) -> None:
        even = theta_diagonal_sector_contribution(
            sector=0,
            measure=2.0,
            structure_weight=3.0,
            primary_times_block=4.0j,
        )
        odd = theta_diagonal_sector_contribution(
            sector=1,
            measure=2.0,
            structure_weight=3.0,
            primary_times_block=4.0j,
        )
        self.assertEqual(even, 96.0)
        self.assertEqual(odd, -96.0)

    def test_intrinsic_primary_parity_changes_the_partition_sign(self) -> None:
        observed = theta_diagonal_sector_contribution(
            sector=0,
            measure=1.0,
            structure_weight=1.0,
            primary_times_block=1.0,
            primary_parities=(1, 0, 0),
        )
        self.assertEqual(observed, -1.0)

    def test_general_term_rejects_wrong_antiholomorphic_sector(self) -> None:
        with self.assertRaisesRegex(ValueError, "violates"):
            theta_partition_term(
                holomorphic_sector=0,
                antiholomorphic_sector=0,
                holomorphic_primary_parities=(1, 0, 0),
                antiholomorphic_primary_parities=(0, 0, 0),
                structure_weight=1.0,
                holomorphic_primary_factor=1.0,
                holomorphic_block=1.0,
                antiholomorphic_primary_factor=1.0,
                antiholomorphic_block=1.0,
            )

    def test_odd_null_flips_sector_and_spectator_lifts(self) -> None:
        odd = theta_null_transport(
            sector=0, lifts=(1, -1, 1), edge=1, rs=3
        )
        self.assertEqual(odd.child_sector, 1)
        self.assertEqual(odd.child_lifts, (-1, -1, -1))
        self.assertEqual(odd.edge_character, -1)

        even = theta_null_transport(
            sector=1, lifts=(1, -1, 1), edge=1, rs=4
        )
        self.assertEqual(even.child_sector, 1)
        self.assertEqual(even.child_lifts, (1, -1, 1))
        self.assertEqual(even.edge_character, 1)


if __name__ == "__main__":
    unittest.main()
