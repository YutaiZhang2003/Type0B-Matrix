from __future__ import annotations

import unittest

from genus_2.glasses_partition import (
    glasses_diagonal_sector_contribution,
    glasses_null_transport,
    glasses_sector_pair,
)


class GlassesPartitionParityTests(unittest.TestCase):
    def test_type0b_sewing_is_even_minus_odd(self) -> None:
        self.assertEqual(glasses_sector_pair(0).sign, 1)
        self.assertEqual(glasses_sector_pair(1).sign, -1)
        self.assertEqual(
            glasses_diagonal_sector_contribution(
                sector=1,
                measure=2.0,
                structure_weight=3.0,
                primary_times_block=4.0j,
            ),
            -96.0,
        )

    def test_absolute_parity_depends_on_bridge_primary(self) -> None:
        pair = glasses_sector_pair(
            0,
            holomorphic_primary_parities=(1, 0, 1),
            antiholomorphic_primary_parities=(0, 1, 0),
        )
        self.assertEqual(pair.absolute_parity, 1)
        self.assertEqual(pair.antiholomorphic_sector, 1)
        self.assertEqual(pair.sign, -1)

    def test_odd_handle_null_flips_only_bridge_lift(self) -> None:
        transport = glasses_null_transport(
            sector=1,
            lifts=(-1, 1, -1),
            edge=0,
            rs=3,
        )
        self.assertEqual(transport.child_sector, 1)
        self.assertEqual(transport.child_lifts, (-1, 1, 1))
        self.assertEqual(transport.edge_character, -1)

    def test_odd_bridge_null_toggles_sector_without_lift_flip(self) -> None:
        transport = glasses_null_transport(
            sector=1,
            lifts=(-1, 1, -1),
            edge=2,
            rs=5,
        )
        self.assertEqual(transport.child_sector, 0)
        self.assertEqual(transport.child_lifts, (-1, 1, -1))
        self.assertEqual(transport.edge_character, -1)


if __name__ == "__main__":
    unittest.main()
