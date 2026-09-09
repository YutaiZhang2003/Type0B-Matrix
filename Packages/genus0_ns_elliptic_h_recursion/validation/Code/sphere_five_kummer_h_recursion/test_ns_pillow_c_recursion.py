"""Small, independent regression checks for the literature comparison driver."""

import unittest

import mpmath as mp

from check_ns_pillow_c_recursion import (
    CachedLiteratureCRecursion, NSSphereLinearCRecursion, CASES,
    sector_labels, scaled_error,
)
from ns_pillow_elliptic_audit import NSEllipticRecursion, PillowMap, indices


class LiteratureComparisonTests(unittest.TestCase):
    def setUp(self):
        self.previous_dps = mp.mp.dps
        mp.mp.dps = 60
        bt, ds, hs = CASES["A"]
        self.b = mp.mpf(bt)
        self.c = mp.mpf("1.5") + 3*(self.b+1/self.b)**2
        self.d = tuple(map(mp.mpf, ds))
        self.h = tuple(map(mp.mpf, hs))
        self.common = dict(central_charge=self.c, external_weights=self.d,
                           internal_weights=self.h, working_precision=60)

    def tearDown(self):
        mp.mp.dps = self.previous_dps

    def test_cached_all_parity_entry_point_matches_unmodified_public_api(self):
        cached = CachedLiteratureCRecursion(vertex_sectors=(0,0,0,0), **self.common)
        blocks = {}
        for key in indices(3,6):
            labels = sector_labels(key)
            if labels not in blocks:
                blocks[labels] = NSSphereLinearCRecursion(vertex_sectors=labels, **self.common)
            self.assertEqual(cached.coefficient_for_parity(key), blocks[labels].coefficient(key))
        self.assertEqual(len(blocks), 8)

    def test_six_point_degree_three_comparison_all_parities(self):
        block = CachedLiteratureCRecursion(vertex_sectors=(0,0,0,0), **self.common)
        keys = list(indices(3,6))
        self.assertEqual(len(keys), 84)
        plane = {key:block.coefficient_for_parity(key) for key in keys}
        pulled = PillowMap(3,3).pullback(plane,self.c,self.d,self.h)
        recursion = NSEllipticRecursion(self.b,self.d,self.h)
        for key in keys:
            self.assertLess(scaled_error(pulled.get(key,0),recursion.coefficient(key)), mp.mpf("1e-48"))

    def test_global_seed_and_odd_phase_are_not_unit_seed(self):
        block = CachedLiteratureCRecursion(vertex_sectors=(0,0,0,0), **self.common)
        # No moving-c Kac pole exists below level 3/2: these are global
        # coefficients, including a nonzero odd term and level-one term.
        self.assertEqual(block.coefficient_for_parity((0,0,0)), 1)
        self.assertEqual(block.coefficient_for_parity((1,0,0)), -1/(2*self.h[0]))
        self.assertGreater(abs(block.coefficient_for_parity((2,0,0))), mp.mpf("1e-3"))
        for key in ((1,0,0),(2,0,0)):
            seed = block._global_coefficient(key, block.internal_weights, sector_labels(key))
            self.assertEqual(block.coefficient_for_parity(key), seed)

    def test_invalid_twice_levels_rejected(self):
        block = CachedLiteratureCRecursion(vertex_sectors=(0,0,0,0), **self.common)
        for key in ((1,2),(-1,0,0),(True,0,0),(1.5,0,0)):
            with self.assertRaises(ValueError):
                block.coefficient_for_parity(key)

    def test_missing_jacobian_conversion_is_detected(self):
        good = CachedLiteratureCRecursion(vertex_sectors=(0,0,0,0), **self.common)
        bad = CachedLiteratureCRecursion(vertex_sectors=(0,0,0,0), **self.common)
        original_residue = bad._edge_residue

        def wrong_residue(**kwargs):
            pole, scalar, sectors = original_residue(**kwargs)
            return pole, scalar*mp.mpf(2)/3, sectors

        bad._edge_residue = wrong_residue
        # First moving-c pole at level 3/2. Only its Jacobian is deliberately
        # left in BG units, while the denominator is in human-note units.
        key = (3,0,0)
        self.assertGreater(scaled_error(good.coefficient_for_parity(key),
                                       bad.coefficient_for_parity(key)), mp.mpf("1e-5"))


if __name__ == "__main__":
    unittest.main()
