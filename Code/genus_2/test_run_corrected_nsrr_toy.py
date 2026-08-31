"""Actual NSRR toy-node audit against the unchanged PBW reference."""
import copy
from pathlib import Path
import unittest

import sympy as sp

import run_corrected_nsrr_toy as toy
from nsrr_genus2_block import HumanNSRRThetaOracle, level_triples
from theta_star_algebra import fwht


class CorrectedToyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[2]
        cls.config = toy.make_config(root/"Data Set/nsnsns_recompute_fivepoint_R16_N5_20260830/source_geometry_audit.json")
        cls.node = toy.evaluate_node(cls.config, 0)

    def test_production_has_no_pbw_or_physical_partition_claim(self):
        self.assertEqual(self.node["PBW_production_calls"], 0)
        self.assertIsNone(self.config["physical_Z"])
        self.assertIsNone(self.config["physical_Q"])
        self.assertEqual(len(self.node["values"]), 60)
        self.assertEqual(self.node["momenta_slots"], self.node["momenta_geometry"][::-1])
        self.assertEqual(len(toy.tasks(self.config)), 35)

    def test_reject_missing_or_mislabelled_node(self):
        altered = copy.deepcopy(self.node)
        altered["values"].pop()
        with self.assertRaises(ValueError):
            toy.validate_shard(self.config, 0, altered)
        altered = copy.deepcopy(self.node)
        altered["PBW_production_calls"] = 1
        with self.assertRaises(ValueError):
            toy.validate_shard(self.config, 0, altered)
        with self.assertRaises(ValueError):
            toy.node_data(self.config, -1)

    def test_real_grid_node_matches_independent_pbw(self):
        b = sp.Rational(str(self.config["b"]))
        bg = b+1/b
        p = tuple(sp.Rational(str(x)) for x in self.node["momenta_slots"])
        for channel, (f, eta) in enumerate(toy.CHANNELS):
            oracle = HumanNSRRThetaOracle(
                central_charge=sp.Rational(3, 2)+3*bg**2,
                h_ns=bg**2/8+p[0]**2/2,
                beta_r1=sp.I*p[1]/sp.sqrt(2), beta_r2=sp.I*p[2]/sp.sqrt(2),
                form_parity=f, primary_parity=0, etas=(eta, eta))
            vectors = {e: oracle.coefficient_components(e[0], e[1]//2, e[2]//2)
                       for e in level_triples(4)}
            for row in self.node["values"]:
                point = next(p for p in self.config["points"] if p["t"] == row["t"])
                plumbing = toy.NSRRPlumbingInputs(tuple(complex(q) for q in point["q_geometry"]),
                                                  tuple(row["lifts_geometry"]), toy.GEOMETRY_SECTORS)
                character = toy.dv.spin_character_index(plumbing.lifts_slots)
                expected = toy.dv.evaluate_twice_level_series(
                    {e: fwht(v)[character] for e, v in vectors.items() if sum(e) <= 2*row["level"]},
                    plumbing.q_slots)
                actual = toy.decode(row["blocks"][channel])
                self.assertAlmostEqual(actual, expected, delta=1e-9*max(1., abs(expected)))
                c_eta = toy.decode(self.node["C_eta"][0 if eta == 1 else 1])
                amplitude = c_eta*plumbing.primary(self.config["b"], self.node["momenta_geometry"])*expected
                self.assertAlmostEqual(toy.decode(row["amplitudes"][channel]), amplitude,
                                       delta=1e-9*max(abs(amplitude), 1e-20))


if __name__ == "__main__":
    unittest.main()
