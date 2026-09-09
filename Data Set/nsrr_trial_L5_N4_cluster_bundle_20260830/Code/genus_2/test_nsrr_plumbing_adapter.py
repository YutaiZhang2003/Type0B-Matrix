"""Boundary regressions; no changes to the checked conformal-block package."""
import cmath
import unittest
from unittest.mock import patch

from nsrr_plumbing_adapter import NSRRPlumbingInputs, chiral_block_in_geometry


class PlumbingAdapterTests(unittest.TestCase):
    def setUp(self):
        self.plumbing = NSRRPlumbingInputs(
            (.021+.004j, -.032+.009j, -.047-.013j), (1,-1,-1), ("R","R","NS"))

    def test_all_edge_labels_are_transported_together(self):
        self.assertEqual(self.plumbing.q_slots, tuple(reversed(self.plumbing.q_geometry)))
        self.assertEqual(self.plumbing.lifts_slots, (-1,-1,1))
        self.assertEqual(self.plumbing.momenta_slots((.52,.37,.21)), (.21,.37,.52))
        result = object()
        with patch("nsrr_double_virasoro_block.NSRRDoubleVirasoroTheta") as ctor:
            ctor.return_value.block.return_value = result
            actual = chiral_block_in_geometry(plumbing=self.plumbing, b=1.4,
                momenta_geometry=(.52,.37,.21), cutoff=2,
                form_parity=0, eta_left=1, eta_right=1)
            self.assertIs(actual, result)
            self.assertEqual(ctor.call_args.kwargs["physical_momenta"], (.21,.37,.52))
            self.assertEqual(ctor.call_args.kwargs["completion"], "none")
            self.assertEqual(ctor.return_value.block.call_args.kwargs["q_values"], self.plumbing.q_slots)
            self.assertEqual(ctor.return_value.block.call_args.kwargs["lifts"], self.plumbing.lifts_slots)

    def test_primary_uses_the_same_edge_order(self):
        b=1.4; bg=b+1/b
        hg=((1.5+3*bg*bg)/24+.52**2/2,
            (1.5+3*bg*bg)/24+.37**2/2, bg*bg/8+.21**2/2)
        expected=cmath.exp(sum(h*cmath.log(q) for h,q in zip(hg,self.plumbing.q_geometry)))
        self.assertAlmostEqual(self.plumbing.primary(b,(.52,.37,.21)), expected)

    def test_old_source_chart_cannot_be_silently_relabelled(self):
        with self.assertRaisesRegex(ValueError, "Re-plumb"):
            NSRRPlumbingInputs(self.plumbing.q_geometry, (1,1,-1), ("NS","R","R"))
        with self.assertRaisesRegex(ValueError, "three"):
            NSRRPlumbingInputs(self.plumbing.q_geometry, (1,-1), ("R","R","NS"))

    def test_geometry_entry_point_matches_independent_pbw(self):
        import sympy as sp
        from nsrr_genus2_block import HumanNSRRThetaOracle, level_triples
        from theta_star_algebra import fwht
        from nsrr_double_virasoro_block import spin_character_index, evaluate_twice_level_series

        bg=sp.Rational(7,5)+sp.Rational(5,7)
        for f,eta in ((0,-1),(1,1)):
            oracle=HumanNSRRThetaOracle(central_charge=sp.Rational(3,2)+3*bg**2,
                h_ns=bg**2/8+sp.Rational(21,100)**2/2,
                beta_r1=sp.I*sp.Rational(37,100)/sp.sqrt(2),
                beta_r2=sp.I*sp.Rational(52,100)/sp.sqrt(2),
                form_parity=f,primary_parity=0,etas=(eta,eta))
            k=spin_character_index(self.plumbing.lifts_slots)
            reference={e:fwht(oracle.coefficient_components(e[0],e[1]//2,e[2]//2))[k]
                       for e in level_triples(2)}
            expected=evaluate_twice_level_series(reference,self.plumbing.q_slots)
            actual=chiral_block_in_geometry(plumbing=self.plumbing,b=1.4,
                momenta_geometry=(.52,.37,.21),cutoff=1,
                form_parity=f,eta_left=eta,eta_right=eta)
            self.assertAlmostEqual(actual.value,expected,delta=1e-10)


if __name__ == "__main__":
    unittest.main()
