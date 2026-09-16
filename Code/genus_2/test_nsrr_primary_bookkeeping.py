"""Check the public separation of reduced blocks and propagation factors."""
import cmath
import math
import unittest
from unittest.mock import patch

from nsrr_plumbing_adapter import NSRRPlumbingInputs
from nsrr_resummed_sewing import resummed_integrand
from physical_nsrr_sewing import CHANNELS, contract_physical_blocks
from nsrr_bilinear_sewing import CONVENTION, contract_nsrr_bilinear


class PrimaryBookkeepingTests(unittest.TestCase):
    def test_sector_weights_and_edge_order(self):
        q = (.021+.004j, -.032+.009j, -.047-.013j)
        plumbing = NSRRPlumbingInputs(q, (1, -1, 1), ("R", "R", "NS"))
        b, momenta = 1.4, (.52, .37, .21)
        background = b+1/b
        weights = plumbing.weights_slots(b, momenta)
        for h, p, sector in zip(weights, momenta[::-1], ("NS", "R", "R")):
            self.assertAlmostEqual(h, background**2/8+p*p/2+(sector == "R")/16)
        source = plumbing.primary(b, momenta)
        all_ns_same_q = cmath.exp(sum((background**2/8+p*p/2)*cmath.log(z)
                                     for p, z in zip(momenta, q)))
        expected = cmath.exp((cmath.log(q[0])+cmath.log(q[1]))/16)
        self.assertAlmostEqual(source/all_ns_same_q, expected)

    def test_primary_is_separate_from_every_returned_descendant_block(self):
        values = {channel: (complex(1+i/5, .3), complex(.07, -.11))
                  for i, channel in enumerate(CHANNELS)}

        class BackendFixture:
            def __init__(self, *args, **kwargs):
                pass

            def channels(self, q, **kwargs):
                return values

            @staticmethod
            def project(vector, lift):
                return vector[0]+lift[1]*vector[1]

            def diagnostics(self):
                return {"fixture": True}

        q, momenta, b = (.021+.004j, -.032+.009j, -.047-.013j), (.52, .37, .21), 1.4
        lifts, constants = ((1, 1, 1), (1, -1, 1)), (2.0, .7)
        plumbing = NSRRPlumbingInputs(q, lifts[0], ("R", "R", "NS"))
        primary = plumbing.primary(b, momenta)
        # Preserve the exact previous expression/order of operations.
        old_blocks = {
            channel: primary*sum(BackendFixture.project(vector, lift[::-1])
                                 for lift in lifts)/math.sqrt(2)
            for channel, vector in values.items()
        }
        old_result = contract_physical_blocks(old_blocks, constants)
        with patch("nsrr_resummed_sewing.ResummedNSRR", BackendFixture):
            result = resummed_integrand(
                b=b, momenta_geometry=momenta, q_geometry=q, lifts_geometry=lifts,
                bry_constants=constants, branch_level=1, recursion_order=1,
                sewing_convention="legacy-times-four")
        self.assertEqual(result["blocks"], old_blocks)
        self.assertEqual(result["integrand"], {
            name: 4*old_result[name] for name in ("total", "diagonal", "interference")})
        self.assertEqual(result["primary_prefactor"], primary)
        self.assertFalse(result["human_bilinear_pairing_verified"])
        self.assertEqual(result["sewing_convention"], "legacy Hermitian candidate times four")
        self.assertEqual(result["primary_weights_slots"], plumbing.weights_slots(b, momenta))
        self.assertEqual(result["log_q_slots"], tuple(cmath.log(z) for z in q[::-1]))
        for channel in CHANNELS:
            expected = sum(BackendFixture.project(values[channel], lift[::-1])
                           for lift in lifts)/math.sqrt(2)
            self.assertEqual(result["descendant_blocks"][channel], expected)
            self.assertAlmostEqual(result["blocks"][channel], primary*expected)

        with patch("nsrr_resummed_sewing.ResummedNSRR", BackendFixture):
            physical = resummed_integrand(
                b=b, momenta_geometry=momenta, q_geometry=q, lifts_geometry=lifts,
                bry_constants=constants, branch_level=1, recursion_order=1,
                physical_lifts_slots=(1,-1,1))
            with self.assertRaisesRegex(ValueError,"Supply physical_lifts_slots"):
                resummed_integrand(b=b,momenta_geometry=momenta,q_geometry=q,lifts_geometry=lifts,
                    bry_constants=constants,branch_level=1,recursion_order=1)
        wanted=contract_nsrr_bilinear(descendant_blocks=result['descendant_blocks'],
            antiholomorphic_blocks={k:z.conjugate() for k,z in result['descendant_blocks'].items()},
            left_bry=constants,right_bry=constants,physical_lifts_slots=(1,-1,1),
            primary=primary,antiholomorphic_primary=primary.conjugate())
        self.assertTrue(physical['human_bilinear_pairing_verified'])
        self.assertEqual(physical['sewing_convention'],CONVENTION)
        self.assertEqual(physical['descendant_blocks'],result['descendant_blocks'])
        for name in ('total','diagonal','interference'):
            self.assertAlmostEqual(physical['integrand'][name],wanted[name].real)


if __name__ == "__main__":
    unittest.main()
