"""Compare tensor evaluation against the independent scalar PCO contraction."""
import unittest
from unittest import mock
import numpy as np
import mpmath
import type0b_ns_five_tachyon as f
from fivepoint_batch import momentum_densities, MomentumTensor


def kernel(**kwargs):
    return f.BRYNSFiveTachyonIntegrand(
        outgoing_energies=(.25+.02j,)*4, central_charge_shift=0,
        global_max_twice_levels=(8,8), global_max_total_twice_level=16,
        momentum_orders=(2,3), momentum_maximum=1, structure_precision=22,
        block_working_precision=45, **kwargs)


class BatchTests(unittest.TestCase):
    def assertClose(self, actual, expected):
        self.assertLess(abs(actual-expected), 2e-10 * max(1, abs(expected)))

    def test_all_face_orbits_and_corner_spin_lifts(self):
        k = kernel(batch_c_evaluation=True)
        pair = (.37, .71)
        for ordering, _ in f.BOUNDARY_FACE_RAISED_ORBITS:
            for q, edges in ((-.23+0j, (0,)), (.23-.07j, (1,)), (1e-5, (0,1))):
                with self.subTest(ordering=ordering, q=q, edges=edges):
                    actual = k._linear_q_primary_densities(ordering, 1e-5, q, [pair], edges)[0]
                    expected = k.linear_q_momentum_primary_density(
                        ordering=ordering, q1=1e-5, q2=q, internal_momenta=pair, boundary_edges=edges)
                    self.assertClose(actual, expected)

    def test_bulk_and_primary_integrals_at_several_moduli(self):
        k = kernel(batch_c_evaluation=True)
        ordering = (0,1,2,3,4)
        for q1, q2 in ((.18+.03j,.31-.04j), (-.12+0j,.27+.02j), (1e-5,.25+.04j)):
            positions = f._to_fixed_gauge(q1,q2,ordering)
            channel = f.linear_channel_from_ordering(positions,ordering)
            k.batch_c_evaluation = True
            actual = k.integrand_positions(positions,channel=channel)
            primary = k.linear_q_primary_density(ordering=ordering,q1=q1,q2=q2,boundary_edges=(0,))
            k.batch_c_evaluation = False
            self.assertClose(actual,k.integrand_positions(positions,channel=channel))
            self.assertClose(primary,k.linear_q_primary_density(ordering=ordering,q1=q1,q2=q2,boundary_edges=(0,)))
        self.assertGreater(k._momentum_tensor_cache.hits, 0)

    def test_face_finite_part_with_subtraction_and_collar_changes(self):
        k = kernel(batch_c_evaluation=True)
        # This face has a finite-part momentum root inside [0,1].
        for q, collar in ((.22+.05j,.01),(.25-.03j,.005)):
            args = dict(ordering=(1,2,0,3,4),remaining_modulus=q,collar_radius=collar,
                        momentum_refinement_shells=1,momentum_singularity_subtraction=True)
            k.batch_c_evaluation = True
            actual = k.boundary_face_finite_part_density(**args)
            k.batch_c_evaluation = False
            expected = k.boundary_face_finite_part_density(**args)
            self.assertClose(actual,expected)

    def test_unsafe_row_uses_scalar_reference(self):
        k=kernel(batch_c_evaluation=True)
        positions=f._to_fixed_gauge(.2+.03j,.3-.05j,(0,1,2,3,4))
        channel=f.linear_channel_from_ordering(positions,(0,1,2,3,4))
        pair=(.37,.71)
        expected=k.momentum_integrand(positions,pair,channel=channel)
        with mock.patch.object(MomentumTensor,'evaluate',return_value=(np.array([1j]),np.array([True]))):
            actual=momentum_densities(k,positions,channel,[pair])[0]
        self.assertEqual(actual,expected)
        self.assertEqual(k._momentum_tensor_cache.fallback_rows,1)

    def test_forest_remainder_against_high_precision_subtraction(self):
        k=kernel(batch_c_evaluation=True)
        pair=(.37,.71)
        for q1,q2,edges in ((1e-8,.27+.03j,(0,)),(2e-6,3e-5,(0,1))):
            positions=f._to_fixed_gauge(q1,q2,(0,1,2,3,4))
            channel=f.linear_channel_from_ordering(positions,(0,1,2,3,4))
            actual=momentum_densities(k,positions,channel,[pair],remainder_edges=edges)[0]
            with mpmath.workdps(45):
                expected=mpmath.mpc(0)
                for sector in f.ODD_SECTOR_ASSIGNMENTS:
                    v=k._sector_component_kernel(positions,pair,sector,channel)
                    for edge in edges:
                        v-=k._sector_component_kernel_boundary_primary(positions,pair,sector,channel,boundary_edge=(edge,))
                    if len(edges)==2:
                        v+=k._sector_component_kernel_boundary_primary(positions,pair,sector,channel,boundary_edge=(0,1))
                    expected+=k._structure_product(k.external_momenta,pair,sector)*v
                expected=complex(expected/mpmath.pi**2)
            self.assertLess(abs(actual-expected)/abs(expected),2e-9)

    def test_corner_finite_parts_and_face_counterterms(self):
        k=kernel(batch_c_evaluation=True)
        args=dict(ordering=(1,2,0,3,4),collar_radius=.01,
                  momentum_refinement_shells=1,momentum_singularity_subtraction=True)
        for method,extra in ((k.boundary_corner_finite_part,{}),
                             (k.boundary_corner_face_counterterm_density,dict(remaining_modulus=.002+.001j))):
            k.batch_c_evaluation=True
            actual=method(**args,**extra)
            k.batch_c_evaluation=False
            self.assertClose(actual,method(**args,**extra))

    def test_corner_counterterm_accepts_stratified_tangential_projection(self):
        k=kernel(batch_c_evaluation=True)
        # This is the scale of the v8 point that exposed the obsolete 1e-7
        # internal floor. The public configuration projection remains >=1e-7;
        # only its tangential safety cap needs a smaller auxiliary value.
        value=k.boundary_corner_face_counterterm_density(
            ordering=(1,2,0,3,4),remaining_modulus=4.526488334408974e-6,
            collar_radius=.01,projection_radius=4.526488334408974e-8,
            momentum_refinement_shells=1,momentum_singularity_subtraction=True)
        self.assertTrue(np.isfinite(value.real) and np.isfinite(value.imag))

if __name__=='__main__': unittest.main()
