"""Measure and domain checks against independent one-dimensional integrals."""

import math
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.stats import qmc

from fivepoint_sampling import stratified_face_sample


class FaceSamplingTests(unittest.TestCase):
    def test_four_bands_cover_the_cell_with_the_correct_area(self):
        nodes, weights = np.polynomial.legendre.leggauss(256)
        for rho in (.01, .005):
            area = sum(weight/2 * stratified_face_sample((node+1)/2, .37, rho, band)[1]
                       for band in range(4) for node, weight in zip(nodes, weights))
            self.assertAlmostEqual(area, math.pi/3-math.sqrt(3)/4, delta=2e-7)

    def test_singular_annular_density_and_moments_have_correct_jacobians(self):
        nodes, weights = np.polynomial.legendre.leggauss(256)
        rho = .01
        for band, lo, hi in ((1,rho,4*rho),(2,4*rho,.5),(3,.5,1.)):
            actual = sum(weight/2 * jac / abs(z)**4
                         for node, weight in zip(nodes, weights)
                         for z,jac in [stratified_face_sample((node+1)/2,.37,rho,band)])
            def density(r):
                lower = 0 if r <= .5 else math.acos(.5/r)
                return 2*(math.acos(r/2)-lower)/r**3
            expected = quad(density,lo,hi,epsabs=1e-9,epsrel=1e-10)[0]
            self.assertLess(abs(actual-expected), 2e-7*max(1,abs(expected)))

    def test_points_stay_in_their_band_and_cell(self):
        rho = .01
        bounds = (0.,rho,4*rho,.5,1.)
        points = qmc.Sobol(d=2,scramble=True,seed=43280689).random_base2(9)
        for band in range(4):
            for u,v in points:
                z,jac = stratified_face_sample(u,v,rho,band)
                self.assertTrue(bounds[band] < abs(z) < bounds[band+1])
                self.assertTrue(0 < z.real < .5 and abs(z-1) < 1)
                self.assertTrue(math.isfinite(jac) and jac > 0)


if __name__ == '__main__':
    unittest.main()
