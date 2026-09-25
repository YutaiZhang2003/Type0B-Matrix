"""Local superdistance descent at the NS separating divisor.

This fixes the divergent even-primary boundary term, including its sign.
It does not label a full integration as BRST certified. See the accompanying
Euclidean-completion note for the assumptions about sewing and the cusp.
"""
import math
import numpy as np

from spin23_genus1_banked_geometry import theta


def superdistance_boundary(h, radius, left_level=0, right_level=0):
    r"""Berezin coefficient from Theta((z-theta1*theta2)*bar(z)-r**2).

    For z**(h-1+m) bar(z)**(h-2+n), the boundary coefficient is
    -bar(z) delta(|z|**2-r**2) times that bottom-component density.
    Angular integration projects m=n and gives -pi*r**(2*h-2+2*m).
    Levels here are integer Virasoro descendants of the even primary.
    """
    h = np.asarray(h, complex)
    if not np.isfinite(h).all() or not math.isfinite(radius) or radius <= 0:
        raise ValueError('finite weights and a positive radius required')
    if any(type(n) is not int or n < 0 for n in (left_level, right_level)):
        raise ValueError('nonnegative integer descendant levels required')
    if left_level != right_level:
        return np.zeros_like(h)
    return -math.pi*np.exp((2*h-2+2*left_level)*math.log(radius))


def picture_zero_determinants(z, tau, displacement):
    """epsilon**2 det S_spin(x_j-z_i) for x_i=z_i+epsilon.

    Every even spin should approach +1 as epsilon -> 0 at distinct punctures.
    This tests the interior PCO section, not the degenerating surface.
    """
    z, tau, eps = complex(z), complex(tau), complex(displacement)
    if tau.imag <= 0 or eps == 0 or not np.isfinite([z, tau, eps]).all():
        raise ValueError('a finite torus and nonzero PCO displacement required')
    derivative = theta(.5, .5, 0., tau, derivative=1)
    result = []
    for a, b in ((0., 0.), (0., .5), (.5, 0.)):
        constant = theta(a, b, 0., tau)
        def szego(w):
            return theta(a, b, w, tau)*derivative/(constant*theta(.5, .5, w, tau))
        result.append(eps**2*(szego(eps)**2-szego(eps-z)*szego(eps+z)))
    return np.asarray(result)


def euclidean_status(*, spectral=False, collision_overlap=False,
                     separating_descent=False, cusp_overlap=False,
                     integrated_matching=False, global_gluing=False):
    """Real-energy continuation is deliberately absent from this target."""
    checks = dict(spectral=spectral, collision_overlap=collision_overlap,
                  separating_descent=separating_descent, cusp_overlap=cusp_overlap,
                  integrated_matching=integrated_matching, global_gluing=global_gluing)
    if any(type(v) is not bool for v in checks.values()):
        raise ValueError('Boolean validation records required')
    return dict(target='imaginary-energy genus-one VV amplitude', checks=checks,
                euclidean_amplitude_certified=all(checks.values()))
